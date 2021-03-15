"""
Native PyTorch LSTM implementation adapted from https://github.com/daehwannam/pytorch-rnn-util.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..ops import emulate_int
from .common import no_dropout, no_layer_norm, get_indicator, get_module_device


class LSTMFrame(nn.Module):
    def __init__(self, rnn_cells, dropout=0, batch_first=False, bidirectional=False):
        """
        :param rnn_cells: ex) [(cell_0_f, cell_0_b), (cell_1_f, cell_1_b), ..]
        :param dropout:
        :param bidirectional:
        """
        super().__init__()

        if bidirectional:
            assert all(len(pair) == 2 for pair in rnn_cells)
        elif not any(isinstance(rnn_cells[0], iterable)
                     for iterable in [list, tuple, nn.ModuleList]):
            rnn_cells = tuple((cell,) for cell in rnn_cells)

        self.rnn_cells = nn.ModuleList(nn.ModuleList(pair)
                                       for pair in rnn_cells)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = len(rnn_cells)

        if dropout > 0 and self.num_layers > 1:
            # dropout is applied to output of each layer except the last layer
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = no_dropout

    def align_sequence(self, seq, lengths, shift_right):
        """
        :param seq: (seq_len, batch_size, *)
        """
        multiplier = 1 if shift_right else -1
        example_seqs = torch.split(seq, 1, dim=1)
        max_length = max(lengths)

        # pdb.set_trace()
        shifted_seqs = [
            torch.roll(
                example_seq, 
                (max_length.item() - length.item()) * multiplier, 
                dims=0
            )
            for example_seq, length in zip(example_seqs, lengths)
        ]
        return torch.cat(shifted_seqs, dim=1)

    def forward(self, input, init_state=None, p_delta=0.0):
        """
        :param input: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: (h_0, c_0) where the size of both is (num_layers * num_directions, batch, hidden_size)
        :returns: (output, (h_n, c_n))
        - output: (seq_len, batch, num_directions * hidden_size)
        - h_n: (num_layers * num_directions, batch, hidden_size)
        - c_n: (num_layers * num_directions, batch, hidden_size)
        """
        if isinstance(input, torch.nn.utils.rnn.PackedSequence):
            input_packed = True
            input, lengths = pad_packed_sequence(input)
            if max(lengths) == min(lengths):
                uniform_length = True
            else:
                uniform_length = False
            assert max(lengths) == input.size()[0]
        else:
            input_packed = False
            lengths = [input.size()[0]] * input.size()[1]
            uniform_length = True

        if not uniform_length:
            indicator = get_indicator(lengths, device=get_module_device(self))
            # valid_example_nums = indicator.sum(0)

        if init_state is None:
            # init_state with heterogenous hidden_size
            init_hidden = init_cell = [
                torch.zeros(
                    input.size()[1],
                    self.rnn_cells[layer_idx][direction].hidden_size,
                    device=get_module_device(self))
                for layer_idx in range(self.num_layers)
                for direction in range(self.num_directions)]
            init_state = init_hidden, init_cell

        init_hidden, init_cell = init_state

        last_hidden_list = []
        last_cell_list = []

        layer_output = input

        for layer_idx in range(self.num_layers):
            layer_input = layer_output
            if layer_idx != 0:
                layer_input = self.dropout(layer_input)

            direction_output_list = []

            for direction in range(self.num_directions):
                cell = self.rnn_cells[layer_idx][direction]
                state_idx = layer_idx * self.num_directions + direction
                step_state = (init_hidden[state_idx], init_cell[state_idx])

                direction_output = torch.zeros(
                    layer_input.size()[:2] + (cell.hidden_size,),
                    device=get_module_device(self))  # (seq_len, batch_size, hidden_size)
                step_state_list = []

                if direction == 0:
                    step_input_gen = enumerate(layer_input)
                else:
                    step_input_gen = reversed(list(enumerate(
                        layer_input if uniform_length else
                        self.align_sequence(layer_input, lengths, True))))

                for seq_idx, cell_input in step_input_gen:
                    if cell.__class__ == IntLSTMCell:
                        h, c = step_state = cell(cell_input, step_state, p_delta)
                    else:
                        h, c = step_state = cell(cell_input, step_state)
                    direction_output[seq_idx] = h
                    step_state_list.append(step_state)
                if direction == 1 and not uniform_length:
                    direction_output = self.align_sequence(
                        direction_output, lengths, False)

                if uniform_length:
                    # hidden & cell's size = (batch, hidden_size)
                    direction_last_hidden, direction_last_cell = step_state_list[-1]
                else:
                    direction_last_hidden, direction_last_cell = map(
                        lambda x: torch.stack([x[length - 1][example_id]
                                               for example_id, length in enumerate(lengths)], dim=0),
                        zip(*step_state_list))

                direction_output_list.append(direction_output)
                last_hidden_list.append(direction_last_hidden)
                last_cell_list.append(direction_last_cell)

            if self.num_directions == 2:
                layer_output = torch.stack(direction_output_list, dim=2).view(
                    direction_output_list[0].size()[:2] + (-1,))
            else:
                layer_output = direction_output_list[0]

        output = layer_output
        last_hidden_tensor = torch.stack(last_hidden_list, dim=0)
        last_cell_tensor = torch.stack(last_cell_list, dim=0)

        if not uniform_length:
            # the below one line code cleans out trash values beyond the range of lengths.
            # actually, the code is for debugging, so it can be removed to enhance computing speed slightly.
            output = (
                output.transpose(0, 1) * indicator).transpose(0, 1)

        if input_packed:
            # always batch_first=False --> trick to process input regardless of batch_first option
            output = pack_padded_sequence(output, lengths, enforce_sorted=False)

        return output, (last_hidden_tensor, last_cell_tensor)


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        in_features = input_size + hidden_size 
        out_features = hidden_size * 4

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, input, state):
        hidden_tensor, cell_tensor = state

        cat_input = torch.cat([input, hidden_tensor], dim=1)
        fio_linear, u_linear = torch.split(
            F.linear(cat_input, self.weight, self.bias),
            self.hidden_size * 3, 
            dim=1
        )

        f, i, o = torch.split(torch.sigmoid(fio_linear),
                              self.hidden_size, dim=1)
        u = torch.tanh(u_linear)

        new_cell = i * u + (f * cell_tensor)
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


class IntLSTMCell(nn.Module):

    def __init__(
        self, 
        input_size, 
        hidden_size,
        p=0,
        update_step=3000,
        bits=8,
        method="histogram",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        in_features = input_size + hidden_size 
        out_features = hidden_size * 4

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        # quantization parameters
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, input, state, p_delta):
        # train with QuantNoise and evaluate the fully quantized network
        if self.training:
            p = self.p - p_delta
            if self.jitter:
                downside = 0.25 * p
                upside = 0.5 * p
                rand_val = torch.rand(1).item()
                p -= downside
                p += upside * rand_val
        else:
            p = 1

        # update parameters every <update_step> iterations
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1

        # quantize weight
        if not self.training:
            breakpoint()
        weight_quantized, self.scale, self.zero_point = emulate_int(
            self.weight.detach(),
            bits=self.bits,
            method=self.method,
            scale=self.scale,
            zero_point=self.zero_point,
        )

        # mask to apply noise
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)

        # using straight-through estimator (STE)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = (
            torch.clamp(self.weight, clamp_low.item(), clamp_high.item())
            + noise.detach()
        )

        hidden_tensor, cell_tensor = state

        cat_input = torch.cat([input, hidden_tensor], dim=1)
        fio_linear, u_linear = torch.split(
            F.linear(cat_input, weight, self.bias),
            self.hidden_size * 3, 
            dim=1
        )

        f, i, o = torch.split(
            torch.sigmoid(fio_linear),
            self.hidden_size, 
            dim=1
        )
        u = torch.tanh(u_linear)

        new_cell = i * u + (f * cell_tensor)
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell

import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import (
    AverageMeter,
    save_checkpoint as save_snapshot,
    copy_checkpoint as copy_snapshot,
)
from src.datasets.harper_valley import HarperValley
from src.models.ctc import (
    ConnectionistTemporalClassification,
    GreedyDecoder,
)
from src.models.tasks import (
    TaskTypePredictor
)
import pytorch_lightning as pl

torch.autograd.set_detect_anomaly(True)


class CTC_System(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.create_model()

    def create_datasets(self):
        wav_maxlen = self.config.data_params.wav_maxlen
        transcript_maxlen = self.config.data_params.transcript_maxlen
        root = self.config.data_params.harpervalley_root
        train_dataset = HarperValley(
            root,
            split='train',
            add_eps_tok=True,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            split_by_speaker=self.config.data_params.speaker_split,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        val_dataset = HarperValley(
            root,
            split='val',
            add_eps_tok=True,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=self.config.data_params.n_mels,
            split_by_speaker=self.config.data_params.speaker_split,
            min_utterance_length=self.config.data_params.min_utterance_length,
            min_speaker_utterances=self.config.data_params.min_speaker_utterances,
        )
        return train_dataset, val_dataset

    def create_asr_model(self):
        asr_model = ConnectionistTemporalClassification(
            self.train_dataset.input_dim,
            self.train_dataset.num_class,
            num_layers=self.config.model_params.num_layers,
            hidden_dim=self.config.model_params.hidden_dim,
            bidirectional=self.config.model_params.bidirectional,
        )
        self.asr_model = asr_model.to(self.device)
        self.embedding_dim = asr_model.embedding_dim

    def create_auxiliary_models(self):
        task_type_model = TaskTypePredictor(
            self.embedding_dim,
            self.train_dataset.task_type_num_class,
        )
        self.task_type_model = task_type_model.to(self.device)

    def create_model(self):
        self.create_asr_model()
        self.create_auxiliary_models()

    def configure_optimizers(self):
        parameters = chain(
            self.asr_model.parameters(),
            self.task_type_model.parameters(),
        )
        optim = torch.optim.AdamW(
            parameters,
            lr=self.config.optim_params.learning_rate,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    def get_asr_loss(self, log_probs, labels, input_lengths, label_lengths):
        loss = self.asr_model.get_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            blank=self.train_dataset.eps_index,
        )
        return loss

    def get_asr_decode_error(self, log_probs, input_lengths, labels, label_lengths):
        ctc_decoder = GreedyDecoder(
            blank_index=self.train_dataset.eps_index,
            space_index=-1,  # no space label
        )
        wer = ctc_decoder.get_wer(
            log_probs,
            input_lengths,
            labels,
            label_lengths,
        )
        return wer

    def forward(self, inputs, input_lengths, labels, label_lengths):
        log_probs, embedding = self.asr_model(inputs, input_lengths)
        return log_probs, embedding

    def get_losses_and_metrics_for_batch(self, batch, train=True):
        indices = batch[0].to(self.device)
        char_inputs = batch[1].to(self.device)
        char_input_lengths = batch[2].to(self.device)
        char_labels = batch[3].to(self.device)
        char_label_lengths = batch[4].to(self.device)
        task_type_labels = batch[5].to(self.device)
        batch_size = indices.size(0)

        char_log_probs, embedding = self.forward(char_inputs, char_input_lengths,
                                                 char_labels, char_label_lengths)
        task_type_log_probs = self.task_type_model(embedding)

        # asr_loss = self.get_asr_loss(
        #     char_log_probs,
        #     char_labels,
        #     char_input_lengths,
        #     char_label_lengths,
        # )
        task_type_loss = self.task_type_model.get_loss(
            task_type_log_probs,
            task_type_labels,
        )
        loss = task_type_loss

        with torch.no_grad():
            # asr_wer = self.get_asr_decode_error(
            #     char_log_probs,
            #     char_input_lengths,
            #     char_labels,
            #     char_label_lengths,
            # )

            task_type_preds = torch.argmax(task_type_log_probs, dim=1)
            num_task_type_correct = (task_type_preds == task_type_labels).sum().item()
            num_task_type_total = batch_size

            prefix = 'train' if train else 'val'

            metrics = {
                f'{prefix}_task_type_loss': task_type_loss,
                f'{prefix}_num_task_type_correct': num_task_type_correct,
                f'{prefix}_num_task_type_total': num_task_type_total,
            }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=True)
        return {'loss': loss, 'log': metrics}

    def training_epoch_end(self, outputs):
        loss_batches = [out['loss'] for out in outputs]
        avg_train_loss = torch.stack(loss_batches).mean()
        
        return {
            'log': {
                'train_loss': avg_train_loss,
            }
        }

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.get_losses_and_metrics_for_batch(batch, train=False)
        metrics['val_loss'] = loss
        return OrderedDict(metrics)

    def validation_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            if key not in ['dialog_acts_preds_npy', 'dialog_acts_labels_npy']:
                metrics[key] = torch.tensor([elem[key]
                                            for elem in outputs]).float().mean()
        metric_keys = ['task_type']

        for name in metric_keys:
            num_correct = sum([out[f'val_num_{name}_correct'] for out in outputs])
            num_total = sum([out[f'val_num_{name}_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            metrics[f'val_{name}_acc'] = val_acc

        return {'val_loss': metrics['val_loss'], 'log': metrics}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)

def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        num_workers=config.data_loader_workers
    )
    return loader

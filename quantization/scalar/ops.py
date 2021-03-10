# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import logging

def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f"emulate_int{bits}_{method}"]
    return q(w, scale=scale, zero_point=zero_point)


def quantize(w, scale, zero_point, size=8):
    if size == 8:
        return (
            torch.clamp(torch.round(w / scale + zero_point), 0, 255) - zero_point
        ) * scale
    elif size == 4:
        return (
            torch.clamp(torch.round(w / scale + zero_point), 0, 127) - zero_point
        ) * scale
    elif size == 1:
        return (
            torch.clamp(torch.round(w / scale + zero_point), 0, 1) - zero_point
        ) * scale
    raise ValueError(f"Invalid size provided {size}")

def emulate_int1_histogram(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.HistogramObserver()
        obs.quant_min, obs.quant_max = 0, 1
        obs.has_customized_qrange = True
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, size=1), scale, zero_point

def emulate_int1_channel(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        obs.quant_min, obs.quant_max = 0, 1
        obs.has_customized_qrange = True
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, size=1), scale, zero_point

def emulate_int1_tensor(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.MinMaxObserver()
        obs.quant_min, obs.quant_max = 0, 1
        obs.has_customized_qrange = True
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, size=1), scale, zero_point

def emulate_int4_histogram(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.HistogramObserver()
        obs.quant_min, obs.quant_max = 0, 127
        obs.has_customized_qrange = True
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, size=4), scale, zero_point


def emulate_int4_channel(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        obs.quant_min, obs.quant_max = 0, 127
        obs.has_customized_qrange = True
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, size=4), scale, zero_point

def emulate_int4_tensor(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.MinMaxObserver()
        obs.quant_min, obs.quant_max = 0, 127
        obs.has_customized_qrange = True
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point, size=4), scale, zero_point

def emulate_int8_histogram(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.HistogramObserver()
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


def emulate_int8_channel(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


def emulate_int8_tensor(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.MinMaxObserver()
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point

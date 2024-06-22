# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import numpy as np
import math
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib


# ================================== #
# Quantization and noise adding
# ================================== #
# Quantize input data
def data_quantization_sym(data_float, half_level = 15, scale = None, isint = 0, clamp_std = None):
    # data_float -> Input data needs to be quantized
    # half_level -> Quantization levels. Total levels = [-half_level, 0, half_level]
    # scale -> Define a scale. The quantized value would range from [- data_float / scale, data_float / scale]
    # isint ->
    #   isint = 1 -> return quantized values as integer levels
    #   isint = 0 -> return quantized values as float numbers with the same range as input
    # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]

    if half_level <= 0:
        return data_float, 0

    std = data_float.std()

    if clamp_std != None and clamp_std != 0:
        data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)

    if scale == None or scale == 0:
        scale = abs(data_float).max()

    if scale == 0:
        return data_float, 0

    data_quantized = (data_float / scale * half_level).round()

    if isint == 0:
        data_quantized = data_quantized * scale / half_level

    return data_quantized, scale


# Add noise to input data
def add_noise(weight, method = 'add', n_scale = 0.074, n_range = 'max'):
    # weight -> input data, usually a weight
    # method ->
    #   'add' -> add a Gaussian noise to the weight, preferred method
    #   'mul' -> multiply a noise factor to the weight, rarely used
    # n_scale -> noise factor
    # n_range ->
    #   'max' -> use maximum range of weight times the n_scale as the noise std, preferred method
    #   'std' -> use weight std times the n_scale as the noise std, rarely used
    std = weight.std()
    w_range = weight.max() - weight.min()

    if n_range == 'max':
        factor = w_range
    if n_range == 'std':
        factor = std

    if method == 'add':
        w_noise = factor * n_scale * torch.randn_like(weight)
        weight_noise = weight + w_noise
    if method == 'mul':
        w_noise = torch.randn_like(weight) * n_scale + 1
        weight_noise = weight * w_noise
    return weight_noise


# ================================== #
# Autograd Functions
# ================================== #
class Round_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.round()
        # ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Quantize weight and add noise
class Weight_Quant_Noise(torch.autograd.Function):
    # Number of inputs (excluding ctx, only weight, bias, half_level, isint, clamp_std, noise_scale)
    # for forward need to be the same as the number of return in def backward()
    # (return weight_grad, bias_grad, None, None, None, None)
    @staticmethod
    def forward(ctx, weight, bias, half_level, isint, clamp_std, noise_scale):
        # weight -> input weight
        # bias -> input bias
        # half_level -> quantization level
        # isint -> return int (will result in scaling) or float (same scale)
        # clamp_std -> clamp weight to [- std * clamp_std, std * clamp_std]
        # noise_scale -> noise scale, equantion can be found in add_noise()
        ctx.save_for_backward()

        std = weight.std()
        if clamp_std != 0:
            weight = torch.clamp(weight, min = -clamp_std * std, max = clamp_std * std)

        # log down the max scale for input weight
        scale_in = abs(weight).max()

        # log down the max scale for input weight
        weight, scale = data_quantization_sym(weight, half_level, scale = scale_in,
                                              isint = isint, clamp_std = 0)
        # add noise to weight
        weight = add_noise(weight, n_scale = noise_scale)

        # No need for bias quantization, since the bias is added to the feature map on CPU (or GPU)
        if bias != None:
            bias /= scale

        return weight, bias

    # Use default gradiant to train the network
    # Number of inputs (excluding ctx, only weight_grad, bias_grad) for backward need to be the same as the
    # number of return in def forward() (return weight, bias)
    @staticmethod
    def backward(ctx, weight_grad, bias_grad):
        return weight_grad, bias_grad, None, None, None, None


# Quantize feature map
class Feature_Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature, half_level, isint):

        feature_q, _ = data_quantization_sym(feature, half_level, scale = None, isint = isint, clamp_std = 0)
        return feature_q

    @staticmethod
    def backward(ctx, feature_grad):
        return feature_grad, None, None

# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
# A quantization layer
class Layer_Quant(nn.Module):
    def __init__(self, bit_level, isint, clamp_std):
        super().__init__()
        self.isint = isint
        self.output_half_level = 2 ** bit_level / 2 - 1
        self.clamp_std = clamp_std

    def forward(self, x):
        x = Feature_Quant.apply(x, self.output_half_level, self.isint, self.clamp_std)
        return x

# A convolution layer which adds noise and quantize the weight and output feature map
class Conv2d_quant_noise(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 noise_scale,
                 bias = False,
                 ):
        # weight_bit -> bit level for weight
        # output_bit -> bit level for output feature map
        # isint, clamp_std, noise_scale -> same arguments as Weight_Quant_Noise()
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         bias = False,
                         )
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale

    def forward(self, x):
        # quantize weight and add noise first
        weight_q, bias_q = Weight_Quant_Noise.apply(self.weight, self.bias,
                                                    self.weight_half_level, self.isint, self.clamp_std,
                                                    self.noise_scale)
        # calculate the convolution next
        x = self._conv_forward(x, weight_q, bias_q)

        # quantize the output feature map at last
        x = Feature_Quant.apply(x, self.output_half_level, self.isint)

        return x


# A fully connected layer which adds noise and quantize the weight and output feature map
# See notes in Conv2d_quant_noise
class Linear_quant_noise(nn.Linear):
    def __init__(self, in_features, out_features,
                 weight_bit,
                 output_bit,
                 isint,
                 clamp_std,
                 noise_scale,
                 bias = False, ):
        super().__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.output_bit = output_bit
        self.isint = isint
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.output_half_level = 2 ** output_bit / 2 - 1

    def forward(self, x):
        weight_q, bias_q = Weight_Quant_Noise.apply(self.weight, self.bias,
                                                    self.weight_half_level, self.isint, self.clamp_std,
                                                    self.noise_scale)
        x = F.linear(x, weight_q, bias_q)
        x = Feature_Quant.apply(x, self.output_half_level, self.isint)
        return x
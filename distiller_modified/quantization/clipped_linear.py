#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import torch.nn as nn

from .quantizer import Quantizer
from .q_utils import *
import logging
msglogger = logging.getLogger()

import numpy as np
import pickle
import csv

B_W = 8
B_X = 8
N_mult = 8
B_ADC = 12
B_lost = B_W-1 + B_X-1 + np.log2(N_mult) - B_ADC


###
# Clipping-based linear quantization (e.g. DoReFa, WRPN)
###


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale_factor, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None


class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale_factor = symmetric_linear_quantization_scale_factor(num_bits, clip_val.data[0]) # Use of asymmetric scale factor assumes hw can repurpose sign bit; use symmetric scale factor instead
        output = clamp(input, 0, clip_val.data[0], inplace)
        output = linear_quantize(output, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(0)] = 0
        grad_input[input.ge(clip_val.data[0])] = 0

        grad_alpha = grad_output.clone()
        grad_alpha[input.lt(clip_val.data[0])] = 0
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, None, None, None


class ClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        super(ClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale_factor = symmetric_linear_quantization_scale_factor(num_bits, clip_val) # Use of asymmetric scale factor assumes hw can repurpose sign bit; use symmetric scale factor instead
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        input = LinearQuantizeSTE.apply(input, self.scale_factor, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val, dequantize=True, inplace=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = LearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)

class MixedSignalErrorWrapper(nn.Module): # ADDED CLASS
    """
    Base class for module that wraps an existing Conv2D module with addition of mixed-signal error

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
    """
    def __init__(self, wrapped_module, name, bits_acts):
        super(MixedSignalErrorWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.name = name
        self.bits_acts = bits_acts

    def forward(self, *inputs):
        # If first layer, normalize and quantize inputs
        if self.name == 'module.conv1':
            input_quant_scale_factor = symmetric_linear_quantization_scale_factor(self.bits_acts, inputs[0].abs().max())
            input_dequant_scale_factor = symmetric_linear_quantization_scale_factor(self.bits_acts, 1)
            linear_quantize(inputs[0], input_quant_scale_factor, inplace=True)
            linear_dequantize(inputs[0], input_dequant_scale_factor, inplace=True)

        # Forward through wrapped module
        accum = self.wrapped_module.forward(*inputs)

        # Add mixed-signal error to fixed-point accumulation
        if 'Conv2d' in str(self.wrapped_module):
            N_tot = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1] * self.wrapped_module.in_channels
        elif 'Linear' in str(self.wrapped_module):
            N_tot = self.wrapped_module.in_features
        stddev = np.sqrt(N_tot/N_mult * 1/12) * 2**B_lost * 2**(-(B_W-1 + B_X-1))
        accum.add_(torch.cuda.FloatTensor(accum.size()).normal_(0,stddev))

        return torch.autograd.Variable(accum, requires_grad=True)

class BNLogWrapper(nn.Module): # ADDED CLASS
    """
    Base class for module that wraps an existing BN module with parameter logging

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
    """
    def __init__(self, wrapped_module, name):
        super(BNLogWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.name = name

    def forward(self, *inputs):
        # Log BN parameters

        # Forward through wrapped module
        BN_out = torch.autograd.Variable(self.wrapped_module.forward(*inputs), requires_grad=True)
        out = BN_out.clone()

        return out

class WRPNQuantizer(Quantizer):
    """
    Quantizer using the WRPN quantization scheme, as defined in:
    Mishra et al., WRPN: Wide Reduced-Precision Networks (https://arxiv.org/abs/1709.01134)

    Notes:
        1. This class does not take care of layer widening as described in the paper
        2. The paper defines special handling for 1-bit weights which isn't supported here yet
    """
    def __init__(self, model, optimizer, bits_activations=32, bits_weights=32, bits_overrides=OrderedDict(),
                 quantize_bias=False):
        super(WRPNQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_overrides=bits_overrides,
                                            train_with_fp_copy=True, quantize_bias=quantize_bias)

        def wrpn_quantize_param(param_fp, num_bits):
            scale_factor = symmetric_linear_quantization_scale_factor(num_bits, 1)
            out = param_fp.clamp(-1, 1)
            out = LinearQuantizeSTE.apply(out, scale_factor, True, False)
            return out

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        self.param_quantization_fn = wrpn_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn


def dorefa_quantize_param(param_fp, num_bits):
    scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, 1) # Use of asymmetric scale factor is correct
    out = param_fp.tanh()
    out = out / (2 * out.abs().max()) + 0.5
    out = LinearQuantizeSTE.apply(out, scale_factor, True, False)
    out = 2 * out - 1
#    out = param_fp
    return out


class DorefaQuantizer(Quantizer):
    """
    Quantizer using the DoReFa scheme, as defined in:
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
    (https://arxiv.org/abs/1606.06160)

    Notes:
        1. Gradients quantization not supported yet
        2. The paper defines special handling for 1-bit weights which isn't supported here yet
    """
    def __init__(self, model, optimizer, bits_activations=32, bits_weights=32, bits_overrides=OrderedDict(),
                 quantize_bias=False):
        super(DorefaQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                              bits_weights=bits_weights, bits_overrides=bits_overrides,
                                              train_with_fp_copy=True, quantize_bias=quantize_bias)

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return ClippedLinearQuantization(bits_acts, 1, dequantize=True, inplace=module.inplace)

        def conv_lin_replace_fn(module, name, qbits_map): # ADDED FUNCTION
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return MixedSignalErrorWrapper(module, name, bits_acts)

        def bn_replace_fn(module, name, qbits_map): # ADDED FUNCTION
            return BNLogWrapper(module, name)

        self.param_quantization_fn = dorefa_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn
        self.replacement_factory[nn.Conv2d] = conv_lin_replace_fn # ADDED LINE
#        self.replacement_factory[nn.Linear] = conv_lin_replace_fn # ADDED LINE
#        self.replacement_factory[nn.BatchNorm2d] = bn_replace_fn # ADDED LINE
 
class PACTQuantizer(Quantizer): # Note: error injection magnitude not correct for PACT (does not take alpha into account)
    """
    Quantizer using the PACT quantization scheme, as defined in:
    Choi et al., PACT: Parameterized Clipping Activation for Quantized Neural Networks
    (https://arxiv.org/abs/1805.06085)

    Args:
        act_clip_init_val (float): Initial clipping value for activations, referred to as "alpha" in the paper
            (default: 8.0)
        act_clip_decay (float): L2 penalty applied to the clipping values, referred to as "lambda_alpha" in the paper.
            If None then the optimizer's default weight decay value is used (default: None)
    """
    def __init__(self, model, optimizer, bits_activations=32, bits_weights=32, bits_overrides=OrderedDict(),
                 quantize_bias=False, act_clip_init_val=8.0, act_clip_decay=None):
        super(PACTQuantizer, self).__init__(model, optimizer=optimizer, bits_activations=bits_activations,
                                            bits_weights=bits_weights, bits_overrides=bits_overrides,
                                            train_with_fp_copy=True, quantize_bias=quantize_bias)

        def relu_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return LearnedClippedLinearQuantization(bits_acts, act_clip_init_val, dequantize=True,
                                                    inplace=module.inplace)

        def conv_replace_fn(module, name, qbits_map): # ADDED FUNCTION
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return MixedSignalErrorWrapper(module)

        self.param_quantization_fn = dorefa_quantize_param

        self.replacement_factory[nn.ReLU] = relu_replace_fn
        self.replacement_factory[nn.Conv2d] = conv_replace_fn # ADDED LINE

        self.act_clip_decay = act_clip_decay

    # In PACT, LearnedClippedLinearQuantization is used for activation, which contains a learnt 'clip_val' parameter
    # We optimize this value separately from the main model parameters
    def _get_updated_optimizer_params_groups(self):
        base_group = {'params': [param for name, param in self.model.named_parameters() if 'clip_val' not in name]}
        clip_val_group = {'params': [param for name, param in self.model.named_parameters() if 'clip_val' in name]}
        if self.act_clip_decay is not None:
            clip_val_group['weight_decay'] = self.act_clip_decay
        return [base_group, clip_val_group]

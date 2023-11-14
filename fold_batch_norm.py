# This file is part of PWLQ repository.
# Copyright (c) Samsung Semiconductor, Inc.
# All rights reserved.

import torch

##########################################################################################
####  Fold Batch Normalization
##########################################################################################

# References: 
#   https://arxiv.org/abs/1712.05877 

# before folding:
#   x_bn_out = gamma * (x_bn_in - mean) / sqrt(var + eps) + beta
#   x_bn_in = conv(w, x_conv_in) with weights w and bias b = 0
# 
# after folding (remove bn):
#   x_bn_out = conv(w_new, x_conv_in) with new weights w_new and new bias b_new
#   w_new = gamma * w / sqrt(var + eps)
#   b_new = beta - gamma * mean / sqrt(var + eps)
# 
# after folding (remain bn):
#   x_bn_in = conv(w_new, x_conv_in) with new weights w_new = gamma * w / sqrt(var + eps) and bias b = 0
# new bn variables: 
#   beta_new = beta - gamma * mean / sqrt(var + eps)
#   gamma_new = 1, mean_new = 0, var_new = 1 - eps
# then: 
#   x_bn_out = gamma_new * (x_bn_in - mean_new) / sqrt(var_new + eps) + beta_new 


def fold_batch_norm(checkpoint, arch='resnet50'):
    print('folding BN laryers for %s ... ' % arch)
    weight_layers, bn_layer_counts = [], 0
    layers_list = list(checkpoint.keys())

    # ksh: 'layers_list' example in resnet50
    # conv1.weight
    # bn1.weight
    # bn1.bias
    # bn1.running_mean
    # bn1.running_var
    # bn1.num_batches_tracked
    # layer1.0.conv1.weight
    # layer1.0.bn1.weight
    # layer1.0.bn1.bias
    # layer1.0.bn1.running_mean
    # layer1.0.bn1.running_var
    # layer1.0.bn1.num_batches_tracked
    # ... 
    # layer1.0.conv3.weight
    # layer1.0.bn3.weight
    # layer1.0.bn3.bias
    # layer1.0.bn3.running_mean
    # layer1.0.bn3.running_var
    # layer1.0.bn3.num_batches_tracked
    # layer1.0.downsample.0.weight
    # layer1.0.downsample.1.weight
    # layer1.0.downsample.1.bias
    # layer1.0.downsample.1.running_mean
    # layer1.0.downsample.1.running_var
    # layer1.0.downsample.1.num_batches_tracked
    # ... (more layers)

    # ksh: bn_base
    # layer1.0.bn1.running_mean -> layer1.0.bn1
    # layer1.0.downsample.1.running_mean -> layer1.0.downsample.1
    
    # ksh: conv_layer
    # layer1.0.bn1.running_mean -> layer1.0.conv1.weight
    # layer1.0.downsample.1.running_mean -> layer1.0.downsample.0.weight
    
    if arch == 'resnet50':
        var_eps  = 1e-5
        for layer in layers_list:
            if '.running_mean' in layer:
                bn_base = layer.replace('.running_mean', '')    # ksh: ex) layer1.0.bn1.running_mean -> layer1.0.bn1 
                if 'downsample' in layer:
                    conv_layer_num = int(bn_base.split('.')[-1]) - 1    # ksh: ex) layer1.0.downsample.1 -> conv_layer_num = 0
                    conv_layer = '.'.join(bn_base.split('.')[:-1] + [str(conv_layer_num), 'weight'])    # ksh: bn_base.split('.')[:-1] = [layer1, 0, downsample] 
                else:
                    conv_layer = bn_base.replace('bn', 'conv') + '.weight'    # ksh: ex) layer1.0.bn1 -> layer1.0.conv1.weight
                weight_layers.append(conv_layer)
                fold_batch_norm_for_one_layer(checkpoint, conv_layer, bn_base, var_eps)
                bn_layer_counts += 1
        print('conv and batch normalization layers: ', bn_layer_counts)
        assert(bn_layer_counts == 53)
        weight_layers.append('fc.weight')
    else:
        raise RuntimeError("Please implement BatchNorm folding for %s !!!" % arch) 

    return checkpoint, weight_layers


def fold_batch_norm_for_one_layer(checkpoint, conv_layer, bn_base, var_eps=1e-5):
    conv_weights = checkpoint[conv_layer].clone()

    bn_gamma = checkpoint[bn_base + '.weight'].clone()
    bn_beta = checkpoint[bn_base + '.bias'].clone()
    bn_mean = checkpoint[bn_base + '.running_mean'].clone()
    bn_var = checkpoint[bn_base + '.running_var'].clone()

    # x_bn_in = conv(w_new, x_conv_in) with new weights w_new = gamma * w / sqrt(var + eps) and bias b=0
    for c in range(conv_weights.size()[0]):
        conv_weights[c, :, :, :] *= bn_gamma[c] * torch.rsqrt(torch.add(bn_var[c], var_eps))            
    checkpoint[conv_layer] = conv_weights
    
    # new bn variables: beta_new = beta - gamma * mean / sqrt(var + eps), gamma_new = 1, mean_new = 0, var_new = 1 - eps
    checkpoint[bn_base + '.bias'] = bn_beta - bn_gamma * bn_mean * torch.rsqrt(torch.add(bn_var, var_eps))
    checkpoint[bn_base + '.weight'] = bn_gamma * 0.0 + 1.0
    checkpoint[bn_base + '.running_mean'] = bn_mean * 0.0
    checkpoint[bn_base + '.running_var'] = bn_var * 0.0 + 1.0 - var_eps 

    return checkpoint

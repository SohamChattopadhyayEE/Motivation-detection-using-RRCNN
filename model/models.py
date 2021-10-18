import torch
import torch.nn as nn

from model import RRCNN_C

def model_version(num_channels, num_classes, num_residual_features, num_resedual_blocks, model = 'RRCNN_C'):
    if model == 'RRCNN_C':
        return RRCNN_C(num_channels = num_channels,  num_classes = num_classes, 
                num_res_ft = num_residual_features, num_res = num_resedual_blocks)
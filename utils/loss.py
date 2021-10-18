import torch
import torch.nn as nn

def loss_function(loss_fun = 'Cross entropy loss'):
    if loss_fun == 'Cross entropy loss':
        return nn.CrossEntropyLoss()
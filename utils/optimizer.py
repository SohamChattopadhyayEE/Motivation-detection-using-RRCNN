import torch 
import torch.nn as nn

def optimizer_function(parameters, lr = 0.001, momentum = 0.58, optm = 'Adam'):
    if optm == 'Adam' : 
        optimizer = torch.optim.Adam(parameters, lr, momentum = momentum)
    elif optm == 'SGD' : 
        optimizer = torch.optim.SGD(parameters, lr, momentum = momentum)
    elif optm == 'RMSProp' : 
        optimizer = torch.optim.RMSprop(parameters, lr, momentum = momentum)
    else :
        optimizer = torch.optim.Adagrad(parameters, lr, momentum = momentum)
    return optimizer
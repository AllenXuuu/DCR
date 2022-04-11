import torch
import torch.nn as nn
from .orderNet import orderNet
from .DCR import DCR

def build_model(config, dataset):
    if config.name == 'DCR':
        model = DCR(
            config, dataset
        )
    elif config.name == 'orderNet':
        model = orderNet(
            config, dataset
        )
    else:
        raise NotImplementedError(config.name)

    return model

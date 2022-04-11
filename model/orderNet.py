from os import confstr
from typing import OrderedDict
import torch
import torch.nn as nn
from termcolor import colored
from .transformer import transformer
from .model_utils import CosineClassifier,CrossEntropy
import math

class orderNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        
        self.config = config
        self.full_frame = self.config.past_frame + self.config.anticipation_frame + self.config.action_frame
        self.encoder = nn.Linear(config.feat_dim, config.reasoner.d_model)
        assert config.reasoner.name == 'transformer'
        self.reasoner = transformer(config.reasoner, self.full_frame)
        self.classifier = CosineClassifier(config.reasoner.d_model, self.full_frame)
        self.loss_func = CrossEntropy(smooth=config.loss.smooth)


        if self.config.loss.sigma is None or self.config.loss.sigma == 0:
            self.target = torch.eye(self.full_frame)
        else:
            print('Gassian Smooth Label')
            self.target = torch.zeros((self.full_frame,self.full_frame))
            for i in range(self.target.shape[0]):
                for j in range(self.target.shape[1]):
                    self.target[i,j] = math.exp(- ( (i-j) / self.config.loss.sigma) ** 2 )

        self.target =nn.Parameter(self.target, requires_grad=False)
        
    def forward(
        self,
        batch,
        is_training=False):
        
        frames = batch['past_frame']
        bz, cur_frame, dim = frames.shape

        assert cur_frame == self.full_frame 
        
        out_frames = self.reasoner(self.encoder(frames))
        permutation_logits = self.classifier(out_frames)
        if not is_training:
            return permutation_logits
        else:
            target = self.target.expand(bz,   self.full_frame,   self.full_frame).reshape(-1,   self.full_frame)
            permutation_logits = permutation_logits.reshape(-1,   self.full_frame)
            loss_dict = {
                'loss_total' : self.loss_func(permutation_logits,     target)
            }
            return loss_dict
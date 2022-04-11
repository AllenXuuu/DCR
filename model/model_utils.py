import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F
import numpy as np

class CosineClassifier(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim,out_dim))
        
    def l2_normalize(self,x):
        x = x/torch.sum(x**2, dim=-1, keepdim=True).sqrt()
        return x

    def forward(self, x):
        t = x.shape[1]
        assert len(x.shape) == 3 and t == self.weight.shape[1]
        x = self.l2_normalize(x)
        w = self.l2_normalize(self.weight)
        return x @ w 


class ClassifierWithLoss(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[],dropout = 0., loss_smooth = 0,   loss_weight  = None):
        super().__init__()
        self.layers = []     
        for i,dim in enumerate(hiddens):
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_dim,dim))
            self.layers.append(nn.ReLU())
            in_dim = dim
        if dropout > 0:
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(in_dim,out_dim))
        self.layers = nn.Sequential(*self.layers)
        self.loss_func = CrossEntropy(loss_smooth,loss_weight)

    def forward(self,x,y = None):
        if y is None:
            return self.layers(x)
        else:
            x = x.reshape(-1,x.shape[-1])
            y = y.flatten()
            mask = (y>=0)
            x,y = x[mask], y[mask]
            y_pred = self.layers(x)
            loss = self.loss_func(y_pred,y)
            return loss



class CrossEntropy(nn.Module):
    def __init__(self,smooth = 0.,weight = None):
        super().__init__()
        self.smooth = smooth
        self.weight = weight

        if self.weight is None:
            pass
        elif isinstance(self.weight, str):
            self.weight = nn.Parameter(
                torch.tensor(pkl.load(open(weight,'rb'))).float(), requires_grad=False)
        elif isinstance(self.weight, np.ndarray):
            self.weight = nn.Parameter(
                torch.tensor(self.weight).float(), requires_grad=False)
        elif isinstance(self.weight, torch.Tensor):
            self.weight = nn.Parameter(self.weight.float(), requires_grad=False)
        else:
            raise TypeError(weight)

    def forward(self,x,target):
        if target.shape!= x.shape:
            target = F.one_hot(target,num_classes = x.shape[-1])
        if self.weight is not None:
            target = target * self.weight
        if self.smooth > 0:
            num_cls = x.shape[-1]
            target = target * (1-self.smooth) + self.smooth/num_cls
        loss = - target * F.log_softmax(x, dim=-1)
        return loss.mean(0).sum()

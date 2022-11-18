import pickle as pkl
import numpy as np
from sklearn.metrics import recall_score
import pandas as pd
import os
import json
import torch 
import torch.nn as nn
 
root = '.'
ek55path = [
    ['rgbtsn'   , './EK55RGBTSN.pt.valid.pkl'],
    ['rgbcsn'   , './EK55RGBCSN.pt.valid.pkl'],
    ['rgbtsm'   , './EK55RGBTSM.pt.valid.pkl'],
    ['objfrcnn' , './EK55OBJFRCNN.pt.valid.pkl'],
]
weights = [1,1,1,1]
 
def path2res(x):
    pred = []
    keys, res = zip(*[[k, pkl.load(open(os.path.join(root,v),'rb'))] for k,v in x])
    for r in res:
        assert np.all(r['gt']['id'] == res[0]['gt']['id'])
        assert np.all(r['gt']['action'] == res[0]['gt']['action'])
        pred.append(r['pred']['action'] )
    gt = res[0]['gt']['action']
    idx = res[0]['gt']['id']
    return keys,pred,gt,idx


def fuse(pred,weight):
    assert len(pred) == len(weight)
    return sum([p*w for p, w in zip(pred,weight)])
 
 
keys,pred,gt,idx = path2res(ek55path)
 
fusion = fuse(pred,weights)
rank = np.argsort(fusion,axis=1)[:,::-1]
 
t1_acc = (rank[:,:1] == np.expand_dims(gt,1)).sum(1).mean()
t5_acc = (rank[:,:5] == np.expand_dims(gt,1)).sum(1).mean()
print('T1 %.3f T5 %.3f' % (t1_acc, t5_acc))
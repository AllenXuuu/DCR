import pickle as pkl
import numpy as np
from sklearn.metrics import recall_score
import pandas as pd
import os
import json
import torch 
import torch.nn as nn
 
 
root = '.'
ek100path = [
    ['rgbtsn'   , 'EK100RGBTSN.pt.valid.pkl'],
    ['rgbtsm'   , 'EK100RGBTSM.pt.valid.pkl'],
    ['objfrcnn' , 'EK100OBJFRCNN.pt.valid.pkl'],
]
weights = [1,1,1]
 
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

def compute_recall(rank,gt_class,classes,topk=5):
    if len(classes) == 0:
        return 0
    is_topk_correct = np.any(rank[:,:topk] == np.expand_dims(gt_class,1),axis=1)
    recall = recall_score(
        y_true = gt_class,
        y_pred = np.where(is_topk_correct,gt_class,rank[:,0]),
        labels = classes,
        average = None
    )
    return np.nanmean(recall) 

keys,pred,gt,idx = path2res(ek100path)
fusion = fuse(pred,weights)
 

rank = np.argsort(fusion,axis=1)[:,::-1]
overall = compute_recall(rank,gt,np.unique(gt))

tail_class_verbs = pd.read_csv("./data/EK100/EPIC_100_tail_verbs.csv", index_col="verb").index.values.tolist()
tail_class_nouns = pd.read_csv("./data/EK100/EPIC_100_tail_nouns.csv", index_col="noun").index.values.tolist()
tail_class_action = []
for i,(v,n) in enumerate(json.load(open('./data/EK100/EK100_action_composition.json'))):
    if v in tail_class_verbs or n in tail_class_nouns:
        tail_class_action.append(i)
tail = compute_recall(rank, gt, np.intersect1d(tail_class_action, np.unique(gt)))

unseen_participant_ids = pd.read_csv("./data/EK100/EPIC_100_unseen_participant_ids_validation.csv",index_col="participant_id").index.values.tolist()
is_unseen = [i.split('_')[0] in unseen_participant_ids for i in idx]
is_unseen = np.array(is_unseen,dtype=bool)
unseen = compute_recall(rank[is_unseen], gt[is_unseen], np.unique(gt[is_unseen]))

print('Overall %.3f Unseen %.3f Tail %.3f' % (overall, unseen, tail)) 
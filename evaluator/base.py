
from asyncio import FastChildWatcher
from collections import defaultdict
from typing import (
    OrderedDict,
)
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
import os
import pickle as pkl


def softmax(x: np.ndarray):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))




class Base_Evaluator():
    def __init__(self, dataset):
        if dataset.name =='EPIC-KITCHENS-55':
            self.skip_recall = True
            # require many shot in EK55. Not implemented yet.
        else:
            self.skip_recall = False
    
    def compute_score(self,rank,gt_class,classes,topk):
        is_topk_correct = np.any(rank[:,:topk] == np.expand_dims(gt_class,1),axis=1)
        acc = is_topk_correct.mean()
        recall = recall_score(
            y_true=gt_class,
            y_pred=np.where(is_topk_correct,gt_class,rank[:,0]),
            labels = classes,
            average = None
        ).mean()
        return acc,recall

    def __call__(self, prediction,gt):  

        if 'verb' in prediction:
            score_action = prediction['action']
            score_verb = prediction['verb']
            score_noun = prediction['noun']
            
            rank_verb = np.argsort(score_verb,axis=1)[:,::-1]
            rank_noun = np.argsort(score_noun,axis=1)[:,::-1]
            rank_action = np.argsort(score_action,axis=1)[:,::-1]

            acc_verb_top1, recall_verb_top1 = self.compute_score(rank_verb,gt['verb'],np.unique(gt['verb']),topk=1)
            acc_verb_top5, recall_verb_top5 = self.compute_score(rank_verb,gt['verb'],np.unique(gt['verb']),topk=5)

            acc_noun_top1, recall_noun_top1 = self.compute_score(rank_noun,gt['noun'],np.unique(gt['noun']),topk=1)
            acc_noun_top5, recall_noun_top5 = self.compute_score(rank_noun,gt['noun'],np.unique(gt['noun']),topk=5)
            
            acc_action_top1, recall_action_top1 = self.compute_score(rank_action,gt['action'],np.unique(gt['action']),topk=1)
            acc_action_top5, recall_action_top5 = self.compute_score(rank_action,gt['action'],np.unique(gt['action']),topk=5)
        
            if not self.skip_recall:
                return OrderedDict([
                    ('T1_V',                acc_verb_top1),
                    ('T1_N',                acc_noun_top1),
                    ('T1_A',                acc_action_top1),
                    ('T5_V',                acc_verb_top5),
                    ('T5_N',                acc_noun_top5),
                    ('T5_A',                acc_action_top5),
                    ('Rec1_V',              recall_verb_top1),
                    ('Rec1_N',              recall_noun_top1),
                    ('Rec1_A',              recall_action_top1),
                    ('Rec5_V',              recall_verb_top5),
                    ('Rec5_N',              recall_noun_top5),
                    ('Rec5_A',              recall_action_top5)
                ])
            else:
                return OrderedDict([
                    ('T1_V',                acc_verb_top1),
                    ('T1_N',                acc_noun_top1),
                    ('T1_A',                acc_action_top1),
                    ('T5_V',                acc_verb_top5),
                    ('T5_N',                acc_noun_top5),
                    ('T5_A',                acc_action_top5)
                ])
        else:
            
            score_action = prediction['action']
            rank_action = np.argsort(score_action,axis=1)[:,::-1]
            acc_action_top1, recall_action_top1 = self.compute_score(rank_action,gt['action'],np.unique(gt['action']),topk=1)
            acc_action_top5, recall_action_top5 = self.compute_score(rank_action,gt['action'],np.unique(gt['action']),topk=5)
            return OrderedDict([
                ('T1_A',                acc_action_top1),
                ('T5_A',                acc_action_top5),
                ('Rec1_A',              recall_action_top1),
                ('Rec5_A',              recall_action_top5)
            ])

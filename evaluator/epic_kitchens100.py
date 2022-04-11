import argparse
import logging
import sys
from pathlib import Path
from cv2 import exp
import numpy as np
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Sequence
from typing import Any
from typing import Iterable
import json
import os
from collections import OrderedDict
import copy

import pandas as pd
from collections import defaultdict
from sklearn.metrics import recall_score

def softmax(x: np.ndarray):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

class EPIC_KITCHENS100_Evaluator():
    def __init__(self,dataset ,root='./data/EK100'):

        self.action_composition = dataset.action_composition
        self.vn2action = {(v,n) :i for i,(v,n) in enumerate(self.action_composition)}
        self.verb_to_action_set = defaultdict(list)
        self.noun_to_action_set = defaultdict(list)
        for i,(v,n) in enumerate(self.action_composition):
            self.verb_to_action_set[v].append(i)
            self.noun_to_action_set[n].append(i)
        
        self.num_action = dataset.num_action
        self.num_verb = dataset.num_verb
        self.num_noun = dataset.num_noun

        self.tail_class_verbs = pd.read_csv(
            os.path.join(root,"EPIC_100_tail_verbs.csv"), index_col="verb"
        ).index.values.tolist()
        self.tail_class_nouns = pd.read_csv(
            os.path.join(root,"EPIC_100_tail_nouns.csv"), index_col="noun"
        ).index.values.tolist()
        self.tail_class_action = []
        for i,(v,n) in enumerate(self.action_composition):
            if v in self.tail_class_verbs or n in self.tail_class_nouns:
                self.tail_class_action.append(i)

        self.unseen_participant_ids = pd.read_csv(
            os.path.join(root,"EPIC_100_unseen_participant_ids_validation.csv"),
            index_col="participant_id",
        ).index.values.tolist()

    
    def compute_recall(self,rank,gt_class,classes,topk=5):
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
    
    def intersection(self,classes, labels):
        return np.intersect1d(classes, np.unique(labels))

    def __call__(self, prediction,gt):
        if 'action' in prediction and 'verb' in prediction and 'noun' in prediction:
            score_verb = prediction['verb']
            score_noun = prediction['noun']
            score_action = prediction['action']
        elif 'action' not in prediction:
            score_verb = prediction['verb']
            score_noun = prediction['noun']
            prob_verb = softmax(score_verb)
            prob_noun = softmax(score_noun)
            score_action = np.zeros((prob_verb.shape[0],self.num_action))
            for i, (v,n) in enumerate(self.action_composition):
                score_action[:,i] = prob_verb[:,v] * prob_noun[:,n]
        elif 'verb' not in prediction and 'noun' not in prediction:
            score_action = prediction['action']
            score_noun = np.zeros((score_action.shape[0],self.num_noun))
            score_verb = np.zeros((score_action.shape[0],self.num_verb))
            for i in range(self.num_noun):
                if i not in self.noun_to_action_set:
                    score_noun[:,i]  = 0
                else:
                    score_noun[:,i] = score_action[:,self.noun_to_action_set[i]].max(1)
            for i in range(self.num_verb):
                if i not in self.verb_to_action_set:
                    score_verb[:,i] = 0
                else:
                    score_verb[:,i] = score_action[:,self.verb_to_action_set[i]].max(1)
        else:
            raise NotImplementedError
        
        dirty = False
        if dirty:
            score_verb = copy.deepcopy(score_verb)
            score_noun = copy.deepcopy(score_noun)
            score_action = copy.deepcopy(score_action)
            score_verb[: , ~ np.in1d(np.arange(self.num_verb),np.unique(gt['verb']))] = - np.inf
            score_noun[: , ~ np.in1d(np.arange(self.num_noun),np.unique(gt['noun']))] = - np.inf
            score_action[: , ~ np.in1d(np.arange(self.num_action),np.unique(gt['action']))] = - np.inf

        rank_verb = np.argsort(score_verb,axis=1)[:,::-1]
        rank_noun = np.argsort(score_noun,axis=1)[:,::-1]
        rank_action = np.argsort(score_action,axis=1)[:,::-1]

        all_verb = self.compute_recall(rank_verb, gt['verb'], np.unique(gt['verb']))
        all_noun = self.compute_recall(rank_noun, gt['noun'], np.unique(gt['noun']))
        all_action = self.compute_recall(rank_action, gt['action'], np.unique(gt['action']))

        tail_verb = self.compute_recall(rank_verb, gt['verb'], self.intersection(self.tail_class_verbs,gt['verb']))
        tail_noun = self.compute_recall(rank_noun, gt['noun'], self.intersection(self.tail_class_nouns,gt['noun']))
        tail_action = self.compute_recall(rank_action, gt['action'],self.intersection(self.tail_class_action,gt['action']))

        is_unseen = [i.split('_')[0] in self.unseen_participant_ids for i in gt['id']]
        is_unseen = np.array(is_unseen,dtype=bool)
        
        unseen_verb = self.compute_recall(rank_verb[is_unseen], gt['verb'][is_unseen], np.unique(gt['verb'][is_unseen]))
        unseen_noun = self.compute_recall(rank_noun[is_unseen], gt['noun'][is_unseen], np.unique(gt['noun'][is_unseen]))
        unseen_action = self.compute_recall(rank_action[is_unseen], gt['action'][is_unseen], np.unique(gt['action'][is_unseen]))

        return OrderedDict([
            ('All_V',               all_verb),
            ('All_N',               all_noun),
            ('All_A',               all_action),
            ('Tail_V',              tail_verb),
            ('Tail_N',              tail_noun),
            ('Tail_A',              tail_action),
            ('Uns_V',               unseen_verb),
            ('Uns_N',               unseen_noun),
            ('Uns_A',               unseen_action)
        ])



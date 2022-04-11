from collections import defaultdict
import json
from operator import index
from pandas.core import frame
import torch
import pandas as pd
import os
import pickle as pkl
import numpy as np
import cv2
import h5py
import tqdm
import functools
import lmdb


class _50SALADS_DATASET(torch.utils.data.Dataset):
    def __init__(self, logger, config, root = None):
        super().__init__()
        
        self.root = './data/50S'
        
        self.name  = config.name
        self.split = config.split
        assert os.path.exists(self.root)
        assert config.name in ['50salads']
        self.label_root = os.path.join(self.root,'groundTruth')

        self.feature = config.feature
        self.feature_fps = config.feature_fps   # 30
        self.feature_dim = config.feature_dim   # 768

        self.config = config
        self.model_fps = config.fps 
        self.tau_a = config.tau_a

        with open(os.path.join(self.root,'mapping.txt')) as f:
            self.action_classes = [line.strip().split()[1] for line in f.readlines()]
        self.num_action = len(self.action_classes)
        self.num_noun = 0
        self.num_verb = 0

        split_file = {
            'train1' : 'train.split1.bundle',
            'train2' : 'train.split2.bundle',
            'train3' : 'train.split3.bundle',
            'train4' : 'train.split4.bundle',
            'train5' : 'train.split5.bundle',
            'test1' : 'test.split1.bundle',
            'test2' : 'test.split2.bundle',
            'test3' : 'test.split3.bundle',
            'test4' : 'test.split4.bundle',
            'test5' : 'test.split5.bundle',
        }[config.split]
        with open(os.path.join(self.root,'splits',split_file)) as f:
            self.videos = [v.strip() for v in f.readlines()]
            self.videos = [v.replace('.txt','') for v in self.videos if len(v) > 0]


        assert config.past_frame>0
        self.data = []
        for v in self.videos:
            label_file = os.path.join(self.label_root, v + '.txt')
            with open(label_file) as f:
                labels = [line.strip() for line in f.readlines()]
                labels = [self.action_classes.index(lb) for lb in labels]

            segment_label = []
            segment_start = []
            for i,a in enumerate(labels):
                if i == 0 or a!=segment_label[-1]:
                    segment_label.append(a)
                    segment_start.append(i)
            segment_end = [a - 1 for a in segment_start[1:]] + [len(labels)]
            
            n_frame = len(labels)

            for start_frame, end_frame, action_label in zip(segment_start, segment_end, segment_label):

                segment = {
                    'video_id' : v,
                    'next_action_class' : action_label,
                }

                if config.drop and start_frame <= self.tau_a *  self.feature_fps:
                    continue
                
                frame_index = np.arange(
                    start_frame - self.tau_a * self.feature_fps + config.forward_frame * self.feature_fps / self.model_fps, 
                    start_frame - self.tau_a * self.feature_fps - config.past_frame * self.feature_fps / self.model_fps,
                    - self.feature_fps / self.model_fps
                ).astype(int)[::-1]
                assert len(frame_index) == config.past_frame + config.forward_frame
                frame_index[frame_index < 0] = 0
                frame_index[frame_index >= n_frame] = n_frame - 1
                segment['frame_index'] = frame_index

                self.data.append(segment)
    
            # debug 
            # break
        assert config.feat_file
        self.f = lmdb.open(config.feat_file, readonly=True, lock=False)

        logger.info('[%s] # Frame: Past %d. Forward %d.' % (
            config.split, config.past_frame,config.forward_frame))
        logger.info('[%s] # segment %d. verb %d. noun %d. action %d.' % (
            config.split, len(self.data), self.num_verb, self.num_noun, self.num_action))

        self.action_weight = None
        self.cache = {}
        if config.cache:
            self.make_cache(logger)

    def make_cache(self,logger):
        logger.info('Cache: Load all feature into memory')
        for segment in self.data:
            for fid in segment['frame_index']:            
                key = '%s_frame_%010d' % (segment['video_id'],fid)
                if key not in self.cache:
                    res = self._read_one_frame_feat(key)
                    self.cache[key] = res
        logger.info('Cache: Finish loading. Cache Size %d' % len(self.cache))

    

    def _read_one_frame_feat(self,key):
        if key in self.cache:
            return self.cache[key]
        with self.f.begin() as e:
            buf = e.get(key.strip().encode('utf-8'))
            if buf is not None:
                res = np.frombuffer(buf,'float32')
            else:
                res = None
        return res
      
    def _load_feat(self,video_id, frame_ids):
        frames = []
        dim = self.feature_dim
        
        for fid in frame_ids:                
            key = '%s_frame_%010d' % (video_id,fid)
            frame_feat = self._read_one_frame_feat(key)
            if frame_feat is not None:
                frames.append(frame_feat)
            elif len(frames) > 0:
                frames.append(frames[-1])
                # print('Copy frame:    %s' % key)
            else:
                frames.append(np.zeros(dim))
                # print('Zero frame:    %s' % key)
        return torch.from_numpy(np.stack(frames,0)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        segment = self.data[i]

        out = {}
        out['id'] = i
        out['index'] = i
        out['next_action_class'] = segment['next_action_class']
        

        out['past_frame'] = self._load_feat(
            segment['video_id'], 
            segment['frame_index'], 
        )
        return out

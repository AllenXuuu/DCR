from collections import defaultdict
import json
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


class EGTEA_GAZE_DATASET(torch.utils.data.Dataset):
    def __init__(self, logger, config, root = None):
        super().__init__()

        self.root = './data/EG+'
        
        self.name  = config.name
        self.split = config.split
        self.config = config
        self.model_fps = config.fps 
        self.tau_a = config.tau_a
        
        self.feature = config.feature
        self.feature_fps = config.feature_fps
        self.feature_dim = config.feature_dim
        
        assert config.name == 'EGTEA_GAZE+'

        self.class_info = pd.read_csv(os.path.join(self.root,'actions.csv'), names=['action_class','verb_noun_class','text'])
        self.num_action = self.class_info.shape[0]
        self.vn2action = []
        for _, a in self.class_info.iterrows():
            v,n = list(map(int,a.verb_noun_class.split('_')))
            self.vn2action.append([v,n])
        self.num_verb = len(set([a[0] for a in self.vn2action]))
        self.num_noun = len(set([a[1] for a in self.vn2action]))

        annotation_file = {
            'train1':'training1.csv',
            'train2':'training2.csv',
            'train3':'training3.csv',
            'valid1':'validation1.csv',
            'valid2':'validation2.csv',
            'valid3':'validation3.csv',
        }[config.split]
        annotation_file = os.path.join(self.root,annotation_file)

        assert config.past_frame > 0 

        self.data = []
        info = pd.read_csv(annotation_file, header=None, names=['video','start','end','verb','noun','action'])
            
        for idx,a in info.iterrows():
            video_name = a.video
            start_frame = a.start
            end_frame = a.end
            aid = a.action
            vid = a.verb
            nid = a.noun

            segment = {
                'id' : idx,
                'video_id' : video_name,
                'next_verb_class' : vid,
                'next_noun_class' : nid,
                'next_action_class' : aid,
            }
            
            if config.drop and start_frame<=self.tau_a * self.feature_fps:
                continue
            
            frame_index = np.arange(
                start_frame - self.tau_a * self.feature_fps + config.forward_frame * self.feature_fps / self.model_fps, 
                start_frame - self.tau_a * self.feature_fps - config.past_frame * self.feature_fps / self.model_fps,
                - self.feature_fps / self.model_fps
            ).astype(int)[::-1]
            assert len(frame_index) == config.past_frame + config.forward_frame
            frame_index[frame_index<1] = 1

            segment['frame_index'] = frame_index
            self.data.append(segment)
    
            # debug 
            # break
        
        self.verb_weight, self.noun_weight, self.action_weight = None, None, None
        
        
        ##### feature
        assert config.feat_file
        self.f = lmdb.open(config.feat_file, readonly=True, lock=False)

        logger.info('[%s] # Frame: Past %d. Forward %d.' % (
            config.split, config.past_frame,config.forward_frame))
        logger.info('[%s] # segment %d. verb %d. noun %d. action %d.' % (
            config.split, len(self.data), self.num_verb, self.num_noun, self.num_action))
        

        self.cache = {}
        if config.cache:
            self.make_cache(logger)

    def make_cache(self,logger):
        logger.info('Cache: Load all feature into memory')
        for segment in self.data:
            for fid in segment['frame_index']:            
                key = '%s_frame_%010d.jpg' % (segment['video_id'],fid)
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
            key = '%s_frame_%010d.jpg' % (video_id,fid)
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

        out = {
            'id' : segment['id'],
            'index' : i
        }
        
        out['next_action_class'] = segment['next_action_class']
        out['next_verb_class'] = segment['next_verb_class']
        out['next_noun_class'] = segment['next_noun_class']
        

        out['past_frame'] = self._load_feat(
            segment['video_id'], 
            segment['frame_index'], 
        )

        return out

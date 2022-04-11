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
import lmdb
from functools import lru_cache


class EPIC_KITCHENS_DATASET(torch.utils.data.Dataset):
    def __init__(self, logger, config):
        super().__init__()
        
        self.data_root = './data/EK55' if config.name == 'EPIC-KITCHENS-55' else './data/EK100'
        
        self.name  = config.name
        self.split = config.split
        self.config = config
        self.challenge = 'test' in config.split
        self.model_fps = config.fps 
        self.tau_a = config.tau_a
        
        self.feature = config.feature
        self.feature_fps = config.feature_fps
        self.feature_dim = config.feature_dim
        
        if config.name == 'EPIC-KITCHENS-55':
            if config.split == 'train':
                video_list = pd.read_csv(os.path.join(self.data_root,'training_videos.csv'),header = None)[0]
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_train_action_labels.pkl'),'rb'))
                self.action_info = self.action_info.loc[self.action_info['video_id'].isin(video_list)]
            elif config.split == 'valid':
                video_list = pd.read_csv(os.path.join(self.data_root,'validation_videos.csv'),header = None)[0]
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_train_action_labels.pkl'),'rb'))
                self.action_info = self.action_info.loc[self.action_info['video_id'].isin(video_list)]
            elif config.split == 'trainval':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_train_action_labels.pkl'),'rb'))
            elif config.split == 'test_seen':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_test_s1_timestamps.pkl'),'rb'))
            elif config.split == 'test_unseen':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_test_s2_timestamps.pkl'),'rb'))
            else:
                raise NotImplementedError('Unknow split [%s] for dataset [%s]' % (config.split, config.name))

            self.num_noun = pd.read_csv(os.path.join(self.data_root,'EPIC_noun_classes.csv')).shape[0]
            self.num_verb = pd.read_csv(os.path.join(self.data_root,'EPIC_verb_classes.csv')).shape[0]
            self.action_composition = json.load(open(os.path.join(self.data_root, 'EK55_action_composition.json'),'r'))
            self.num_action = len(self.action_composition)
            self.vn2action =  {(v,n): i for i, (v,n) in enumerate(self.action_composition)}

        elif config.name == 'EPIC-KITCHENS-100':
            if config.split == 'train':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_100_train.pkl'),'rb'))
            elif config.split == 'valid':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_100_validation.pkl'),'rb'))
            elif config.split == 'trainval':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_100_train.pkl'),'rb'))
                self.action_info = self.action_info.append(
                    pkl.load(open(os.path.join(self.data_root,'EPIC_100_validation.pkl'),'rb')))
            elif config.split == 'test':
                self.action_info = pkl.load(open(os.path.join(self.data_root,'EPIC_100_test_timestamps.pkl'),'rb'))
            else:
                raise NotImplementedError('Unknow split [%s] for dataset [%s]' % (config.split, config.name))

            self.num_noun = pd.read_csv(os.path.join(self.data_root,'EPIC_100_noun_classes.csv')).shape[0]
            self.num_verb = pd.read_csv(os.path.join(self.data_root,'EPIC_100_verb_classes.csv')).shape[0]
            self.action_composition = json.load(open(os.path.join(self.data_root, 'EK100_action_composition.json'),'r'))
            self.num_action = len(self.action_composition)
            self.vn2action =  {(v,n): i for i, (v,n) in enumerate(self.action_composition)}

        else:
            raise NotImplementedError('Unknow dataset: %s' % config.name)
        
        if config.weight:
            self.verb_weight = np.array(pkl.load(open(os.path.join(self.data_root,'verb_weight.pkl'),'rb')))
            self.noun_weight = np.array(pkl.load(open(os.path.join(self.data_root,'noun_weight.pkl'),'rb')))
            self.action_weight = np.array(pkl.load(open(os.path.join(self.data_root,'action_weight.pkl'),'rb')))
        else:
            self.verb_weight, self.noun_weight, self.action_weight = None, None, None

        ##### store source frame index
        assert config.past_frame >= 0
        self.data = []
        self.frame_label = defaultdict(dict)
        for video_id, group in self.action_info.groupby('video_id'):
            for idx, a in group.iterrows():
                segment = {
                    'id' : idx,
                    'participant_id' : a.participant_id,
                    'video_id': video_id,
                }
                
                start_frame = int(self.timestr_to_second(a.start_timestamp) * self.feature_fps)
                end_frame = int(self.timestr_to_second(a.stop_timestamp) * self.feature_fps)

                if not self.challenge:
                    for fid in range(start_frame,end_frame):
                        self.frame_label[video_id][fid] = (a.verb_class,a.noun_class)

                if config.drop and start_frame<=self.tau_a * self.feature_fps:
                    continue
                
                frame_index = np.arange(
                    start_frame - self.tau_a * self.feature_fps + config.forward_frame * self.feature_fps / self.model_fps, 
                    start_frame - self.tau_a * self.feature_fps - config.past_frame * self.feature_fps / self.model_fps,
                    - self.feature_fps / self.model_fps
                ).astype(int)[::-1]
                assert len(frame_index) == config.past_frame + config.forward_frame
                frame_index[frame_index < 1] = 1
                segment['frame_index'] = frame_index

                if not self.challenge:
                    segment['next_verb_class'] = a.verb_class
                    segment['next_noun_class'] = a.noun_class
                    segment['next_action_class'] = self.vn2action[(a.verb_class,a.noun_class)]

                self.data.append(segment)

                # debug 
                # break

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

    def timestr_to_second(self,x):
        a,b,c = list(map(float,x.split(':')))
        return c + 60 * b + 3600 * a

   
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
            
            # handling special case for irCSN feature provided by AVT
            if self.feature == 'irCSN10':
                if fid %3!=0:
                    fid = (fid // 3) * 3 
            if self.feature == 'irCSN25':
                if fid % 6 == 3:
                    fid = fid -1
            
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
        if not self.challenge:
            out['next_action_class'] = segment['next_action_class']
            out['next_verb_class'] = segment['next_verb_class']
            out['next_noun_class'] = segment['next_noun_class']
           
        out['past_frame'] = self._load_feat(
            segment['video_id'], 
            segment['frame_index'], 
        )

        return out

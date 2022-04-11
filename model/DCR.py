from typing import OrderedDict
import torch
import torch.nn as nn
from termcolor import colored
from .transformer import transformer
from .lstm import lstm
from .model_utils import ClassifierWithLoss


def build_reasoner(config,length):
    if config.name == 'transformer':
        return transformer(config, length)
    elif config.name == 'lstm':
        return lstm(config)
    else:
        raise NotImplementedError(config.name)

class DCR(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        
        self.config = config
        self.loss = self.config.loss

        self.full_frame = self.config.past_frame + self.config.anticipation_frame + self.config.action_frame

        self.encoder = nn.Linear(config.feat_dim, config.reasoner.d_model)
        self.decoder = nn.Linear(config.reasoner.d_model, config.feat_dim)
        
        self.reasoner = build_reasoner(config.reasoner, self.config.past_frame + self.config.anticipation_frame + self.config.action_frame)

        self.num_verb = dataset.num_verb
        self.num_noun = dataset.num_noun
        self.num_action = dataset.num_action

        if config.classifier.verb:
            self.verb_classifier = ClassifierWithLoss(
                config.feat_dim,
                self.num_verb,
                hiddens = config.classifier.hidden,
                dropout = config.classifier.dropout,
                loss_smooth = config.loss.smooth,
                loss_weight = dataset.verb_weight,
            )
        else:
            self.verb_classifier = None
        
        if config.classifier.noun:
            self.noun_classifier = ClassifierWithLoss(
                config.feat_dim,
                self.num_noun,
                hiddens = config.classifier.hidden,
                dropout = config.classifier.dropout,
                loss_smooth = config.loss.smooth,
                loss_weight = dataset.noun_weight,
            )
        else:
            self.noun_classifier = None
        
        if config.classifier.action:
            self.action_classifier = ClassifierWithLoss(
                config.feat_dim,
                self.num_action,
                hiddens = config.classifier.hidden,
                dropout = config.classifier.dropout,
                loss_smooth = config.loss.smooth,
                loss_weight = dataset.action_weight,
            )
        else:
            self.action_classifier = None
    
    def generate_mask(self,easiness):
        mask = torch.ones(len(easiness),self.full_frame).to(easiness.device)
        mask[ : ,  -self.config.action_frame : ] = 0
        mask[ : ,  -(self.config.anticipation_frame + self.config.action_frame) : -self.config.action_frame] = \
            torch.bernoulli(easiness.unsqueeze(1).tile(self.config.anticipation_frame))
        return mask
    
        
    def forward(self,
        batch,
        is_training=False):
        
        frames = batch['past_frame']
        bz, cur_frame, dim = frames.shape

        assert cur_frame in [ self.config.past_frame, self.full_frame ]

        if cur_frame == self.config.past_frame:
            frames = torch.nn.functional.pad(frames, (0,0,0,self.config.anticipation_frame + self.config.action_frame))

        if not is_training:
            mask = torch.ones(self.full_frame).to(frames.device)
            mask[-(self.config.anticipation_frame + self.config.action_frame):] = 0
            out_frames = self.decoder(self.reasoner(self.encoder(frames), mask))

            consensus = OrderedDict()
            for i in range(1,1 + self.config.action_frame):
                pred = OrderedDict()
                if self.verb_classifier is not None:
                    pred['verb'] = self.verb_classifier(out_frames[:, - i])
                if self.noun_classifier is not None:
                    pred['noun'] = self.noun_classifier(out_frames[:, - i])
                if self.action_classifier is not None:
                    pred['action'] = self.action_classifier(out_frames[:, - i])

                for k,v in pred.items():
                    if k not in consensus:
                        consensus[k] = v.clone()
                    else:
                        consensus[k] = consensus[k] + v

            return consensus
        else:
            mask = self.generate_mask(batch['easiness'])
            out_frames = self.decoder(self.reasoner(self.encoder(frames), mask))

            loss_dict = OrderedDict()
            loss_dict['loss_total'] = 0
            if self.loss.next_cls > 0:
                if self.verb_classifier is not None:
                    loss_dict['loss_next_verb'] = 0
                    for i in range(1,1+self.config.action_frame):
                        loss_dict['loss_next_verb'] += self.verb_classifier(
                            out_frames[:, - i],
                            batch['next_verb_class']
                            ) * self.loss.verb * self.loss.next_cls
                    loss_dict['loss_total'] += loss_dict['loss_next_verb']

                if self.noun_classifier is not None:
                    loss_dict['loss_next_noun'] = 0
                    for i in range(1,1+self.config.action_frame):
                        loss_dict['loss_next_noun'] += self.noun_classifier(
                            out_frames[:, - i],
                            batch['next_noun_class']
                            ) * self.loss.noun * self.loss.next_cls
                    loss_dict['loss_total'] += loss_dict['loss_next_noun']
                
                if self.action_classifier is not None:
                    loss_dict['loss_next_action'] = 0
                    for i in range(1,1+self.config.action_frame):
                        loss_dict['loss_next_action'] += self.action_classifier(
                            out_frames[:, - i],
                            batch['next_action_class']
                            ) * self.loss.action * self.loss.next_cls
                    loss_dict['loss_total'] += loss_dict['loss_next_action']
        
            if self.loss.feat_mse > 0 :
                mse = nn.functional.mse_loss(
                    frames,
                    out_frames,
                    reduction='none'
                ).mean(-1) * (1 - mask)
                loss_dict['loss_feat_mse'] = mse.mean(0).sum() * self.loss.feat_mse
                loss_dict['loss_total'] += loss_dict['loss_feat_mse']
            loss_dict['MaskedFrame'] = (1 - mask).float().sum() / bz


            last_visible = torch.arange(self.full_frame).expand(bz,self.full_frame).to(frames.device) * mask
            last_visible = last_visible.max(-1)[0]
            assert torch.any(self.full_frame - last_visible > self.config.action_frame)
            criterion_forward = 4
            criterion_index = (last_visible + criterion_forward).long()
            criterion = nn.functional.mse_loss(
                frames.detach()[torch.arange(bz).to(frames.device).long(),       criterion_index],
                out_frames.detach()[torch.arange(bz).to(frames.device).long(),   criterion_index],
                reduction='none'
            ).mean(-1)
            return loss_dict, criterion
import copy
import os
from yacs.config import CfgNode as CN
import argparse
import time

root = CN()

# root.name = '_'.join(time.asctime(time.localtime(time.time())).split())
root.name = None
root.local_rank = None
root.seed = 0
root.AMP_OPT_LEVEL = 'O1'
root.save_freq = None
root.main_metric = None


root.resume = CN()
root.resume.path = None
root.resume.pe_from_cls = True
root.resume.type = None

########################################## Curriculum
root.curriculum = CN()
root.curriculum.gamma_min = 0.9
root.curriculum.gamma_max = 0.98


########################################## TRAIN: DATA
root.train = CN()
root.train.report_iter = None
root.train.data = CN()

root.train.data.name = ""
root.train.data.split = ""
root.train.data.cache = False
root.train.data.drop = False
root.train.data.forward_frame = 0
root.train.data.past_frame = 0
root.train.data.fps = 4
root.train.data.tau_a = 1.0
root.train.data.batch_size = 8
root.train.data.num_workers = 8
root.train.data.weight = True


root.train.data.feat_file = ""
root.train.data.feature = ""
root.train.data.feature_fps = None
root.train.data.feature_dim = None

########################################## TRAIN: LEARNING 
root.train.max_epoch = 100
root.train.clip_grad = 5.0

root.train.optimizer = CN()
root.train.optimizer.name = 'AdamW'
root.train.optimizer.base_lr = 1e-4
root.train.optimizer.momentum = 0.9
root.train.optimizer.betas = (0.9, 0.999)
root.train.optimizer.weight_decay = 1e-2
root.train.optimizer.eps = 1e-8

root.train.scheduler = CN()
root.train.scheduler.name = 'no'
root.train.scheduler.step = []
root.train.scheduler.gamma = 0.1
root.train.scheduler.eta_min = 0.
root.train.scheduler.warmup_epoch = 0

########################################## EVAL: DATA
root.eval = CN()
root.eval.freq = 100
root.eval.report_iter = None

root.eval.data = CN()
root.eval.data.name = ""
root.eval.data.split = ""
root.eval.data.cache = False
root.eval.data.drop = False
root.eval.data.forward_frame = 0
root.eval.data.past_frame = 0
root.eval.data.fps = 4
root.eval.data.tau_a = 1.0
root.eval.data.batch_size = 8
root.eval.data.num_workers = 8
root.eval.data.weight = False

root.eval.data.feat_file = ""
root.eval.data.feature = ""
root.eval.data.feature_fps = None
root.eval.data.feature_dim = None

########################################## MODEL: ARCH
root.model = CN()
root.model.name = None

root.model.feat_dim = 0
root.model.past_frame = 0
root.model.anticipation_frame = 4
root.model.action_frame = 4

root.model.reasoner = CN()
root.model.reasoner.name = None
root.model.reasoner.d_model = 512
root.model.reasoner.nhead = 1
root.model.reasoner.dff = 2048
root.model.reasoner.depth = 1
root.model.reasoner.dropout = 0.1
root.model.reasoner.pe_type = 'learnable'

root.model.classifier = CN()
root.model.classifier.action = False
root.model.classifier.verb = False
root.model.classifier.noun = False
root.model.classifier.hidden = []
root.model.classifier.dropout = 0.


########################################## MODEL: LOSS
root.model.loss = CN()
root.model.loss.name = 'CE'
root.model.loss.smooth = 0.
root.model.loss.sigma = None

## weight
root.model.loss.verb = 1.
root.model.loss.noun = 1.
root.model.loss.action = 1.
root.model.loss.next_cls = 0.
root.model.loss.feat_mse = 0.



def load_config(args= None):
    config = copy.deepcopy(root)
    
    if args is None:
        return config

    config.merge_from_file(args.cfg)

    config.local_rank = args.local_rank
    config.resume.type = args.weight_type
    config.resume.path = args.resume

    if args.name is not None:
        config.name =  args.name
    
    _,folder,_ = args.cfg.split('/')
    config.folder = os.path.join('exp',folder)


    return config

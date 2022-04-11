import torch
import numpy as np
import random
import os
import torch.distributed as dist
import logging
from tensorboardX import SummaryWriter
import time
from apex import amp
import argparse
import pickle as pkl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--weight_type', default='pretrain', type=str)
    return parser.parse_args()


def initialize(cfg):
    if cfg.local_rank is not None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print("DDP! RANK %d WORLD_SIZE %d " % (rank, world_size))
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        cfg.seed = rank + cfg.seed

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.backends.cudnn.benchmark = True



def build_logger(name = '',path=None, console=False, tensorboard_log=False):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.propagate = False

    # create console handlers for master process
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(f'%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # create file handlers
    if path is not None:
        assert path is not None
        fn = 'log.txt' if not dist.is_initialized() else 'log_rank%d.txt' % dist.get_rank()
        file_handler = logging.FileHandler(os.path.join(path, fn), 'w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    if tensorboard_log:
        writer = SummaryWriter(os.path.join(path, 'tb'))
    else:
        writer = None
    
    return logger, writer


def max_lr(optimizer):
    return max([parma['lr'] for parma in optimizer.param_groups])


def load_resume(cfg, model, optimizer = None, scheduler = None,):
    if cfg.path is None:
        return 0,0 
    
    print('load resume from %s.' % cfg.path)
    resume = torch.load(cfg.path, map_location='cpu')
    state_dict = resume['model']
    if cfg.pe_from_cls and 'reasoner.pe' not in state_dict and 'classifier.weight' in state_dict:
        state_dict['reasoner.pe'] = state_dict['classifier.weight'].permute(1,0)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict,strict=False)
    print('Missing Key: %s' % missing_keys)
    print('Unexpected Key: %s' % unexpected_keys)

    if cfg.type == 'continue':
        if optimizer is not None:
            optimizer.load_state_dict(resume['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(resume['scheduler'])
        amp.load_state_dict(resume['amp'])
        return resume['epoch'], resume['best']
    elif cfg.type == 'test':
        return resume['epoch'], 0
    elif cfg.type == 'pretrain':
        return 0, 0
    else:
        raise NotImplementedError(cfg.type)

def save_resume(path, epoch, model, optimizer, scheduler,best):
    print('Save resume into ==> %s' % path)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp': amp.state_dict(),
        'epoch': epoch,
        'best' : best
    }, path)

def calc_learnable_params(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            cur = 1
            for a in p.shape:
                cur *= a
            total_params += cur

    return total_params

def all_gather(tensor):
    if isinstance(tensor[0],str):
        if dist.get_rank() == 0:
            for i in range(1,dist.get_world_size()):
                fn = '/tmp/anticipation_tmp_res%d.pkl.tmp%s' % (i,os.environ['MASTER_PORT'])
                with open(fn,'rb') as f:
                    x = pkl.load(f)
                os.remove(fn)
                tensor = np.concatenate([tensor, x],0)
            return tensor
    
        else:
            fn = '/tmp/anticipation_tmp_res%d.pkl.tmp%s' % (dist.get_rank(),os.environ['MASTER_PORT'])
            with open(fn,'wb') as f:
                pkl.dump(tensor,f)
            dist.barrier()
            return None

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
        

    orig_size = tensor.shape[0]
    max_size = torch.tensor(orig_size).clone().int().cuda()
    dist.all_reduce(max_size, op=dist.ReduceOp.MAX)

    padding_tensor = torch.zeros(max_size, *tensor.shape[1:]).type_as(tensor)
    padding_tensor[:orig_size] = tensor
    mask = torch.zeros(max_size).float()
    mask[:orig_size] = 1
    mask_list = [torch.zeros_like(mask).float().cuda() for _ in range(dist.get_world_size())]
    tensor_list = [torch.zeros_like(padding_tensor).type_as(tensor).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(mask_list, mask.cuda())
    dist.all_gather(tensor_list, padding_tensor.cuda())

    out = []
    for mask, tensor in zip(mask_list, tensor_list):
        out.append(tensor[mask.bool()])
    out = torch.cat(out, 0).cpu().data.numpy()
    
    return out


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.start_time = 0.
        self.clear()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.diff = 0.
        self.average_time = 0.


class AverageLoss(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.meters = {}
        self.reset()

    def reset(self):
        self.meters = {}

    def update(self, cur_meter, bz):
        for k, v in cur_meter.items():
            if k not in self.meters:
                self.meters[k] = {
                    'avg': v.item(),
                    'cnt': bz
                }
            else:
                self.meters[k]['avg'] = self.meters[k]['avg'] + \
                                        (v.item() - self.meters[k]['avg']) * bz / (self.meters[k]['cnt'] + bz)
                self.meters[k]['cnt'] += bz

    def __str__(self):
        out = ''
        for k in self.meters:
            v = self.meters[k]['avg']
            out += '%s: %.4f. ' % (k.replace('_loss', '').replace('loss_', ''), v)
        out = out[:-1]
        return out

    def items(self):
        for k in self.meters:
            yield (k, self.meters[k]['avg'])

    def aggregate(self,):
        return {
            k: self.meters[k]['avg'] for k in self.meters
        }
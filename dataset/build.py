import torch
import torch.distributed as dist
import numpy as np
from prefetch_generator import BackgroundGenerator

from ._50salads import _50SALADS_DATASET
from .egtea_gaze import EGTEA_GAZE_DATASET
from .epic_kitchens import EPIC_KITCHENS_DATASET
from collections import defaultdict

class DataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_dataset(logger,config):
    if config.name in ['EPIC-KITCHENS-55','EPIC-KITCHENS-100']:
        dataset = EPIC_KITCHENS_DATASET( logger, config)
    elif config.name == 'EGTEA_GAZE+' :
        dataset = EGTEA_GAZE_DATASET( logger, config )
    elif config.name in ['50salads']:
        dataset = _50SALADS_DATASET( logger, config)
    else:
        raise NotImplementedError('Not supported dataset: %s' % config.name)
    return dataset

def collate_fn(list_of_dict):
    out = defaultdict(list)

    for item in list_of_dict:
        for k,v in item.items():
            out[k].append(v)
    
    out = dict(out)
    for k in out:
        if 'id' in k or 'index' in k:
            continue
        if isinstance(out[k][0],(int,float)):
            out[k] = torch.tensor(out[k])
        elif isinstance(out[k][0],np.ndarray):
            out[k] = np.stack(out[k],0)
            out[k] = torch.tensor(out[k])
        elif isinstance(out[k][0] , torch.Tensor):
            out[k] = torch.stack(out[k],0)

    return out
    

def build_dataloader(logger, config, shuffle = False, ddp = False):
    dataset = build_dataset(logger, config)
    
    if ddp:
        if shuffle:
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
            )
        else:
            sampler = np.arange(dist.get_rank(), len(dataset), dist.get_world_size())
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            sampler=sampler, drop_last=False,collate_fn = collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            shuffle=shuffle, drop_last=False,collate_fn = collate_fn
        )

    logger.info('[%s] # dataset %s, batch_per_gpu %d, dataloader %d.' % (
        config.split,len(dataset),config.batch_size,len(dataloader)))
    return dataloader

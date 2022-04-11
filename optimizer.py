import torch.optim as optim
import math

def build_optimizer(config, model):

    param_group = model.parameters()

    if config.name == 'SGD':
        optimizer = optim.SGD(param_group, lr=config.base_lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)
    elif config.name == 'Adam':
        optimizer = optim.Adam(param_group, lr=config.base_lr,
                               betas=config.betas, weight_decay=config.weight_decay,
                               eps=config.eps)
    elif config.name == 'AdamW':
        optimizer = optim.AdamW(param_group, lr=config.base_lr,
                                betas=config.betas, weight_decay=config.weight_decay,
                                eps=config.eps)
    else:
        raise NotImplementedError
    return optimizer


def build_lr_scheduler(config, optimizer, n_iter_per_epoch):
    if config.name == 'no':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1
        )
    elif config.name == 'cos':
        step = config.step[0] * n_iter_per_epoch
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=step,
            T_mult=2,
            eta_min=config.eta_min,
        )
    elif config.name == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[n * n_iter_per_epoch for n in config.step],
            gamma=config.gamma
        )
    elif config.name == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma ** (1 / n_iter_per_epoch)
        )
    elif config.name == 'WarmupCos':
        warmup_step = config.warmup_epoch * n_iter_per_epoch
        decay_step = config.step[0] * n_iter_per_epoch

        def func(x):
            if x < warmup_step :
                return 0.99 * x / warmup_step + 0.01
            x,_ = math.modf((x-warmup_step)/decay_step)
            return 0.5 + 0.5 * math.cos(math.pi * x)

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,lr_lambda=func
        )
    elif config.name == 'warmup':
        warmup_step = config.warmup_epoch * n_iter_per_epoch
        WarmLambda = lambda x: (0.99 * x / warmup_step + 0.01) if x < warmup_step else  1
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,lr_lambda=WarmLambda
        )
    else:
        raise NotImplementedError

    return scheduler

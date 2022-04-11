import os,yaml
import torch
import numpy as np
from config import load_config
from dataset import build_dataloader
from model import build_model
from optimizer import build_optimizer, build_lr_scheduler
from utils import *
from apex import amp
from termcolor import colored


train_io_timer = Timer()
train_model_timer = Timer()


def train(config, epoch, loader, model, optimizer, scheduler, logger, writer):
    model.train()

    train_loss = AverageLoss()
    train_io_timer.tic()

    for i, batch in enumerate(loader):    
        train_io_timer.toc()
        train_model_timer.tic()

        subbatch = {
            k : v.cuda(non_blocking=True)
            for k,v in batch.items() if 'class' in k or 'frame' in k
        }
        
        loss = model(subbatch, is_training=True)
        loss_total = loss['loss_total']

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.train.clip_grad:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.train.clip_grad)
        else:
            loss_total.backward()
            if config.train.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        optimizer.step()
        scheduler.step()

        train_loss.update(loss, len(batch['id']))

        if i + 1 == len(loader) or (config.train.report_iter and (i + 1) % config.train.report_iter == 0):     
            num_step = (epoch - 1) * len(loader) + i
            writer.add_scalar('iter_lr/lr', max_lr(optimizer), num_step)
            for k, v in loss.items():
                writer.add_scalar('iter_loss/%s' % k, v.item(), num_step)
            logger.info(
                "Training. %03d epoch. %d/%d iter. IO time %.4f. model time %.4f. lr %.2e. loss %.4f." % (
                    epoch, i + 1, len(loader), train_io_timer.average_time, train_model_timer.average_time,
                    max_lr(optimizer), loss_total
                ))
            train_io_timer.clear()
            train_model_timer.clear()

        train_model_timer.toc()
        train_io_timer.tic()

    return train_loss



def main(config):
    out_path = os.path.join(os.getcwd(), config.folder, config.name)
    if config.resume.type != 'continue':
        os.makedirs(out_path)
    
    yaml.dump(dict(config), open(os.path.join(out_path, 'config.yml'), 'w'))
    logger, writer = build_logger(
        name = config.name,
        path = out_path,
        console=True,
        tensorboard_log=True
    )
    logger.info(config)

    train_loader = build_dataloader(logger,config.train.data,shuffle=True, ddp=False)

    model = build_model(config.model,train_loader.dataset).cuda()
    logger.info(model)

    optimizer = build_optimizer(config.train.optimizer, model)
    scheduler = build_lr_scheduler(config.train.scheduler, optimizer, len(train_loader))

    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    init_epoch, best = load_resume(config.resume, model, optimizer, scheduler)
    logger.info('Start Training!')

    for epoch in range(1 + init_epoch, config.train.max_epoch + 1):
        train_loss = train(config, epoch, train_loader, model, optimizer, scheduler, logger, writer)

        writer.add_scalar('epoch_lr/lr', max_lr(optimizer), epoch)
        for k, v in train_loss.items():
            writer.add_scalar('epoch_loss/%s' % k, v, epoch)
        train_report = "Training. %03d epoch. lr %.2e. " % (epoch, max_lr(optimizer))
        for k, v in train_loss.items():
            train_report += '%s: %.4f. ' % (k, v)
        logger.info(train_report)

        if config.save_freq is not None and epoch % config.save_freq == 0:
            save_resume(os.path.join(out_path, 'epoch_%d.pt' % epoch), epoch, model, optimizer, scheduler, best)


    logger.info('Finish Training!')


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    initialize(config)
    main(config)

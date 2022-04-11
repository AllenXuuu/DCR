from collections import defaultdict
import os,yaml
import torch
import numpy as np
from config import load_config
from dataset import build_dataloader
from model import build_model
from evaluator import build_evaluator
from utils import *
import time
from apex import amp
from termcolor import colored

train_io_timer = Timer()
train_model_timer = Timer()
eval_io_timer = Timer()
eval_model_timer = Timer()


@torch.no_grad()
def eval_and_extract(config, epoch, loader, model, logger, evaluator):
    model.eval()
    
    all_pred = defaultdict(list)
    all_gt = defaultdict(list)
    eval_io_timer.tic()
    for i, batch in enumerate(loader):
        eval_io_timer.toc()
        eval_model_timer.tic()

        subbatch = {
            k : v.cuda(non_blocking=True)
            for k,v in batch.items() if 'class' in k or 'frame' in k
        }
        pred = model(subbatch, is_training=False)

        for k in pred.keys():
            all_pred[k].append(pred[k].data.cpu().numpy())
        
        all_gt['action'].append(batch['next_action_class'])
        all_gt['verb'].append(batch['next_verb_class'])
        all_gt['noun'].append(batch['next_noun_class'])
        all_gt['id'].append(batch['id'])

        if config.eval.report_iter and ((i + 1) % config.eval.report_iter == 0 or i + 1 == len(loader)):
            logger.info(
                "Evaluation. %03d epoch. %d/%d iter. IO time %.4f. model time %.4f." % (
                    epoch, i + 1, len(loader), train_io_timer.average_time, train_model_timer.average_time
                ))
            eval_io_timer.clear()
            eval_model_timer.clear()

        eval_model_timer.toc()
        eval_io_timer.tic()

    for k in all_pred.keys():
        all_pred[k] = np.concatenate(all_pred[k],0)
    for k in all_gt:
        all_gt[k] = np.concatenate(all_gt[k],0)

    result = evaluator(all_pred,all_gt)

    return result,all_pred,all_gt


def main(config):
    logger, _ = build_logger(path=None, console=True, tensorboard_log=False)
    logger.info(config)

    eval_loader = build_dataloader(logger,config.eval.data, shuffle=False, ddp=False)
    evaluator = build_evaluator(eval_loader.dataset)

    model = build_model(config.model,eval_loader.dataset).cuda()
    logger.info(model)

    if config.AMP_OPT_LEVEL != "O0":
        model = amp.initialize(model, opt_level=config.AMP_OPT_LEVEL)


    config.resume.type = 'test'
    epoch, bst = load_resume(config.resume,model)


    logger.info('Start!')
    result,all_pred,all_gt = eval_and_extract(config, epoch, eval_loader, model, logger, evaluator)
    
    eval_report = "Eval %03d. " % epoch
    for metric, val in result.items():
        eval_report += '%s: %.4f. ' % (metric, val)
    logger.info(colored(eval_report,'yellow'))
    
    tgt_fn = config.resume.path + '.%s.pkl' % config.eval.data.split
    assert not os.path.exists(tgt_fn),tgt_fn
    logger.info('Save score into ==> %s' % tgt_fn)

    with open(tgt_fn, 'wb') as f:
        pkl.dump({
            'pred' : dict(all_pred),
            'gt' : dict(all_gt)
        },f)

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    initialize(config)
    main(config)

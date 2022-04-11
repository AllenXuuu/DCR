from collections import defaultdict
import os,yaml
from pickle import TRUE
import torch
import numpy as np
from config import load_config
from dataset import build_dataloader
from model import build_model
from evaluator import build_evaluator
from utils import *
import time
from apex import amp
from train import eval
from termcolor import colored

train_io_timer = Timer()
train_model_timer = Timer()
eval_io_timer = Timer()
eval_model_timer = Timer()


def main(config):
    logger, _ = build_logger(path=None,console=True,tensorboard_log=False)
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
    result = eval(config, epoch, eval_loader, model, logger, evaluator)
    eval_report = "Eval %03d. " % epoch
    for metric, val in result.items():
        eval_report += '%s: %.4f. ' % (metric, val)
    logger.info(colored(eval_report,'yellow'))
    logger.info('Finish!')


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    initialize(config)
    main(config)

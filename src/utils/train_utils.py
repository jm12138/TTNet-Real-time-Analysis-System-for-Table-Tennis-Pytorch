"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: utils functions that use for training process
"""

import copy
import os
import math

import paddle
from paddle.optimizer import SGD, Adam
from paddle.optimizer.lr import StepDecay, ReduceOnPlateau, LambdaDecay
import paddle.distributed as dist
from paddle.distributed import ReduceOp


def create_optimizer(configs, model):
    """Create optimizer for training process"""
    if hasattr(model, 'module'):
        train_params = [param for param in model.module.parameters() if param.requires_grad]
    else:
        train_params = [param for param in model.parameters() if param.requires_grad]

    if configs.optimizer_type == 'sgd':
        optimizer = SGD(parameters=train_params, learning_rate=configs.lr, momentum=configs.momentum,
                                    weight_decay=configs.weight_decay)
    elif configs.optimizer_type == 'adam':
        optimizer = Adam(parameters=train_params, learning_rate=configs.lr, weight_decay=configs.weight_decay)
    else:
        assert False, "Unknown optimizer type"

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """Create learning rate scheduler for training process"""
    
    if configs.lr_type == 'step_lr':
        lr_scheduler = StepDecay(optimizer._learning_rate, step_size=configs.lr_step_size, gamma=configs.lr_factor)
    elif configs.lr_type == 'plateau':
        lr_scheduler = ReduceOnPlateau(optimizer._learning_rate, factor=configs.lr_factor, patience=configs.lr_patience)
    elif configs.optimizer_type == 'cosin':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / configs.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaDecay(optimizer._learning_rate, lr_lambda=lf)
        lr_scheduler.last_epoch = configs.start_epoch - 1  # do not move
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)
    else:
        raise TypeError
    optimizer._learning_rate = lr_scheduler
    return optimizer


def get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss, earlystop_count):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    saved_state = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': lr_scheduler.state_dict(),
        'state_dict': model_state_dict,
        'best_val_loss': best_val_loss,
        'earlystop_count': earlystop_count,
    }

    return saved_state


def save_checkpoint(checkpoints_dir, saved_fn, saved_state, is_best, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    if is_best:
        save_path = os.path.join(checkpoints_dir, '{}_best.pdparams'.format(saved_fn))
    else:
        save_path = os.path.join(checkpoints_dir, '{}_epoch_{}.pdparams'.format(saved_fn, epoch))

    paddle.save(saved_state, save_path)
    print('save a checkpoint at {}'.format(save_path))


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=ReduceOp.SUM)
    rt /= world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

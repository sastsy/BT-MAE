# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
from math import ceil
import torch
import warnings

import util.misc as misc
import util.lr_sched as lr_sched

import torch.distributed as dist

from loss_func import uniformity_loss


def bt_coeff_ramp(epoch, max_epoch):
    """
    Linearly ramps from 0 → 1 over first half of training.
    """
    mid = max_epoch // 2
    if epoch >= mid:
        return 1.0
    return epoch / float(mid)


  
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    total = []
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            out_dict = model(samples, mask_ratio=args.mask_ratio)
            loss_mae = out_dict["mae_loss"]
            bt_loss = out_dict.get("bt_loss", None)
            cls_feats = out_dict["cls_feats"]
            outputs = out_dict["outputs"]
            
            on_diag = out_dict.get("on_diag", None)
            off_diag = out_dict.get("off_diag", None)

            if args.reg == 'none':
                loss_reg = torch.zeros_like(loss_mae)
            else:
                loss_reg = uniformity_loss(cls_feats)
            
            loss_ce = torch.nn.functional.cross_entropy(outputs, targets)

        loss = out_dict["loss"] + args.lamb * loss_reg + loss_ce

        loss_mae_value = loss_mae.item()
        loss_reg_value = loss_reg.item()
        loss_ce_value = loss_ce.item()
        loss_bt_value = bt_loss.item() if bt_loss is not None else 0.0
        loss_value = loss.item()
        train_acc = (outputs.argmax(dim=1) == targets).float().mean()
        on_diag_value = on_diag.item() if on_diag is not None else 0.0
        off_diag_value = off_diag.item() if off_diag is not None else 0.0


        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mae=loss_mae_value)
        metric_logger.update(loss_reg=loss_reg_value)
        metric_logger.update(loss_ce=loss_ce_value)
        metric_logger.update(train_acc=train_acc)
        if bt_loss is not None:
            metric_logger.update(loss_bt=loss_bt_value)
        if on_diag is not None:
            metric_logger.update(on_diag=on_diag_value)
        if off_diag is not None:
            metric_logger.update(off_diag=off_diag_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_mae_value_reduce = misc.all_reduce_mean(loss_mae_value)
        loss_reg_value_reduce = misc.all_reduce_mean(loss_reg_value)
        loss_ce_value_reduce = misc.all_reduce_mean(loss_ce_value)
        loss_bt_value_reduce = misc.all_reduce_mean(loss_bt_value)
        train_acc_reduce = misc.all_reduce_mean(train_acc)
        on_diag_value_reduce = misc.all_reduce_mean(on_diag_value)
        off_diag_value_reduce = misc.all_reduce_mean(off_diag_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_mae', loss_mae_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_reg', loss_reg_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_ce', loss_ce_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_acc', train_acc_reduce, epoch_1000x)
            if bt_loss is not None:
                log_writer.add_scalar('train_loss_bt', loss_bt_value_reduce, epoch_1000x)
            if on_diag is not None:
                log_writer.add_scalar("bt_on_diag", on_diag_value_reduce, epoch_1000x)
                log_writer.add_scalar("bt_off_diag", off_diag_value_reduce, epoch_1000x)

            log_writer.add_scalar('lr', lr, epoch_1000x)
        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

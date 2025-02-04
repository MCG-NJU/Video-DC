import os
import random
import warnings
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.v2 as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import logging
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import accelerate
from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.config import Config
from mmengine.runner import Runner
from synthesize.utils import load_model
from validation.utils import (
    get_dataset,
    UCF101,
    Syn_Video,
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
)


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    main_worker(args)


def main_worker(args):
    accelerator = accelerate.Accelerator(log_with='wandb')
    # name = "rded" if args.ablation=='' else 'ablation'
    group = args.wandb_name.split('-')[-1]
    accelerator.init_trackers(project_name=args.subset, init_kwargs={"wandb":{'name': args.wandb_name, 'group': group}})
    accelerator.print(args)
    print('num_processes:', accelerator.num_processes)
    args.re_batch_size = int(args.re_batch_size / accelerator.num_processes)
    print('evaluate batch size: ', args.re_batch_size)
    device = accelerator.device
    print("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))

    teacher_model = load_model(
        model_name=args.arch_name,
        pretrained=True,
        args=args
    )
    student_model = load_model(
        model_name=args.stud_name,
        pretrained=False,
        args=args
    )
    teacher_model.to(device)
    student_model.to(device)
    # teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    # student_model = torch.nn.DataParallel(student_model).cuda()

    teacher_model.eval()
    student_model.train()
    # student_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_model)

    # freeze all layers
    for param in teacher_model.parameters():
        param.requires_grad = False

    cudnn.benchmark = True

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.scheduler == 'cos':
        scheduler = LambdaLR(
            optimizer,
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
            if step <= args.re_epochs
            else 0,
            last_epoch=-1,
        )
    elif args.scheduler == "step":
        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=10)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 130], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    elif args.scheduler == 'linear':
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
            last_epoch=-1,
        )

    print("process data from {}".format(args.syn_data_path))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augment = []
    augment.append(ShufflePatches(args.factor))
    augment.append(
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)

    train_dataset = Syn_Video(
        path=args.syn_data_path,
        transform=transforms.Compose(augment),
        ipc=args.ipc,
    )
    if args.wandb_name.split('-')[0] == 'soft':
        video_all = []
        label_all = []
        for i in trange(len(train_dataset)):
            inputs = train_dataset[i][0]
            print(inputs.shape)
            video_all.append(inputs)
            label_all.append(teacher_model(inputs.unsqueeze(0).unsqueeze(0).to(device), stage='head').cpu())
        video_all = torch.stack(video_all)
        label_all = torch.cat(label_all, dim=0)
        print(label_all.shape)
        train_dataset = torch.utils.data.TensorDataset(video_all, label_all)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.re_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    # cfg = Config.fromfile(args.config[args.stud_name])
    # dataloader_cfg = cfg.get('val_dataloader')
    # val_loader = Runner.build_dataloader(dataloader_cfg)

    val_loader = torch.utils.data.DataLoader(
        get_dataset(args.subset, 'val', root=args.root, cr=0, mipc=args.mipc),
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    print("load data successfully")

    teacher_model, student_model, optimizer, train_loader, val_loader = accelerator.prepare(
        teacher_model, student_model, optimizer, train_loader, val_loader
    )
    best_acc1 = 0
    best_acc5 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    args.accelerator = accelerator

    for epoch in tqdm(range(args.re_epochs), disable=(not accelerator.is_main_process)):
        # if epoch == 150:
        #     train_loader.dataset.set_stage(stage=2)
        #     accelerator.print(train_loader.dataset.video_list[:10])
        train(epoch, train_loader, teacher_model, student_model, args)

        if epoch % 10 == 0 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.2 or epoch == 0:
                top1, top5 = validate(student_model, args, epoch)
            else:
                top1 = 0
                top5 = 0
        else:
            top1 = 0
            top5 = 0

        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_acc5 = max(top5, best_acc5)
            best_epoch = epoch
        
    accelerator.log({'result/best_epoch': best_epoch, 'result/best_acc1': best_acc1, 'result/best_acc5': best_acc5})
    accelerator.print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}, top5:{best_acc5}")


def train(epoch, train_loader, teacher_model, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    teacher_model.eval()
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            mix_images, _, _, _ = mix_aug(images, args)

            pred_label = student_model(images.unsqueeze(1), stage='head')

            soft_mix_label = teacher_model(mix_images.unsqueeze(1), stage='head')
            if args.wandb_name.split('-')[0] == 'soft':
                soft_mix_label = labels
            # soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)

        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        pred_mix_label = student_model(mix_images.unsqueeze(1), stage='head')

        if args.loss_type== "mse_gt":
            loss = F.mse_loss(pred_mix_label, soft_mix_label) + F.cross_entropy(pred_mix_label, labels) * 0.1
        elif args.loss_type == 'kl':
            soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)
            soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)
            loss = loss_function_kl(soft_pred_mix_label, soft_mix_label)
        elif args.loss_type == 'ce':
            loss = F.cross_entropy(pred_mix_label, labels)

        loss = loss / args.re_accum_steps

        # loss.backward()
        args.accelerator.backward(loss)
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        if args.wandb_name.split('-')[0] == 'soft':
            labels = soft_mix_label.argmax(dim=1)
        pred_label_all, labels_all = args.accelerator.gather_for_metrics([pred_label, labels])
        prec1, prec5 = accuracy(pred_label_all, labels_all, topk=(1, 5))
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    printInfo = (
        "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 acc = {:.6f},\t".format(top1.avg)
        + "Top-5 acc = {:.6f},\t".format(top5.avg)
        + "train_time = {:.6f}".format((time.time() - t1))
    )
    args.accelerator.log({'train/loss': objs.avg, 'train/top1': top1.avg, 'train/top5': top5.avg})
    args.accelerator.print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        # for data, target in args.val_loader:
        #     target = target.type(torch.LongTensor)
        #     data, target = data.cuda(), target.cuda()

        #     if len(data.shape) == 6:
        #         data = data.squeeze(1)
        #     output = model(data.unsqueeze(1), stage='head')
        # for i, batch in (enumerate(args.val_loader)):
        #     pred = model.module.val_step(batch)
        #     output = torch.stack([x.pred_score for x in pred])
        #     target = torch.cat([x.gt_label for x in pred])
        for i, (images, target) in enumerate(args.val_loader):
            assert len(images.shape) == 6
            output = model(images, stage='head')
            loss = loss_function(output, target)

            output_all, target_all = args.accelerator.gather_for_metrics([output, target])
            prec1, prec5 = accuracy(output_all, target_all, topk=(1, 5))
            n = output_all.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 acc = {:.6f},\t".format(top1.avg)
        + "Top-5 acc = {:.6f},\t".format(top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    args.accelerator.log({'val/loss': objs.avg, 'val/top1': top1.avg, 'val/top5': top5.avg})
    args.accelerator.print(logInfo)
    return top1.avg, top5.avg


if __name__ == "__main__":
    pass
    # main(args)

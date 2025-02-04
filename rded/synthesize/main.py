import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
import sys
sys.path.append('..')
from dataset import load_dataset as get_dataset
# from validation.utils import get_dataset


def init_images(args, model=None):
    trainset = get_dataset(args.subset, 'select', args.tau, args.root, args.num_crop, args.mipc)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc if args.mipc else args.ipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        assert len(images.shape) == 6
        print('images shape:', images.shape)
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
            inter=args.inter_mode,
        )
        # images = mix_images(images, args.input_size, args.factor, args.ipc)
        print('mixed images shape:', images.shape)
        save_images(args, denormalize(images), labels)


def save_images(args, images, class_id):
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(args.syn_data_path, class_id[id])
        num_clips = len(os.listdir(dir_path))//8 if os.path.exists(dir_path) else 0
        # print(class_id[id], num_clips)
        for t in range(images.shape[2]):
            place_to_store = dir_path + "/class{:05d}_id{:05d}_t{:03d}.jpg".format(class_id[id], num_clips, t)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            image_np = images[id, :, t].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)


def main(args):
    print(args)
    with torch.no_grad():
        if not os.path.exists(args.syn_data_path):
            os.makedirs(args.syn_data_path)
        else:
            shutil.rmtree(args.syn_data_path)
            os.makedirs(args.syn_data_path)

        model_teacher = load_model(
            model_name=args.arch_name,
            pretrained=True,
            args=args
        )

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)


if __name__ == "__main__":
    pass

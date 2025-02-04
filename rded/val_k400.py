from validation.utils import UCF101, K400
import torch
from accelerate import Accelerator
import torchvision
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from mmaction.apis import init_recognizer
from mmengine.config import Config
from mmengine.runner import Runner

if __name__=='__main__':
    # config = '/data0/chenyang/ViD/configs/recognition/r2plus1d/k400-pre_r2plus1d_r18_8xb8-8x8x1-180e_ucf101.py' # 18:78.6 34:79.3
    config = '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-kinetics-8x56x56.py'
    checkpoint = '/data0/chenyang/VDC/mmaction2/work_dirs/conv3-kinetics-8x56x56/best_acc_top1_epoch_115.pth'
    model = init_recognizer(config, checkpoint, device='cuda')
    
    transform = transforms.Compose([
        transforms.CenterCrop((56,56)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = K400(root='data/k400', transform=transform, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    model.eval()
    true = 0
    for images, labels in (val_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images, stage='head')
        _, predicted = torch.max(outputs, 1)
        true += predicted.eq(labels).sum().item()
    accuracy = true/len(val_loader.dataset)
    print(accuracy)

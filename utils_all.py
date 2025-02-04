import os
from mmaction.apis import init_recognizer
import torch
import numpy as np
import argparse
import torch.nn.functional as F
import torch.nn as nn

class ReFiner(nn.Module):
    def __init__(self, depth, channels) -> None:
        super(ReFiner, self).__init__()
        assert len(channels) - 1 == depth
        assert channels[0] == 3
        assert channels[-1] == 3
        layers = []
        for i in range(depth):
            layers.append(self._make_plain_conv(channels[i], channels[i+1]))
            layers.append(nn.BatchNorm3d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        
    def _make_plain_conv(self, in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=(1,1,1))

    def forward(self, x):
        assert len(x.shape) == 5
        return self.layers(x)
        
def shared_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hmdb51', help='dataset to distill')
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--size', type=int, default=112)
    parser.add_argument('--inter-mode', type=str, default='duplicate', help='interpretation mode')
    parser.add_argument('--gpu-id', type=str, default='0,1')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--syn_data_path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--syn_label_path', type=str,
                        default='./syn_label', help='where to store synthetic label')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--wandb-project', type=str,
                        default='Temperature', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--teacher-list', nargs='+', default=[])
    parser.add_argument('--pre', default=False, action='store_true', help='pretrained model')
    return parser

def adapt_params(args):
    root = '/data0/chenyang/ViD/'
    args.statistic_path = os.path.join(args.statistic_path, args.dataset)
    if args.dataset == 'hmdb51':
        args.size = 112
        args.num_classes = 51
        args.config = {'conv4': '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-hmdb51-8x112x112.py'}
        args.checkpoint = {'conv4': '/data0/chenyang/VDC/mmaction2/work_dirs/conv3-hmdb51-8x112x112/best_acc_top1_epoch_130.pth'}
    elif args.dataset == 'k400':
        args.size = 56
        args.num_classes = 400
        args.config = {'conv4': '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-kinetics-8x56x56.py'}
        args.checkpoint = {'conv4': '/data0/chenyang/VDC/mmaction2/work_dirs/conv3-kinetics-8x56x56/best_acc_top1_epoch_70.pth'}
    elif args.dataset == 'ucf101':
        args.size = 112
        args.num_classes = 101
        args.config = {'conv4': '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-ucf101-8x112x112.py', 'slowonly': '/data0/chenyang/VDC/mmaction2/configs/recognition/slowonly/slowonly-r18_8x8_ucf101-frame.py', 'r2plus1d': '/data0/chenyang/VDC/mmaction2/configs/recognition/r2plus1d/r2plus1d_r18_8xb8-8x8x1-180e_ucf101.py', 'i3d': '/data0/chenyang/VDC/mmaction2/configs/recognition/i3d/i3d_k400-pretrained-r18_8x8_ucf101_frame.py'}
        args.checkpoint = {'conv4': '/data0/chenyang/VDC/mmaction2/work_dirs/conv3-ucf101-8x112x112/best_acc_top1_epoch_145.pth', 'slowonly': '/data0/chenyang/VDC/mmaction2/work_dirs/slowonly-r18_8x8_ucf101-frame/best_acc_top1_epoch_134.pth', 'r2plus1d': '/data0/chenyang/VDC/mmaction2/work_dirs/r2plus1d_r18_8xb8-8x8x1-180e_ucf101/best_acc_top1_epoch_41.pth', 'i3d': '/data0/chenyang/VDC/mmaction2/work_dirs/i3d_k400-pretrained-r18_8x8_ucf101_frame/best_acc_top1_epoch_48.pth'}
    return args

def load_model(config, checkpoint, device='cpu', pretrained=False):
    model = init_recognizer(config, checkpoint, device=device)
    return model

def Interpolate(inputs, refiner=None, mode='duplicate', start=None):
    assert len(inputs.shape) == 6
    b, s, c, t, h, w = inputs.shape
    # inputs = inputs.cpu()
    if mode=='duplicate':
        start = torch.randint(0, 9, (b,))
        outputs = torch.stack([inputs,inputs],dim=4).reshape(b,s,c,t*2,h,w)
        assert outputs[:,:,:,0].equal(outputs[:,:,:,1])
    elif mode == 'linear':
        # print(inputs.shape)
        inputs = inputs.squeeze(1)
        outputs = torch.stack([inputs[i] for i in range(b)], dim=0)
        outputs = F.interpolate(outputs, size=(8,h,w), mode='trilinear', align_corners=False).unsqueeze(1)
        # print(outputs.shape)
    elif mode=='sample':
        if start is None:
            start = torch.randint(0, 9, (b,))
        outputs = torch.stack([inputs[i, :, :, start[i]:start[i]+8] for i in range(b)], dim=0)
        assert len(outputs.shape) == 6
        assert outputs.shape[0] == b
    elif mode=='sample_dup':
        if start is None:
            start = torch.randint(0, t-3, (b,))
        sample_len = 4
        outputs = torch.stack([inputs[i, :, :, start[i]:start[i]+sample_len] for i in range(b)], dim=0)
        outputs = torch.stack([outputs,outputs],dim=4).reshape(b,s,c,2*sample_len,h,w)
        # if not outputs[:,:,:,0].equal(outputs[:,:,:,1]):
        #     indices = torch.where(outputs[:,:,:,0] != outputs[:,:,:,1])
        #     print(indices)
        #     print(outputs[:,:,:,0][indices])
        #     print(outputs[:,:,:,1][indices])
        assert torch.allclose(outputs[:,:,:,0], outputs[:,:,:,1])
    elif mode=='sample_linear':
        if start is None:
            start = torch.randint(0, t-3, (b,))
        sample_len = 4
        inputs = inputs.squeeze(1)
        outputs = torch.stack([inputs[i, :, start[i]:start[i]+sample_len] for i in range(b)], dim=0)
        outputs = F.interpolate(outputs, size=(8,h,w), mode='trilinear', align_corners=False).unsqueeze(1)
    elif mode == 'refine':
        if start is None:
            start = torch.randint(0, t-3, (b,))
        sample_len = 4
        outputs = torch.stack([inputs[i, :, :, start[i]:start[i]+sample_len] for i in range(b)], dim=0)
        outputs = refiner(torch.stack([outputs,outputs],dim=4).reshape(b,s,c,2*sample_len,h,w))
        
    elif mode=='none':
        start = torch.randint(0, 9, (b,))
        outputs = inputs
    else:
        assert 0, 'not Impletmented'
        
    return outputs


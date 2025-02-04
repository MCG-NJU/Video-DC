import torch.nn.functional as F
import torch.nn as nn
# from argument import args as sys_args
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.models as thmodels
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from synthesize.models import ConvNet
from mmaction.apis import init_recognizer
import sys
sys.path.append('..')
from utils_all import Interpolate

# use 0 to pad "other three picture"
def pad(input_tensor, target_height, target_width=None):
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(-2)
    horizontal_padding = target_width - input_tensor.size(-1)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor, (left_padding, right_padding, top_padding, bottom_padding)
    )

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    # print(tensor.shape, batch_size)
    total_samples = tensor.size(0)

    all_outputs = []

    model.eval()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i : min(i + batch_size, total_samples)]

            output = model(batch_data.unsqueeze(1), stage='head')

            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)

    return final_output


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=224, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def denormalize(images):
    images = images.permute(0, 2, 1, 3, 4)
    denormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )
    images = denormalize(images)
    return images.permute(0, 2, 1, 3, 4)


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def selector(n, model, images, labels, size, m=5, inter='none'):
    # last batch when only apply temporal condensation
    n = images.shape[0] if n > images.shape[0] else n
    with torch.no_grad():
        # [mipc, m, 3, (8), 224, 224]
        # print(images.shape)
        images = images.cuda()
        s = images.shape

        # [mipc * m, 3, (8), 224, 224]
        images = images.permute(1, 0, *list(range(len(s)))[2:])
        images = images.reshape(s[0] * s[1], *s[2:])
        # print(images.shape)
        
        # [mipc * m, 1]
        labels = labels.repeat(m).cuda()

        # [mipc * m, n_class]
        batch_size = s[0]  # Change it for small GPU memory
        images_ = Interpolate(images.unsqueeze(1), mode=inter).squeeze(1)
        preds = batched_forward(model, pad(images_, size).cuda(), batch_size)

        # [mipc * m]
        dist = cross_entropy(preds, labels)

        # [m, mipc]
        dist = dist.reshape(m, s[0])

        # [mipc]
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0])]

        # [mipc, 3, 224, 224]
        sa = images.shape
        images = images.reshape(m, s[0], *sa[1:])
        images = images[index, torch.arange(s[0])]
    if n == images.shape[0]:
        torch.cuda.empty_cache()
        return images.detach()
    indices = torch.argsort(dist, descending=False)[:n]
    torch.cuda.empty_cache()
    return images[indices].detach()


def mix_images(input_img, out_size, factor, n):
    # last batch when only apply temporal condensation
    n = input_img.shape[0] if n > input_img.shape[0] else n
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros(
        (n, 3, 8, out_size, out_size),
        requires_grad=False,
        dtype=torch.float,
    )
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            img_part = F.interpolate(
                input_img.data[k * n : (k + 1) * n], size=(8, h_r, w_r)
            )
            assert img_part.shape == (n, 3, 8, h_r, w_r)
            mixed_images.data[
                0:n,
                :,
                :,
                h_loc : h_loc + h_r,
                w_loc : w_loc + w_r,
            ] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images

class ConvNet3D(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, frames, im_size = (32,32),dropout_keep_prob=0.5):
        super(ConvNet3D, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, frames)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]*shape_feat[3]
        # self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2),stride=(1, 1, 1)) if (im_size[0] > 64) else nn.AvgPool3d(kernel_size=(2, 1, 1),stride=(1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logit = nn.Conv3d(net_width, num_classes, kernel_size=(1,1,1), stride=(1,1,1), bias=True)
    
    def forward(self, x, stage=None):
        # x=x.permute(0,2,1,3,4)
        if len(x.shape) == 6:
            x=x.squeeze(1)
        out = self.features(x)
        out = self.logit(self.dropout(self.avg_pool(out)))
        
        logits = out.squeeze(3).squeeze(3)
        logits = torch.max(logits, 2)[0]
        return logits
    
    def embed(self, x):
        x=x.permute(0,2,1,3,4)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
    
    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling, flag):
        if net_pooling == 'maxpooling':
            if flag == 1:
                return nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
            else:
                return nn.MaxPool3d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool3d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm3d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, frames):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, frames, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv3d(in_channels, 64 if d==0 else net_width, kernel_size=(3,7,7), padding=(1,3,3), stride=(1,2,2))]
            shape_feat[2] //= 2
            shape_feat[3] //= 2
            shape_feat[0] = 64 if d==0 else net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = shape_feat[0]
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling, 1 if d == 0 else 0)]
                if d!=0:
                    shape_feat[1] //= 2
                shape_feat[2] //= 2
                shape_feat[3] //= 2

        return nn.Sequential(*layers), shape_feat

def load_model(model_name, pretrained, args, device='cpu'):
    model = init_recognizer(args.config[model_name], args.checkpoint[model_name] if pretrained else None, device=device)
    if args.from_scratch==False and model_name != 'conv4':
        checkpoint = torch.load(args.init[model_name], map_location='cpu')
        stat_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key[:8] == 'cls_head':
                continue
            else:
                stat_dict[key] = value
        model.load_state_dict(state_dict=stat_dict, strict=False)
    return model

    # whether load k400 param
    # if model_name == 'conv4':
    #     model = ConvNet3D(channel=3, num_classes=args.nclass, net_width=128, net_depth=3, net_act='relu', net_norm='batchnorm', net_pooling='maxpooling', frames=8, im_size=(args.input_size,args.input_size)).to('cpu')
    #     if pretrained:
    #         checkpoint = torch.load('/data0/chenyang/ViD/G_VBSM/squeeze/checkpoint/ucf-conv3-best.pth', map_location='cpu')
    #         new_checkpoint = {}
    #         for key, value in checkpoint.items():
    #             new_key = key[7:]
    #             new_checkpoint[new_key] = value
    #         model.load_state_dict(new_checkpoint)
    # else:
        # model = thmodels.video.__dict__[model_name](pretrained=pretrained)
        # if args.subset=='ucf101':
        #     model.fc = nn.Linear(512, 101)
        #     if pretrained == True:
        #         state_dict = torch.load('r2plus1d_fc101_acc73.pth', map_location='cpu')
        #         for key in list(state_dict.keys()):
        #             if key.startswith('module.'):
        #                 state_dict[key[7:]] = state_dict.pop(key)
        #         model.load_state_dict(state_dict)
        #         print('load ucf101 finetune param')

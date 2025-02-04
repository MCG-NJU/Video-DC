import torch
import numpy as np
import os
import torch.distributed
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as t_F
import torch.nn.functional as F
import random
from PIL import Image
from decord import VideoReader, cpu
import warnings
import torch.utils.data as data

# keep top k largest values, and smooth others
def keep_top_k(p, k, n_classes=1000):  # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes - k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(
        topk_smooth_p.sum().item(), p.shape[0]
    ), f"{topk_smooth_p.sum().item()} not close to {p.shape[0]}"
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups


class Syn_Video(data.Dataset):
    def __init__(self, path, transform, ipc, clip_len=8):
        super().__init__()
        self.path = path
        self.transform = transform
        self.ipc = ipc
        self.video_list = self.load_data_list()
        self.all_data = self.video_list
        self.clip_len = clip_len
    
    def load_data_list(self):
        data_list = []
        for label in os.listdir(self.path):
            ilabel = int(label)
            for i in range(len(os.listdir(os.path.join(self.path, label)))//8):
                data_list.append((os.path.join(self.path, label), i, ilabel))
        return data_list
    
    def _load_video(self, video_path, label, ipc_id):
        video = []
        for i in range(self.clip_len):
            frame = Image.open(os.path.join(video_path, 'class{:05d}_id{:05d}_t{:03d}.jpg'.format(label, ipc_id, i)))
            frame = t_F.to_dtype(t_F.pil_to_tensor(frame), dtype=torch.float32, scale=True)
            video.append(frame)
        return torch.stack(video, dim=0)
    
    def __getitem__(self, idx):
        path, ipc_id, label = self.video_list[idx]
        video = self._load_video(path, label, ipc_id)
        video = self.transform(video)
        # [T, C, H, W] -> [C, T, H, W]
        video = video.permute(1, 0, 2, 3)
        return video, label

    def __len__(self):
        return len(self.video_list)
    
    def set_stage(self, stage):
        if stage == 1:
            print('use real data')
            self.video_list = [item for item in self.all_data if item[1] < self.ipc//2]
        elif stage == 2:
            print('use recover data')
            self.video_list = [item for item in self.all_data if item[1] >= self.ipc//2]

class VideoClsDataset(data.Dataset):
    def __init__(self, root, transform, mode, T=8, tau=8, nclips=0):
        self.root = root
        self.T = T
        self.tau = tau
        self.nclips = 10 if mode == 'test' else nclips
        self.transform = transform
        self.mode = mode
        self.video_dirs, self.labels = self.load_annotation()
    
    def load_annotation(self,):
        pass
    
    def read_images(self, path, frames):
        X = []
        for i in frames:
            image = Image.open(
                os.path.join(path, "img_{:05d}.jpg".format(i))
            )
            convert = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(self.pre_size, antialias=True),
            ])
            X.append(convert(image))
        X = torch.stack(X, dim=0)
        if self.transform is not None:
            X = self.transform(X)
        if len(X.shape) == 4:
            X = X.unsqueeze(0)
        # [S, T, C, H, W] -> [S, C, T, H, W]
        X = X.permute(0, 2, 1, 3, 4)
        assert len(X.shape) == 5
        return X
    
    def __getitem__(self, index):
        path = self.video_dirs[index]
        label = self.labels[index]
        
        length = len(os.listdir(path))
        if length < self.T * self.tau:
            # print(length, self.T, self.tau)
            # print(path, label)
            interval = length // self.T
        else:
            interval = self.tau
        assert interval >= 0
        if self.nclips == 0:
            if self.mode == 'train':
                start = np.random.randint(1, length - (self.T - 1) * interval + 1)
            elif self.mode == 'val':
                start = (length - (self.T - 1) * interval) // 2
            if interval == 0:
                frames = np.array([start] * self.T)
            else:
                frames = np.arange(start, start + self.T * interval, interval).tolist()
            
            X = self.read_images(
                path, frames
            )
            # [S, C, T, H, W]
            return X, label
        else:
            X = []
            start_list = np.linspace(1, length - (self.T - 1) * interval, self.nclips, dtype=int)
            for start in start_list:
                if interval == 0:
                    frames = np.array([start] * self.T)
                else:
                    frames = np.arange(start, start + self.T * interval, interval).tolist()
                X.append(self.read_images(path, frames))
            return torch.cat(X, dim=0), label
    
    def prune_dataset(self, num=1):
        lset = set(self.labels)
        labels = []
        video_dirs = []
        for target in lset:
            indexes = [i for i,x in enumerate(self.labels) if x == target]
            random.shuffle(indexes)
            while num > len(indexes):
                indexes.extend(indexes[:min(num-len(indexes), len(indexes))])
                random.shuffle(indexes)
            # print(num, len(indexes))
            assert num <= len(indexes)
            indexes = indexes[:num]
            labels.extend([self.labels[i] for i in indexes])
            video_dirs.extend([self.video_dirs[i] for i in indexes])
        self.labels = labels
        self.video_dirs = video_dirs
        
    def __len__(self):
        return len(self.video_dirs)
    
class UCF101(VideoClsDataset):
    def __init__(self, root, transform, mode, T=8, tau=8, nclips=0):
        super().__init__(root, transform, mode, T, tau, nclips)
        self.pre_size = (128, 170)
        
    def load_annotation(self):
        ann_mode = 'val' if self.mode in ['val', 'test'] else 'train'
        annotation_path = os.path.join(self.root, f'ucf101_{ann_mode}_split_1_rawframes.txt')
        data_path = os.path.join(self.root, "rawframes")
        
        self.video_dirs = []
        self.labels = []

        with open(annotation_path, 'r') as fp:
            for line in fp:
                name, _, label = line.strip().split(" ")
                sample_dir = os.path.join(data_path, name)

                self.labels.append(int(label))
                self.video_dirs.append(sample_dir)
        return self.video_dirs, self.labels
        
class HMDB51(VideoClsDataset):
    def __init__(self, root, transform, mode, T=8, tau=8, nclips=0):
        super().__init__(root, transform, mode, T, tau, nclips)
        self.pre_size = (128, 170)
        
    def load_annotation(self):
        ann_mode = 'val' if self.mode in ['val', 'test'] else 'train'
        annotation_path = os.path.join(self.root, f'hmdb51_{ann_mode}_split_1_rawframes.txt')
        data_path = os.path.join(self.root, "rawframes")
        
        self.video_dirs = []
        self.labels = []

        with open(annotation_path, 'r') as fp:
            for line in fp:
                name, _, label = line.strip().split(" ")
                sample_dir = os.path.join(data_path, name)

                self.labels.append(int(label))
                self.video_dirs.append(sample_dir)
        return self.video_dirs, self.labels

class K400(VideoClsDataset):
    def __init__(self, root, transform, mode, T=8, tau=8, nclips=0):
        super().__init__(root, transform, mode, T, tau, nclips)
        self.pre_size = (64, 64)
    
    def load_annotation(self):
        ann_mode = 'val' if self.mode in ['val', 'test'] else 'train'
        annotation_path = os.path.join(self.root, f'kinetics400_{ann_mode}_list_rawframes.txt')
        
        self.video_dirs = []
        self.labels = []
        with open(annotation_path, 'r') as fp:
            for line in fp:
                name, label = line.strip().split(" ")
                sample_dir = os.path.join(self.root, name)
                
                self.labels.append(int(label))
                self.video_dirs.append(sample_dir)
        return self.video_dirs, self.labels
 
class ThreeCrop:
    def __init__(self, size):
        self.five_crop = transforms.FiveCrop(size)
    
    def __call__(self, img):
        # Get the five crops
        crops = self.five_crop(img)
        # Select three out of the five (top-left, bottom-right, center)
        return torch.stack([crops[0], crops[2], crops[4]])

def get_dataset(data, mode, root='/data0/chenyang/RDED', cr=0, mipc=0):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] # use imagenet statistics
    size = (112, 112) if data in ['ucf101', 'hmdb51'] else (56, 56)
    
    if mode in ['train', 'select']:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif mode == 'val':
        transform = transforms.Compose([
            transforms.CenterCrop(size),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif mode == 'test':
        transform = transforms.Compose([
            ThreeCrop(size),
            transforms.Normalize(mean=mean, std=std)
        ])
    root = os.path.join(root, 'data', data)
    if data == 'ucf101':
        if mode == 'select':
            dataset = UCF101(root, transform, 'train', nclips=cr)
            if mipc:
                dataset.prune_dataset(mipc)
        else:
            dataset = UCF101(root, transform, mode)
    elif data == 'hmdb51':
        if mode == 'select':
            dataset = HMDB51(root, transform, 'train', nclips=cr)
            if mipc:
                dataset.prune_dataset(mipc)
        else:
            dataset = HMDB51(root, transform, mode)
    elif data == 'k400':
        if mode == 'select':
            dataset = K400(root, transform, 'train', nclips=cr)
            if mipc:
                dataset.prune_dataset(mipc)
        else:
            dataset = K400(root, transform, mode)
    return dataset

# def get_dataset(subset, mipc, mode, root='/data0/chenyang/RDED', cr=0):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]  # use imagenet statistics
#     if subset in ['ucf101', 'hmdb51']:
#         train_transform = transforms.Compose([
#             transforms.RandomResizedCrop((112,112), antialias=True),
#             transforms.RandomHorizontalFlip(),
#             transforms.Normalize(mean=mean, std=std)
#         ])
#         val_transform = transforms.Compose([
#             transforms.CenterCrop((112,112)),
#             transforms.Normalize(mean=mean, std=std)
#         ])
#         test_transform = transforms.Compose([
#             ThreeCrop((112,112)),
#             transforms.Normalize(mean=mean, std=std)
#         ])
#         root = os.path.join(root, 'data', subset)
#         if mode == 'select':
#             dataset = UCF101(root, subset, transform=train_transform, mode='train', nclips=cr)
#             if mipc:
#                 dataset.prune_dataset(mipc)
#         elif mode == 'train':
#             dataset = UCF101(root, subset, transform=train_transform, mode=mode)
#         elif mode == 'val':
#             dataset = UCF101(root, subset, transform=val_transform, mode='val')
#         elif mode == 'test':
#             dataset = UCF101(root, subset, transform=test_transform, mode=mode, nclips=10)
#     elif subset == 'k400':
#         transform = transforms.Compose([transforms.Resize((64, 64)),
#                                         transforms.CenterCrop((56, 56)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean=mean, std=std)
#         ])
#         dataset = VideoClsDataset(os.path.join(root, 'data/kinetics400'), mode=mode, transforms=transform, nclips=cr)
#     return dataset

def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        # [t, c, h, w]
        h, w = img.shape[-2:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        if self.factor == 1:
            return img
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 1, 3, 2)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 1, 3, 2)
        return img

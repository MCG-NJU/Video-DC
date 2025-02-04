import os
import torch
import numpy as np
import random
from PIL import Image
import torch.utils.data as data

import torchvision.transforms.v2.functional as t_F
import torchvision.transforms.v2 as transforms

print('==================== use dataset.py ====================')
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
    
class Syn_Video(data.Dataset):
    def __init__(self, path, transform, ipc, clip_len=8):
        super().__init__()
        self.path = path
        self.transform = transform
        self.ipc = ipc
        self.clip_len = clip_len
        self.video_list = self.load_data_list()
        self.all_data = self.video_list
    
    def load_data_list(self):
        data_list = []
        for label in os.listdir(self.path):
            if len(label) > 5:
                continue
            ilabel = int(label)
            # for i in range(len(os.listdir(os.path.join(self.path, label)))//self.clip_len):
            for i in range(self.ipc):
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
    
    def get_init(self, cls_idx, num=1):
        indexes = [i for i,x in enumerate(self.labels) if x == cls_idx]
        random.shuffle(indexes)
        assert num <= len(indexes)
        indexes = indexes[:num]
        outputs = []
        for i in indexes:
            outputs.append(self.__getitem__(i)[0])
        return torch.stack(outputs, dim=0)
        
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

def load_dataset(data, mode, tau=8, root='mmaction2', cr=0, mipc=0):
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
            dataset = UCF101(root, transform, 'train', tau=tau, nclips=cr)
            dataset.prune_dataset(mipc)
        else:
            dataset = UCF101(root, transform, mode, tau=tau)
    elif data == 'hmdb51':
        if mode == 'select':
            dataset = HMDB51(root, transform, 'train', tau=tau, nclips=cr)
            dataset.prune_dataset(mipc)
        else:
            dataset = HMDB51(root, transform, mode, tau=tau)
    elif data == 'k400':
        if mode == 'select':
            dataset = K400(root, transform, 'train', tau=tau, nclips=cr)
            dataset.prune_dataset(mipc)
        else:
            dataset = K400(root, transform, mode, tau=tau)
    return dataset

def init_real(args, cls_idx, num):
    dataset = load_dataset(args.dataset, 'train', tau=8)
    init = dataset.get_init(cls_idx, num)
    return init

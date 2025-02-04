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
import os

# if __name__=='__main__':
#     accelerator = Accelerator(log_with='wandb')
#     accelerator.init_trackers(project_name="my_project",)
#     # config = '/data0/chenyang/ViD/configs/recognition/r2plus1d/k400-pre_r2plus1d_r18_8xb8-8x8x1-180e_ucf101.py' # 18:78.6 34:79.3
#     config = '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-kinetics-8x56x56.py'
#     model = init_recognizer(config, None, device='cpu')
#     # checkpoint = torch.load('/data0/chenyang/ViD/mmaction2/work_dirs/r2plus1d_r18_8xb8-8x8x1-180e_kinetics400-rgb/best_acc_top1_epoch_180.pth')
#     # stat_dict = {}
#     # for key, value in checkpoint['state_dict'].items():
#     #     if key[:8] == 'cls_head':
#     #         continue
#     #     else:
#     #         stat_dict[key] = value
#     # model.load_state_dict(state_dict=stat_dict, strict=False)
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop((56,56), antialias=True),
#         transforms.RandomHorizontalFlip(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     transform = transforms.Compose([
#         transforms.CenterCrop((56,56)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     train_dataset = K400(root='data/k400', transform=train_transform, mode='train')
#     val_dataset = K400(root='data/k400', transform=transform, mode='val')
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

#     num_epoch = 150
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0001)
#     scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=10)
#     scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 130], gamma=0.1)
#     scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
#     # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     # scheduler = torch.optim.lr_scheduler.LambdaLR(
#     #         optimizer,
#     #         lambda step: 0.5 * (1.0 + math.cos(math.pi * step / num_epoch / 2))
#     #         if step <= num_epoch
#     #         else 0,
#     #         last_epoch=-1,
#     #     )
#     model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
#     best_acc = 0
#     lr = []
#     for epoch in tqdm(range(num_epoch)):
#         lr.append(optimizer.param_groups[0]['lr'])
#         model.train()
#         for videos, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(videos, stage='head')
#             loss = torch.nn.functional.cross_entropy(outputs, labels)
#             accelerator.backward(loss)
#             accelerator.clip_grad_norm_(model.parameters(), max_norm=40)
#             optimizer.step()
#             accelerator.log({'train_loss': loss})
#         scheduler.step()
#         if epoch % 10 == 0:
#             model.eval()
#             true = 0
#             for images, labels in (val_loader):
#                 outputs = model(images, stage='head')
#                 _, predicted = torch.max(outputs, 1)
#                 all_pred, all_labels = accelerator.gather_for_metrics((predicted, labels))
#                 true += all_pred.eq(all_labels).sum().item()
#             accuracy = true/len(val_loader.dataset)
#             if accuracy > best_acc:
#                 best_acc = accuracy
#                 if accelerator.is_main_process:
#                     torch.save(model.state_dict(), 'conv4-k400.pth')
#             accelerator.log({'val_acc': accuracy})
#             accelerator.print(f'Epoch {epoch}/{num_epoch} Loss: {loss.item()} Accuracy: {accuracy}')
#     print(best_acc)
#     accelerator.end_training()
#     plt.plot(lr)
#     plt.savefig('lr.png')

# k400
if __name__=='__main__':
    accelerator = Accelerator(log_with='wandb')
    accelerator.init_trackers(project_name="my_project",)
    # config = '/data0/chenyang/ViD/configs/recognition/r2plus1d/k400-pre_r2plus1d_r18_8xb8-8x8x1-180e_ucf101.py' # 18:78.6 34:79.3
    # config = '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-kinetics-8x56x56.py'
    # model = init_recognizer(config, None, device='cpu')
    # checkpoint = torch.load('/data0/chenyang/ViD/mmaction2/work_dirs/r2plus1d_r18_8xb8-8x8x1-180e_kinetics400-rgb/best_acc_top1_epoch_180.pth')
    # stat_dict = {}
    # for key, value in checkpoint['state_dict'].items():
    #     if key[:8] == 'cls_head':
    #         continue
    #     else:
    #         stat_dict[key] = value
    # model.load_state_dict(state_dict=stat_dict, strict=False)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((56,56), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.CenterCrop((56,56)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = K400(root='data/k400', transform=train_transform, mode='train')
    val_dataset = K400(root='data/k400', transform=transform, mode='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    num_experts = 20
    save_dir = './k400/conv4'
    os.makedirs(save_dir, exist_ok=True)
    trajectories = []
    for it in range(num_experts):
        config = '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-kinetics-8x56x56-inconv.py'
        model = init_recognizer(config, None, device='cpu')
        num_epoch = 150
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0001)
        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=10)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 130], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         lambda step: 0.5 * (1.0 + math.cos(math.pi * step / num_epoch / 2))
        #         if step <= num_epoch
        #         else 0,
        #         last_epoch=-1,
        #     )
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        
        best_acc = 0
        lr = []
        timestamps = []
        timestamps.append([p.detach().cpu() for p in model.parameters()])
        for epoch in tqdm(range(num_epoch//3)):
            lr.append(optimizer.param_groups[0]['lr'])
            model.train()
            for videos, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(videos, stage='head')
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=40)
                optimizer.step()
                accelerator.log({'train_loss': loss})
            scheduler.step()
            if epoch % 10 == 0:
                model.eval()
                true = 0
                for images, labels in (val_loader):
                    outputs = model(images, stage='head')
                    _, predicted = torch.max(outputs, 1)
                    all_pred, all_labels = accelerator.gather_for_metrics((predicted, labels))
                    true += all_pred.eq(all_labels).sum().item()
                accuracy = true/len(val_loader.dataset)
                if accuracy > best_acc:
                    best_acc = accuracy
                    # if accelerator.is_main_process:
                    #     torch.save(model.state_dict(), 'conv4-k400.pth')
                accelerator.log({'val_acc': accuracy})
                accelerator.print(f'Epoch {epoch}/{num_epoch} Loss: {loss.item()} Accuracy: {accuracy}')
            timestamps.append([p.detach().cpu() for p in model.parameters()])
        if accelerator.is_main_process:
            trajectories.append(timestamps)
        if len(trajectories) == 10 and accelerator.is_main_process:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
        accelerator.print(f'The {it+1}th model finished: ', best_acc)
    accelerator.end_training()

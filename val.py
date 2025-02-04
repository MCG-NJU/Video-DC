import torch
from mmaction.apis import init_recognizer
import argparse
import yaml
from config.utils import update_config
from dataset import *
from utils_all import *
from utils import *
import math
import accelerate
from tqdm import tqdm
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
import random
import json

train_loss = []
val_acc = []
def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ucf-rded.yaml')
    # parser.add_argument('--data_path', type=str, default='./data/syn_data')
    return parser.parse_args()

def train(accelerator, dataloader, teacher_model, student_model, optimizer, criterion, mix_type, lam, temperature, label_method='distill', im_size=112, inter_mode='none'):
    objs = AverageMeter()
    teacher_model.eval()
    student_model.train()
    for i, data in enumerate(dataloader):
        if label_method=='soft':
            images, labels, hard_label = data
            augment = []
            augment.append(
                transforms.RandomResizedCrop(
                    size=im_size,
                    scale=(0.08, 1),
                    antialias=True,
                )
            )
            augment.append(transforms.RandomHorizontalFlip())
            augment = transforms.Compose(augment)
            images = augment(images)
        else:
            images, labels = data
        optimizer.zero_grad()
        # images = images.cuda()
        # labels = labels.cuda()
        # print(images.shape)
        images = Interpolate(images.unsqueeze(1), mode=inter_mode).squeeze(1)
        if mix_type == 'cutmix':
            images, lam, rand_index = cutmix(images, lam)
        else:
            lam = 1
            rand_index = torch.tensor(range(images.size(0)))
        
        if len(images.shape) == 5:
            images = images.unsqueeze(1)
        pred = student_model(images, stage='head')
        if label_method == 'distill':
            with torch.no_grad():
                teacher_label = teacher_model(images, stage='head')
        elif label_method == 'soft':
            teacher_label = labels
            labels = hard_label
        elif label_method == 'hard':
            teacher_label = None
        if criterion == 'ce':
            loss = lam * F.cross_entropy(pred, labels) + (1 - lam) * F.cross_entropy(pred, labels[rand_index])
        elif criterion == 'kl':
            teacher_label = F.softmax(teacher_label / temperature, dim=1)
            soft_pred = F.log_softmax(pred / temperature, dim=1)
            loss = nn.KLDivLoss(reduction='batchmean')(soft_pred, teacher_label)
        elif criterion == 'mse_gt':
            loss = F.mse_loss(pred, teacher_label) + 0.1 * F.cross_entropy(pred, labels)
        accelerator.backward(loss)
        optimizer.step()
        n = images.size(0)
        objs.update(loss.item(), n)
    accelerator.log({'train/loss': objs.avg})
    train_loss.append(objs.avg)
    return objs.avg
    
def validate(accelerator, dataloader, model):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(dataloader):
            # images = images.cuda()
            # target = target.cuda() b s c t h w
            assert len(images.shape) == 6
            assert images.shape[3] == 8
            output = model(images, stage='head')
        
        # for i, batch in (enumerate(dataloader)):
        #     pred = model.val_step(batch)
        #     output = torch.stack([x.pred_score for x in pred])
        #     target = torch.cat([x.gt_label for x in pred])
                       
            output, target = accelerator.gather_for_metrics([output, target])
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = output.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        accelerator.log({'val/top1': top1.avg, 'val/top5': top5.avg})
        val_acc.append(top1.avg)
    return top1.avg, top5.avg

def manual_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
     
def main(config: dict):
    manual_seed()
    accelerator = accelerate.Accelerator(log_with='wandb')
    group = config['WANDB_NAME'].split('-')[-1]
    print(group)
    accelerator.init_trackers(project_name=config['DATA'], init_kwargs={'wandb': {'name': config['WANDB_NAME'], 'group': group}})
    accelerator.print(config)
    config['TRAIN_LOADER']['batch_size'] = config['TRAIN_LOADER']['batch_size'] // accelerator.num_processes
    accelerator.print('Num process:', accelerator.num_processes, 'Batch size:', config['TRAIN_LOADER']['batch_size'])
    device = accelerator.device
    
    # build dataloader
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augment = []
    # augment.append(ShufflePatches(config['FACTOR']))
    augment.append(
        transforms.RandomResizedCrop(
            size=config['SIZE'],
            scale=(0.08, config['MAX_SCALE_CROPS']),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)
    train_transform = normalize if config['TRAIN_CONFIG']['label_method']=='soft' else transforms.Compose(augment)
    print(train_transform)
    train_dataset = Syn_Video(config['SYN_PATH'], train_transform, config['IPC'], config['clip_len'])
    print(len(train_dataset))
    teacher_model = init_recognizer(**config['TEACHER'], device=device)
    if config['TRAIN_CONFIG']['label_method']=='soft':
        if group == 'datm' or group == 'datmslow':
            print('Preload data with tensor dataset')
            video_all = torch.load(os.path.join(config['SYN_PATH'], 'images_best.pt'))
            label_all = torch.load(os.path.join(config['SYN_PATH'], 'labels_best.pt'))
            hard_label = torch.tensor([ [i] * config['IPC'] for i in range(int(label_all.shape[0]/config['IPC']))], dtype=torch.long).view(-1)
            print(label_all.shape, hard_label.shape)
            # hard_label = torch.arange(0, label_all.shape[0])
            # assert config['IPC'] == 1
        else:
            video_all = []
            label_all = []
            hard_label = []
            for i in range(len(train_dataset)):
                video, label = train_dataset[i]
                # print(video.shape)
                video_all.append(video)
                hard_label.append(label)
                label_all.append(teacher_model(video.unsqueeze(0).unsqueeze(0).to(device), stage='head').cpu().detach())
            video_all = torch.stack(video_all)
            label_all = torch.cat(label_all, dim=0)
            hard_label = torch.tensor(hard_label)

        print(video_all.shape, label_all.shape, hard_label.shape)
        train_dataset = torch.utils.data.TensorDataset(video_all, label_all, hard_label)
    # train_dataset = load_dataset(config['DATA'], 'train', config['MM_ROOT'])
    val_dataset = load_dataset(config['DATA'], 'val', root=config['MM_ROOT'])
    train_loader = torch.utils.data.DataLoader(train_dataset, **config['TRAIN_LOADER'])
    # cfg = Config.fromfile(config['TEACHER']['config'])
    # dataloader_cfg = cfg.get('val_dataloader')
    # print(dataloader_cfg)
    # val_loader = Runner.build_dataloader(dataloader_cfg)
    val_loader = torch.utils.data.DataLoader(val_dataset, **config['VAL_LOADER'])
    
    # load model
    student_model = init_recognizer(**config['STUDENT'], device=device)
    
    # setting optimizer
    if config['OPTIMIZER'] == 'SGD':
        optimizer = torch.optim.SGD(get_parameters(student_model), **config['SGD'])
    elif config['OPTIMIZER'] == 'Adam':
        optimizer = torch.optim.AdamW(get_parameters(student_model), **config['Adam'])
    
    # setting scheduler
    if config['SCHEDULER'] == 'warm-step':
        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, **config['WARMUP'])
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['MULTISTEPLR'])
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    elif config['SCHEDULER'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / config['VAL_EPOCH'] / 2)) if step <= config['VAL_EPOCH'] else 0,
            last_epoch=-1,
        )
    elif config['SCHEDULER'] == 'linear':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (1.0 - step / config['VAL_EPOCH']) if step <= config['VAL_EPOCH'] else 0,
            last_epoch=-1,
        )
    best_acc = 0
    best_epoch = 0
    top1 = 0
    teacher_model, student_model, optimizer, train_loader, val_loader = accelerator.prepare(
        teacher_model, student_model, optimizer, train_loader, val_loader
    )
    # initial_params = {name: param.clone() for name, param in student_model.named_parameters()}
    for epoch in tqdm(range(config['VAL_EPOCH']), disable=(not accelerator.is_main_process)):
        train_loss = train(accelerator, train_loader, teacher_model, student_model, optimizer, **config['TRAIN_CONFIG'], im_size=config['SIZE'], inter_mode=config['Inter'])
        if epoch > 200 and epoch % 10 == 9:
            top1, top5 = validate(accelerator, val_loader, student_model)
        scheduler.step()
        
        # parameters_changed = False
        # for name, param in student_model.named_parameters():
        #     if not torch.equal(param, initial_params[name]):
        #         parameters_changed = True
        #     else:
        #         print(f"Parameter {name} did not change")
        # if not parameters_changed:
        #     print("No parameters changed. Exiting early.")
        # break
        if top1 > best_acc:
            best_acc = max(top1, best_acc)
            best_epoch = epoch
            accelerator.print(f'Epoch {epoch}, train_loss: {train_loss}, top1: {top1}, top5: {top5}')
    accelerator.print(f'Best acc: {best_acc} at epoch {best_epoch}')
    accelerator.log({'result/best_epoch': best_epoch, 'result/best_acc1': best_acc})
    
if __name__=='__main__':
    args = parse_args()
    cfg = yaml_to_dict(args.config)
    
    merged_config = update_config(config=cfg, option=args)
    main(merged_config)
    # with open('log/' + cfg['WANDB_NAME']+'.json', 'w') as f:
    #     json.dump({'train_loss': train_loss, 'val_acc': val_acc}, f)
    
import argparse
import os
import math

parser = argparse.ArgumentParser("RDED")
"""Synthesis"""
parser.add_argument(
    "--arch-name",
    type=str,
    default="resnet18",
    help="arch name from pretrained torchvision models",
)
parser.add_argument(
    "--subset",
    type=str,
    default="imagenet-1k",
)
parser.add_argument(
    "--root",
    type=str,
    default='/data0/chenyang/RDED',
    help='path of the project'
)
parser.add_argument(
    "--train-dir",
    type=str,
    default="../../data/imagenet-1k/train/",
    help="path to training dataset",
)
parser.add_argument(
    "--nclass",
    type=int,
    default=1000,
    help="number of classes for synthesis",
)
parser.add_argument(
    "--mipc",
    type=int,
    default=0,
    help="number of pre-loaded images per class",
)
parser.add_argument(
    '--tau',
    type=int,
    default=8,
    help='gap between frames'
)
parser.add_argument(
    "--ipc",
    type=int,
    default=50,
    help="number of images per class for synthesis",
)
parser.add_argument(
    "--num-crop",
    type=int,
    default=1,
    help="number of croped images for first scoring",
)
parser.add_argument(
    "--input-size",
    default=224,
    type=int,
    metavar="S",
)
parser.add_argument(
    "--factor",
    default=2,
    type=int,
)
parser.add_argument(
    '--wandb_name',
    type=str,
    default='default',
)
"""Re Train"""
parser.add_argument("--re-batch-size", default=0, type=int, metavar="N")
parser.add_argument(
    "--re-accum-steps",
    type=int,
    default=1,
    help="gradient accumulation steps for small gpu memory",
)
parser.add_argument(
    "--mix-type",
    default="cutmix",
    type=str,
    choices=["mixup", "cutmix", "None"],
    help="mixup or cutmix or None",
)
parser.add_argument(
    "--stud-name",
    type=str,
    default="resnet18",
    help="arch name from torchvision models",
)
parser.add_argument(
    "--val-ipc",
    type=int,
    default=30,
)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--classes",
    type=list,
    help="number of classes for synthesis",
)
parser.add_argument(
    "--temperature",
    type=float,
    help="temperature for distillation loss",
)
parser.add_argument(
    "--loss-type",
    type=str,
    default='kl',
)
parser.add_argument(
    "--val-dir",
    type=str,
    default="../../data/imagenet-1k/val/",
    help="path to validation dataset",
)
parser.add_argument(
    "--min-scale-crops", type=float, default=0.08, help="argument in RandomResizedCrop"
)
parser.add_argument(
    "--max-scale-crops", type=float, default=1, help="argument in RandomResizedCrop"
)
parser.add_argument("--re-epochs", default=300, type=int)
parser.add_argument(
    "--syn-data-path",
    type=str,
    default="syn_data",
    help="where to store synthetic data",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
)
parser.add_argument(
    "--inter-mode",
    type=str,
    default='none',
)
parser.add_argument("--scheduler", type=str, default='cos', choices=['cos', 'linear', 'step'], help="cosine lr scheduler")

# sgd
parser.add_argument("--sgd", default=False, action="store_true", help="sgd optimizer")
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.02,
    help="sgd init learning rate",
)
parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="sgd weight decay")

# adamw
parser.add_argument("--adamw-lr", type=float, default=0.001, help="adamw learning rate")
parser.add_argument(
    "--adamw-weight-decay", type=float, default=0.01, help="adamw weight decay"
)
parser.add_argument(
    "--exp-name",
    type=str,
    help="name of the experiment, subfolder under syn_data_path",
)
parser.add_argument(
    "--mem",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--from-scratch",
    default=False,
    action='store_true'
)
args = parser.parse_args()

args.train_dir = f"./data/{args.subset}/train/"
args.val_dir = f"./data/{args.subset}/val/"
args.dataset_root = f"./data/{args.subset}/"


# set up batch size
args.workers=4
if args.re_batch_size == 0:
    if args.ipc == 0:
        args.re_batch_size = 128
    else:
        args.re_batch_size = 10 * args.ipc
    # if args.ipc == 50:
    #     args.re_batch_size = 100
    # elif args.ipc == 70:
    #     args.re_batch_size = 100
    # elif args.ipc == 100:
    #     args.re_batch_size = 100
    # elif args.ipc == 20:
    #     args.re_batch_size = 50
    # elif args.ipc == 10:
    #     args.re_batch_size = 50
    # elif args.ipc == 30:
    #     args.re_batch_size = 50
    # elif args.ipc == 5:
    #     args.re_batch_size = 20
    # elif args.ipc == 1:
    #     args.re_batch_size = 10
    #     args.workers = 0

    # if args.nclass == 10:
    #     args.re_batch_size *= 1
    # if args.nclass > 200:
    #     args.re_batch_size *= 2

# reset batch size below ipc * nclass
# if args.re_batch_size > args.ipc * args.nclass:
#     args.re_batch_size = int(args.ipc * args.nclass)

# reset batch size with re_accum_steps
if args.re_accum_steps != 1:
    args.re_batch_size = int(args.re_batch_size / args.re_accum_steps)

# result dir for saving
# args.exp_name = f"{args.subset}_{args.arch_name}_f{args.factor}_mipc{args.mipc}_ipc{args.ipc}_cr{args.num_crop}"
args.exp_name = args.wandb_name
if not os.path.exists(f"./exp/{args.exp_name}"):
    os.makedirs(f"./exp/{args.exp_name}")
args.syn_data_path = os.path.join("./exp/" + args.exp_name, args.syn_data_path)

# temperature
if args.mix_type == "mixup":
    args.temperature = 4
elif args.mix_type == "cutmix":
    args.temperature = 20
else:
    args.loss_type = 'ce'

if args.ipc == 0:
    args.ipc = 70

root = '/data0/chenyang/ViD/'
if args.subset == 'hmdb51':
    args.input_size = 112
    args.nclass = 51
    args.config = {'conv4': '/data0/chenyang/ViD/configs/recognition/conv3/conv3-hmdb51-8x112x112.py', 'i3d': os.path.join(root, 'configs/recognition/i3d/i3d_k400-pretrained-r18_8x8_hmdb51.py'), 'slowonly': os.path.join(root, 'configs/recognition/slowonly/k400-pre_slowonly-r18_8x8_hmdb51-frame.py')}
    args.checkpoint = {'conv4': '/data0/chenyang/ViD/work_dirs/conv3-hmdb51-8x112x112/best_acc_top1_epoch_130.pth', 'i3d': os.path.join(root, 'work_dirs/i3d_k400-pretrained-r18_8x8_hmdb51_frame/sgd50-56.08/best_acc_top1_epoch_47.pth'), 'slowonly': os.path.join(root, 'work_dirs/k400-pre_slowonly-r18_8x8_hmdb51-frame/best_acc_top1_epoch_31.pth')}
elif args.subset == 'k400':
    args.input_size = 56 
    args.nclass = 400
    config = 'slowonly_r18_8xb16-8x8x1-256e_kinetics400-rgb.py'
    args.config = {'conv4': '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-kinetics-8x56x56.py', 'i3d': '/data0/chenyang/ViD/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r18_8xb8-8x8x1-100e_k400.py', 'slowonly': os.path.join('/data0/chenyang/ViD/mmaction2/configs/recognition/slowonly/', config), 'r2plus1d':'/data0/chenyang/ViD/mmaction2/configs/recognition/r2plus1d/r2plus1d_r18_8xb8-8x8x1-180e_kinetics400-rgb.py'}
    args.checkpoint = {'conv4': '/data0/chenyang/VDC/mmaction2/work_dirs/conv3-kinetics-8x56x56/best_acc_top1_epoch_70.pth', 'i3d': '/data0/chenyang/ViD/mmaction2/work_dirs/i3d_imagenet-pretrained-r18_8xb8-8x8x1-100e_k400/best_acc_top1_epoch_99.pth', 'slowonly': '/data0/chenyang/ViD/mmaction2/work_dirs/slowonly_imagenet-pretrained-r18_8xb16-8x8x1-steplr-150e_kinetics400-rgb/best_acc_top1_epoch_150.pth', 'r2plus1d': '/data0/chenyang/ViD/mmaction2/work_dirs/r2plus1d_r18_8xb8-8x8x1-180e_kinetics400-rgb/best_acc_top1_epoch_180.pth'}
elif args.subset == 'ucf101':
    args.input_size = 112
    args.nclass = 101
    args.config = {'conv4': '/data0/chenyang/VDC/mmaction2/configs/recognition/conv3/conv3-ucf101-8x112x112.py', 'slowonly': '/data0/chenyang/ViD/configs/recognition/slowonly/k400-pre_slowonly-r18_8x8_ucf101-frame.py', 'r2plus1d': os.path.join(root, 'configs/recognition/r2plus1d/k400-pre_r2plus1d_r18_8xb8-8x8x1-180e_ucf101.py'), 'i3d': os.path.join(root, 'configs/recognition/i3d/i3d-r18_8x8_ucf101_frame.py')}
    args.checkpoint = {'conv4': '/data0/chenyang/VDC/mmaction2/work_dirs/conv3-ucf101-8x112x112/best_acc_top1_epoch_145.pth', 'slowonly': '/data0/chenyang/ViD/work_dirs/k400-pre_slowonly-r18_8x8_ucf101-frame/best_acc_top1_epoch_43.pth', 'r2plus1d': os.path.join(root, 'work_dirs/r2plus1d_r18_8xb8-8x8x1-180e_ucf101/best_acc_top1_epoch_41.pth'),'i3d': os.path.join(root, 'work_dirs/i3d_k400-pretrained-r18_8x8_ucf101_frame/best_acc_top1_epoch_48.pth')}
    args.init = {'i3d': '/data0/chenyang/ViD/mmaction2/work_dirs/i3d_imagenet-pretrained-r18_8xb8-8x8x1-100e_k400/best_acc_top1_epoch_99.pth', 'slowonly': '/data0/chenyang/ViD/mmaction2/work_dirs/slowonly_imagenet-pretrained-r18_8xb16-8x8x1-steplr-150e_kinetics400-rgb/best_acc_top1_epoch_150.pth', 'r2plus1d': '/data0/chenyang/ViD/mmaction2/work_dirs/r2plus1d_r18_8xb8-8x8x1-180e_kinetics400-rgb/best_acc_top1_epoch_180.pth'}

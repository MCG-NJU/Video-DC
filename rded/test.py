import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import accelerate
import os
import random
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
import torch.nn.functional as F

# imagenet soft label结果不好 应该是有bug val准确率千分之一
def train_epoch(accelerator, model, optimizer, train_loader, teacher_model=None):
    model.train()
    loss_function_kl = torch.nn.KLDivLoss(reduction="batchmean")
    for images, labels in (train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        targets = teacher_model(images)
        soft_targets = F.softmax(targets / 20, dim=1)
        soft_pred_label = F.log_softmax(outputs / 20, dim=1)
        loss = loss_function_kl(soft_pred_label, soft_targets)
        # loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        schedule.step()
    return loss

def validation(accelerator, model, val_loader):
    model.eval()
    true = 0
    for images, labels in tqdm(val_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_pred, all_labels = accelerator.gather_for_metrics((predicted, labels))
        true += all_pred.eq(all_labels).sum().item()
    return true/len(val_loader.dataset)
        
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
accelerator = accelerate.Accelerator()
teacher_model = torchvision.models.resnet18(weights="DEFAULT")
teacher_model.eval()
model = torchvision.models.resnet18()
# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
            size=224,
            scale=(0.08,1),
            antialias=True,
        ),
    transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet validation dataset
train_dataset = datasets.ImageFolder('/data0/chenyang/datasets/ImageNet/train', transform=torchvision.models.ResNet18_Weights.DEFAULT.transforms())

targets = [s[1] for s in train_dataset.samples]
random.shuffle(targets)
indexes = {}
for i, target in enumerate(targets):
    if target not in indexes:
        indexes[target] = []
    if len(indexes[target]) == 10:
        continue
    indexes[target].append(i)
keep = []
for k,v in indexes.items():
    keep.extend(v)
samples = [train_dataset.samples[i] for i in keep]
train_dataset.samples = samples
train_dataset.targets = [s[1] for s in samples]

# imgs = []
# labels = []
# for x, y in tqdm(train_dataset):
#     imgs.append(x)
#     labels.append(torch.tensor(y))
# train_dataset = torch.utils.data.TensorDataset(torch.stack(imgs), torch.stack(labels))
val_dataset = datasets.ImageFolder('/data0/chenyang/datasets/ImageNet/val', transform=torchvision.models.ResNet18_Weights.DEFAULT.transforms())

# Create a data loader for the validation dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
num_epoch = 300
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999], weight_decay=0.01)
schedule = LambdaLR(optimizer,
                    lambda step: 0.5 * (
                            1. + math.cos(math.pi * step / (2 * num_epoch))) if step <= num_epoch else 0,
                    last_epoch=-1)
train_loader, val_loader, model, teacher_model, optimizer, schedule= accelerator.prepare(train_loader, val_loader, model, teacher_model, optimizer, schedule)
# Iterate over the validation dataset

for epoch in tqdm(range(num_epoch)):
    loss = train_epoch(accelerator, model, optimizer, train_loader, teacher_model)
    if (epoch > 0.7*num_epoch and epoch % 10 == 9):
        acc = validation(accelerator, model, val_loader)
        accelerator.print(f'Epoch: {epoch}, val_Accuracy: {acc}')
    accelerator.print(f'Epoch: {epoch}, loss: {loss}')

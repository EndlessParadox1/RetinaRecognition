import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


random.seed(time.time())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getFn_Labels(path1):
    file_labels = []
    for dir_ in os.listdir(path1):
        path2 = os.path.join(path1, dir_)
        for file in os.listdir(path2):
            fn_ = os.path.join(path2, file)
            file_labels.append((fn_, dir_))
    random.shuffle(file_labels)
    return file_labels


transform = transforms.Compose(
    [transforms.Resize(100),
     transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])


class RetinaDataset(Dataset):
    def __init__(self, fn_labels_):
        self.fn_labels = fn_labels_

    def __getitem__(self, idx):
        img1_, label1 = self.fn_labels[idx]
        if random.randint(0, 1):  # 生成同类的三元组
            k = idx + 1
            while True:
                if k >= len(self.fn_labels):
                    k = 0
                img2_, label2_ = self.fn_labels[k]
                k += 1
                if label1 == label2_:
                    break
        else:  # 生成不同类的三元组
            k = idx + 1
            while True:
                if k >= len(self.fn_labels):
                    k = 0
                img2_, label2_ = self.fn_labels[k]
                k += 1
                if label1 != label2_:
                    break
        img1_ = transform(Image.open(img1_))
        img2_ = transform(Image.open(img2_))
        label_ = torch.Tensor(np.array([int(label1 != label2_)], dtype=np.float32))
        return img1_, img2_, label_

    def __len__(self):
        return len(self.fn_labels)


train_path = 'process/train'
fn_labels = getFn_Labels(train_path)
retinaDataset = RetinaDataset(fn_labels)
train_loader = DataLoader(retinaDataset, batch_size=8, shuffle=True)


vgg19 = models.vgg19()
vgg19_cnn = vgg19.features
for param in vgg19_cnn.parameters():
    param.requires_grad = False  # 冻结参数


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 3, 3, padding=1),
            vgg19_cnn,  # (3, 102, 152) -> (512, 3, 4)
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512)
        )

    def forward_one(self, x):
        o = self.cnn(x)
        o = o.reshape(x.size(0), -1)
        o = self.fc(o)
        return o

    def forward(self, i1, i2):
        o1 = self.forward_one(i1)
        o2 = self.forward_one(i2)
        return o1, o2


class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.margin = 2

    def forward(self, o1, o2, y):
        dist_ = torch.pairwise_distance(o1, o2, keepdim=True)
        loss_ = torch.mean((1 - y) * torch.pow(dist_, 2)
                           + y * torch.pow(torch.clamp(self.margin - dist_, min=0), 2))
        return loss_


siameseNet = SiameseNet().to(device)
optimizer = optim.Adam(siameseNet.parameters(), lr=0.001)


for ep in range(30):
    for i, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        pre_o1, pre_o2 = siameseNet(img1, img2)
        loss = LossFunc()(pre_o1, pre_o2, label)
        if i == 8:
            print(ep + 1, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def showImage(stitle, fn1, fn2_):
    img1_ = np.array(Image.open(fn1).convert('RGB'))
    img2_ = np.array(Image.open(fn2_).convert('RGB'))
    figure, ax = plt.subplots(1, 2)
    ax.ravel()[0].imshow(img1_)
    ax.ravel()[0].set_title(fn1)
    ax.ravel()[0].set_axis_off()
    ax.ravel()[1].imshow(img2_)
    ax.ravel()[1].set_title(fn2_)
    ax.ravel()[1].set_axis_off()
    plt.tight_layout()
    plt.suptitle(stitle, fontsize=18, color='red')
    plt.show()


# siameseNet = torch.load('model.pth', map_location=device)
siameseNet.eval()
test_path = 'process/test'
fn_labels = getFn_Labels(test_path)
correct = 0
with torch.no_grad():
    for fn, label in fn_labels:
        dist_min, label_min, fn_min = float('inf'), None, None
        img1 = transform(Image.open(fn)).unsqueeze(0).to(device)
        for fn2, label2 in fn_labels:
            if fn == fn2:
                continue
            img2 = transform(Image.open(fn2)).unsqueeze(0).to(device)
            pre_o1, pre_o2 = siameseNet(img1, img2)
            dist = torch.pairwise_distance(pre_o1, pre_o2, keepdim=True)
            if dist.item() < dist_min:
                dist_min = dist.item()
                label_min = label2
                fn_min = fn2
        correct += int(label == label_min)
        showImage('Similarity: %.2f' % dist_min, fn, fn_min)


print('一共测试了{}张图片，准确率为{:.1f}%'.format(len(fn_labels), 100. * correct/len(fn_labels)))

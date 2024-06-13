import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OneModule(nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n1, n2, 3, padding=1, bias=False),
            nn.BatchNorm2d(n2),
            nn.ReLU(True),
            nn.Conv2d(n2, n2, 3, padding=1, bias=False),
            nn.BatchNorm2d(n2),
            nn.ReLU(True)
        )

    def forward(self, x_):
        return self.cnn(x_)


class UNet(nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.cnn1 = OneModule(n1, 64)
        self.cnn2 = OneModule(64, 128)
        self.cnn3 = OneModule(128, 256)
        self.cnn4 = OneModule(256, 512)
        self.bottleneck = OneModule(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.ucnn4 = OneModule(1024, 512)
        self.ucnn3 = OneModule(512, 256)
        self.ucnn2 = OneModule(256, 128)
        self.ucnn1 = OneModule(128, 64)
        self.contr4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.contr3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.contr2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.contr1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1_1 = nn.Conv2d(64, n2, 1)

    def forward(self, x_):
        skip_cons = []
        d1 = x_
        d1 = self.cnn1(d1)
        skip_cons.append(d1)
        d1 = self.pool(d1)
        d2 = d1
        d2 = self.cnn2(d2)
        skip_cons.append(d2)
        d2 = self.pool(d2)
        d3 = d2
        d3 = self.cnn3(d3)
        skip_cons.append(d3)
        d3 = self.pool(d3)
        d4 = d3
        d4 = self.cnn4(d4)
        skip_cons.append(d4)
        d4 = self.pool(d4)
        bo = d4
        bo = self.bottleneck(bo)
        u4 = bo
        u4 = self.contr4(u4)
        u4 = torch.cat((skip_cons[3], u4), dim=1)
        u4 = self.ucnn4(u4)
        u3 = u4
        u3 = self.contr3(u3)
        u3 = torch.cat((skip_cons[2], u3), dim=1)
        u3 = self.ucnn3(u3)
        u2 = u3
        u2 = self.contr2(u2)
        u2 = torch.cat((skip_cons[1], u2), dim=1)
        u2 = self.ucnn2(u2)
        u1 = u2
        u1 = self.contr1(u1)
        u1 = torch.cat((skip_cons[0], u1), dim=1)
        u1 = self.ucnn1(u1)
        o = self.conv1_1(u1)
        return o


class GetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train_mode=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = os.listdir(img_dir)
        self.train_mode = train_mode

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        if self.train_mode:
            mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace("training.tif", "manual1.gif"))
        else:
            mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace("test.tif", "manual1.gif"))
        img_ = Image.open(img_path).convert("RGB")
        img_ = transform(img_)
        mask_ = Image.open(mask_path).convert("L")
        mask_ = transform(mask_)
        mask_[mask_ >= 0.5] = 1.0
        mask_[mask_ < 0.5] = 0.0
        return img_, mask_


transform = transforms.Compose([transforms.Resize((288, 288)), transforms.ToTensor()])
unet_model = UNet(3, 1).to(device)
optimizer = optim.Adam(unet_model.parameters(), lr=1e-3)


train_dataset = GetDataset(img_dir='data/train/images/',  mask_dir='data/train/masks')
train_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True, shuffle=True)
for ep in range(100):
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = unet_model(x)
        loss = nn.BCEWithLogitsLoss()(pre_y, y)
        if i == 2:
            print(ep + 1, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def showTwoimgs(imgs_, rows, cols):
    figure, ax = plt.subplots(rows, cols)
    for idx, title in enumerate(imgs_):
        ax.ravel()[idx].imshow(imgs_[title])
        ax.ravel()[idx].set_title(title)
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    plt.show()


# unet_model = torch.load('seg.pth', map_location=device)
unet_model.eval()
val_dataset = GetDataset(img_dir='data/test/images', mask_dir='data/test/masks', train_mode=False)
val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True, shuffle=True)
num_correct = num_pixels = dice_score = 0
with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        pre_y = unet_model(x)
        pre_y = torch.sigmoid(pre_y)
        mask = y
        pre_mask = (pre_y >= 0.1).float()
        num_correct += (pre_mask == mask).sum()
        num_pixels += torch.numel(pre_mask)
        tmp = (2 * (pre_mask * mask).sum()) / ((pre_mask + mask).sum())
        dice_score += tmp
        pre_mask = pre_mask[0][0]
        mask = mask[0][0]
        imgs = dict()
        imgs['Original mask'] = np.array(mask)
        imgs['Predictive mask'] = np.array(pre_mask)
        showTwoimgs(imgs, 1, 2)
print(f"准确率为: {100. * num_correct/num_pixels:.2f}%")
print(f"指标Dice score的值为: {dice_score/len(val_loader):.2f}")

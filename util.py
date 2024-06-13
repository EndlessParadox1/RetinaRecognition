import os
import torch
from torchvision import transforms
from torch import nn
from PIL import Image


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


transform = transforms.Compose([transforms.Resize((192, 288)), transforms.ToTensor()])
model = torch.load('seg.pth', map_location=device)
model.eval()


def process_image(image_path, model_):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model_(img)
        out = torch.sigmoid(out)
        out_mask = (out >= 0.1).float()
    out_mask = transforms.ToPILImage()(out_mask.squeeze())
    return out_mask


def process_dir(input_dir, output_dir, model_):
    for root, _, files in os.walk(input_dir):
        for file in files:
            imgPath = os.path.join(root, file)
            relative_path = os.path.relpath(imgPath, input_dir)
            outputPath = os.path.join(output_dir, relative_path)
            output = process_image(imgPath, model_)
            output.save(outputPath)


# 设置输入和输出目录
input_path = 'retina'
output_path = 'process'

# 处理目录
process_dir(input_path, output_path, model)

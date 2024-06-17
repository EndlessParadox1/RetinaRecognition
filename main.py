import redis
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


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


processNet = torch.load('seg.pth', map_location='cpu')
processNet.eval()
vgg19 = models.vgg19()
vgg19_cnn = vgg19.features


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

    def forward(self, x):
        o = self.cnn(x)
        o = o.reshape(x.size(0), -1)
        o = self.fc(o)
        return o


siameseNet = torch.load('model.pth', map_location='cpu')
siameseNet.eval()
redis_client = redis.StrictRedis(host='48.216.251.241', password='2396')
transform1 = transforms.Compose([transforms.Resize((192, 288)), transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Resize((100, 150)), transforms.Normalize((0.5,), (0.5,))])


def process_image(img_path):
    """
    对输入的图像进行预处理，并获取其特征向量
    """
    img = Image.open(img_path).convert('RGB')
    img_ts = transform1(img).unsqueeze(0)
    with torch.no_grad():
        out_ts = torch.sigmoid(processNet(img_ts))
        pro_ts = (out_ts >= 0.1).float()
        input_ts = transform2(pro_ts)
        fea_vec = siameseNet(input_ts)
    return fea_vec


def store_feature(name_, fea_vec):
    """
    将特征向量存储到 Redis数据库中
    """
    fea_bytes = fea_vec.numpy().tobytes()
    redis_client.set(name_, fea_bytes)

def get_feature(name_):
    """
    从 Redis数据库中获取特征向量
    """
    fea_bytes = redis_client.get(name_)
    fea_array = np.frombuffer(fea_bytes, dtype=np.float32).copy()
    return torch.from_numpy(fea_array).unsqueeze(0)


while True:
    print("请选择功能:")
    print("1.输入(存储特征值)")
    print("2.判定(识别图像)")
    print("3.退出")
    match input():
        case '1':
            name = input("请输入名字: ")
            image_path = input("请输入图片路径: ")
            feature_vector = process_image(image_path)
            store_feature(name, feature_vector)
            print("特征值存储成功!")
        case '2':
            image_path = input("请输入要识别的图片路径: ")
            query_feature = process_image(image_path)
            min_dist, min_name, similarity = float('inf'), None, 0.0
            for name in redis_client.keys():
                feature_vector = get_feature(name)
                distance = torch.pairwise_distance(feature_vector, query_feature)
                if distance < min_dist:
                    min_dist = distance
                    min_name = name.decode()
                    cosine_similarity = F.cosine_similarity(query_feature, feature_vector, dim=1)
                    similarity = (cosine_similarity.item() + 1) * 50
            print("识别结果为: ", min_name)
            print("相似度为: %.2f%%" % similarity)
        case '3':
            print("程序退出!")
            break
        case _:
            print("无效的选项，请重新输入")

# coding=gbk
import numpy as np
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image

from config import Config

train_dir = '../coco_c/classify_folder_train_mini/*'


class image(Dataset):
    def __init__(self, root_dir):
        self.files = glob.glob(root_dir)
        self.transform = transforms.Compose([transforms.Resize(256),  # 将图像调整为256×256像素
                                             transforms.CenterCrop(224),  # 将图像中心裁剪出来，大小为224×224像素
                                             transforms.ToTensor()  # 将图像转换为PyTorch张量（tensor）数据类型
                                             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = self.transform(img)
        img = np.array(img)
        return img, img


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(256, 256)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def auto_encoder_train():
    image_data = image(train_dir)
    image_loader = DataLoader(image_data, batch_size=Config.batch_size, shuffle=True)

    for image1, image2 in image_loader:
        print(image1.shape)
        break


if __name__ == '__main__':
    auto_encoder_train()

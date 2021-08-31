import os
import torch
import torchvision
import numpy as np

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = torchvision.models.resnet18(pretrained=True)

model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉网络的最后一层
if torch.cuda.is_available() is True:
    model = model.cuda()

model.eval()


def resnet_18_encoder(imgs):

    img_features = []

    for img in imgs:
        img = torch.unsqueeze(img, 0)
        img_feature = model(img)
        img_feature = img_feature.cpu()
        img_feature = img_feature.detach().numpy()
        img_features.append(img_feature)

    return img_features


def feature_getting(train_loader, train=True):
    print("开始代表集特征提取：")
    feature_set = np.zeros(shape=(0, 512))  # 初始化一个空数组
    label_set = np.zeros(shape=(0, 1))  # 存放标签
    for data in tqdm(train_loader):
        img, label = data
        img = img.cuda()

        img_feature = np.array(resnet_18_encoder(img)).reshape(-1, 512)
        feature_set = np.concatenate((feature_set, img_feature), axis=0)
        label = np.array(label).reshape(-1, 1)
        label_set = np.concatenate((label_set, label), axis=0)

    return feature_set, label_set


if __name__ == "__main__":
    resnet_18_encoder()

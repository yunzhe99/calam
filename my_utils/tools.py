import torch

import numpy as np

from PIL import Image
from torchvision import transforms, models
from torchviz import make_dot
import matplotlib.pyplot as plt


def network_visualization(model):
    """
    使用graphviz进行神经网络可视化
    :param model: 要可视化的神经网络
    :return: 无返回值，生成一个神经网络的pdf文件
    """
    x = torch.rand(8, 3, 256, 512)
    y = model(x)
    g = make_dot(y)
    g.render('espnet_model', view=False)


def get_img_to_inference(img_dir="../Animal/dog/eating/3.jpg"):
    """
    从文件夹中的图片文件，得到能够输入到神经网络的tensor
    :param img_dir: 神经网络地址
    :return: 返回位于gpu上的待推理的图片tensor
    """
    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    img = Image.open(img_dir)

    img_t = transform(img)

    plt.imshow(img)
    plt.show()

    batch_t = torch.unsqueeze(img_t, 0)

    batch_t_gpu = batch_t.cuda()

    return batch_t_gpu


def network_evaluation(__model, __batch_t_gpu):
    """
    使用神经网络网络，得到推理的结果
    只能得到类别编号，具体类别编号属于哪一个类需要手动查神经网络的标签
    :param __model: 推理用的神经网络
    :param __batch_t_gpu: 待推理的tensor
    :return: 直接输出结果，无返回值
    """
    __model.eval()

    out = __model(__batch_t_gpu)

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    print(percentage[index[0]], index)


def network_encoder(_model, _batch_t_gpu):
    """
    采用神经网络的特征提取器进行图片的编码
    :param _model: 神经网络
    :param _batch_t_gpu: 待推理的tensor
    :return: 直接输出结果，无返回值
    """

    _model = torch.nn.Sequential(*list(_model.children())[:-1])  # 去掉网络的最后一层

    _model.eval()

    out = _model(_batch_t_gpu)

    print(out.shape)


def non_iid_index(x, y):
    """
    non iid index (NI) 计算：计算x和y分布的差异，本函数的输入中，x与y一般没有交集
    :param x: x数据集提取出的特征，np数组
    :param y: y数据集提取出的特征，np数组
    :return:
    """

    x_mean = np.mean(x, axis=0)

    y_mean = np.mean(y, axis=0)

    x_y_union = np.concatenate((x, y), axis=0)

    x_y_union_std = np.std(x_y_union, axis=0)

    return np.linalg.norm(((x_mean - y_mean) / x_y_union_std), ord=2)


def get_distance_matrix(cluster_phase1):
    # 计算距离矩阵

    cluster_len = len(cluster_phase1)

    distance_matrix = np.zeros((cluster_len, cluster_len))

    for i in range(cluster_len):
        for j in range(cluster_len):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = non_iid_index(cluster_phase1[i], cluster_phase1[j])

    return distance_matrix


if __name__ == "__main__":
    img_features = np.load("../img_features.npy")

    test1 = img_features[0:1]

    test2 = img_features[800:801]

    NI = non_iid_index(test1, test2)

    print(NI)

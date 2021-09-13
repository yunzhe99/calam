# coding=gbk
import os
import random
import joblib
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from podm.podm import get_pascal_voc_metrics, BoundingBox
from PIL import Image

from sklearn.cluster import KMeans

from sample_node import sample_node, bbox_2leftup_2rightdown
from config import Config
from torchvision import transforms


class sample_test(Dataset):

    def __init__(self, sample_list):
        self.sample_list = sample_list
        self.transform = transforms.Compose([transforms.Resize(256),  # 将图像调整为256×256像素
                                            transforms.CenterCrop(224),  # 将图像中心裁剪出来，大小为224×224像素
                                            transforms.ToTensor()  # 将图像转换为PyTorch张量（tensor）数据类型
                                            ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_dir = self.sample_list[idx].img_dir
        feature = self.sample_list[idx].feature
        image = Image.open(self.sample_list[idx].img_dir)
        image = self.transform(image)
        label = self.sample_list[idx].label_dir
        return [image_dir, image, feature], label


class test_batch_loader:
    def __init__(self, dataset_list, labelset_list, force_loading=False):

        if os.path.exists('test_sample_dhd_traffic_val.s') and force_loading is False:
            self.sample_list = joblib.load('test_sample_dhd_traffic_val.s')
        else:
            self.sample_list = []
            sample_index = 0

            for task_folder_index in range(len(dataset_list)):
                # label存在不全的情况，改用label作为索引去找image

                try:

                    f = open(dataset_list[task_folder_index])
                    file_list = f.readlines()
                    random.shuffle(file_list)
                    f.close()

                    # file_list = np.loadtxt(dataset_list[task_folder_index])
                    length = np.min((Config.test_len, len(file_list)))

                    # for file_obj_index in tqdm(range(length)):  # 构建特征集
                    for file_obj_index in tqdm(range(length)):
                        try:
                            file_obj = file_list[file_obj_index].strip("\n")
                            file_path = file_obj
                            file_dir, file_name = os.path.split(file_path)

                            label_path = os.path.join(os.path.dirname(file_dir), 'labels', file_name[:-4] + '.txt')

                            if os.path.exists(label_path):
                                sample = sample_node(sample_index, file_path, label_path)
                                sample.feature_getting()  # 抽取特征
                                # print('check result: ', sample.sample_check())
                                self.sample_list.append(sample)
                                sample_index = sample_index + 1

                        except Exception as e:
                            print(e)
                            continue

                except Exception as e:
                    print(e)
                    file_list = os.listdir(dataset_list[task_folder_index])

                    length = np.min((Config.test_len, len(file_list)))

                    # for file_obj_index in tqdm(range(length)):  # 构建特征集
                    for file_obj_index in tqdm(range(length)):
                        try:
                            file_obj = file_list[file_obj_index]
                            file_path = os.path.join(dataset_list[task_folder_index], file_obj)
                            label_path = os.path.join(labelset_list[task_folder_index], file_obj[:-4] + '.txt')

                            if os.path.exists(label_path):
                                sample = sample_node(sample_index, file_path, label_path)
                                sample.feature_getting()  # 抽取特征
                                # print('check result: ', sample.sample_check())
                                self.sample_list.append(sample)
                                sample_index = sample_index + 1

                        except:
                            continue

            joblib.dump(self.sample_list, 'test_sample_dhd_traffic_val.s')

    def in_order_test(self):

        test_sample = sample_test(self.sample_list)
        test_loader = DataLoader(test_sample, batch_size=Config.batch_size, shuffle=True)
        return test_loader

    def random_test(self):

        random.shuffle(self.sample_list)
        test_sample = sample_test(self.sample_list)
        test_loader = DataLoader(test_sample, batch_size=Config.batch_size, shuffle=True)
        return test_loader

    def kmeans_test(self):

        feature_list = []

        for sample in self.sample_list:
            feature_list.append(sample.feature)

        feature_list = np.array(feature_list).squeeze()

        n_clusters = 5

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_list)

        sample_list = []

        for index in range(n_clusters):
            for label_index in range(len(kmeans.labels_)):
                if kmeans.labels_[label_index] == index:
                    sample_list.append(self.sample_list[label_index])

        test_sample = sample_test(self.sample_list)
        test_loader = DataLoader(test_sample, batch_size=Config.batch_size, shuffle=True)

        return test_loader


def load_label(label_list):
    """
    从标签文件(txt, yolo格式)中获取BoundingBox信息
    :param label_list: 标签的列表
    :return: BoundingBox列表
    """
    boxs = []
    for label_file in label_list:
        label = pd.read_csv(label_file, names=['class', 'x', 'y', 'w', 'h'], header=None, sep=" ")
        label = np.array(label)
        for obj in label:
            xywh = obj[1:5]
            xyxy = bbox_2leftup_2rightdown(xywh)
            xyxy = list(xyxy)
            (filepath, tempfilename) = os.path.split(label_file)
            (filename, extension) = os.path.splitext(tempfilename)
            bb = BoundingBox(filename, obj[0], xyxy[0], xyxy[1], xyxy[2], xyxy[3], 1)
            boxs.append(bb)
    return boxs


if __name__ == '__main__':
    test_batch = test_batch_loader(1, 1)
    test_batch.kmeans_test()

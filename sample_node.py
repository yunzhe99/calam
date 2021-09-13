# coding=gbk
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from podm.podm import get_pascal_voc_metrics, BoundingBox

from config import Config
from my_utils.preprocess import resnet_18_encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def bbox_2leftup_2rightdown(bbox):
    """计算bbox的左上右下顶点
        bbox：框数据——xywh
    """
    x1 = bbox[0] - bbox[2] / 2.0
    y1 = bbox[1] - bbox[3] / 2.0
    x2 = bbox[0] + bbox[2] / 2.0
    y2 = bbox[1] + bbox[3] / 2.0

    return x1, y1, x2, y2


def box_iou_solve(bbox1, bbox2, mode=True):
    """计算两个框之间的IoU值
        bbox1: 框数据
        bbox2: 框数据
        mode: 框数据表示形式
            True: xyxy
            False: xywh

        IoU的intersection的左上右下顶点: 左上点为

        return IoU, (r_bbox1, r_bbox2, inter_bbox)
            PS：
                IoU： 交并比值
                r_bbox1：转换为xyxy形式的bbox1
                r_bbox2：转换为xyxy形式的r_bbox2
                inter_bbox: 形式为xyxy的交集位置
    """
    if mode is True:  # bbox数据格式: xyxy
        # 左上右下顶点坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        # 框的长宽:长度由具体的像素个数决定，因此需要加1
        b1_w, b1_h = bbox1[2] - bbox1[0] + 1.0, bbox1[3] - bbox1[1] + 1.0
        b2_w, b2_h = bbox2[2] - bbox2[0] + 1.0, bbox1[3] - bbox1[1] + 1.0
    else:  # bbox数据格式: xywh
        # 左上右下顶点坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox_2leftup_2rightdown(bbox1)
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox_2leftup_2rightdown(bbox2)
        # 框的长宽
        b1_w, b1_h = bbox1[2], bbox1[3]
        b2_w, b2_h = bbox2[2], bbox2[3]

    # 各自的面积
    s1 = b1_w * b1_h
    s2 = b2_w * b2_h

    # 交集面积
    # 如果考虑多个框进行计算交集——那么应该使用np.maximum——进行逐位比较
    inter_x1 = max(b1_x1, b2_x1)  # 交集区域的左上角
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)  # 交集区域的右下角
    inter_y2 = min(b1_y2, b2_y2)

    # 长度由具体的像素个数决定，因此需要加1；这里给的是归一化的值，因此不加1
    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    intersection = inter_w * inter_h

    # 并集面积
    union_area = s1 + s2 - intersection

    # 计算IoU交并集
    IoU = intersection / union_area

    # 整合坐标信息——用于展示交集可视化
    # 返回数据均以xyxy表示
    r_bbox1 = b1_x1, b1_y1, b1_x2, b1_y2
    r_bbox2 = b2_x1, b2_y1, b2_x2, b2_y2
    inter_bbox = inter_x1, inter_y1, inter_x2, inter_y2

    return IoU, (r_bbox1, r_bbox2, inter_bbox)


def performance_evaluation(img, label, model):
    results = model(img)

    results = np.array(results.pandas().xywhn[0])

    box_pred_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_pred_list = results[:, 5]

    label = np.loadtxt(label).reshape(-1, 5)  # 标签中只有一行的话，需要升维

    box_label_list = label[:, 1:5]
    class_label_list = label[:, 0]

    label_num = len(box_label_list)
    pred_num = len(box_pred_list)
    label_match = np.zeros(label_num)  # 保存每个标签的匹配结果，如果匹配上了则相应位置1

    for label_index in range(label_num):
        box_label = box_label_list[label_index]
        class_label = class_label_list[label_index]
        for pred_index in range(pred_num):
            box_pred = box_pred_list[pred_index]
            class_pred = class_pred_list[pred_index]
            iou, _ = box_iou_solve(box_label, box_pred, mode=False)
            if iou > 0 and class_label == class_pred:
                label_match[label_index] = 1

    match_result = (label_match == np.ones(label_num)).all()

    if match_result:
        print('match_result: ', match_result)
        return 1
    else:
        return 0


def performance_evaluation_map(img, label, model):
    label_boxes = []

    label = pd.read_csv(label, names=['class', 'x', 'y', 'w', 'h'], header=None, sep=" ")
    label = np.array(label)

    for obj in label:
        xywh = obj[1:5]
        xyxy = bbox_2leftup_2rightdown(xywh)
        xyxy = list(xyxy)
        bb = BoundingBox('filename', obj[0], xyxy[0], xyxy[1], xyxy[2], xyxy[3], 1)
        label_boxes.append(bb)

    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, verbose=False)

    results = model(img)

    results = np.array(results.pandas().xyxyn[0])

    inference_boxs = []

    # print(results)

    box_pred_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_pred_list = results[:, 5]

    for bb_index in range(len(results)):
        bb = BoundingBox('filename', class_pred_list[bb_index], box_pred_list[bb_index][0], box_pred_list[bb_index][1], box_pred_list[bb_index][2], box_pred_list[bb_index][3], confidence_list[bb_index])
        inference_boxs.append(bb)

    # label = np.loadtxt(label).reshape(-1, 5)  # 标签中只有一行的话，需要升维
    #
    # box_label_list = label[:, 1:5]
    # class_label_list = label[:, 0]
    #
    # label_boxes = []
    #
    # for bb_index in range(len(label)):
    #     bb = BoundingBox('filename', class_label_list[bb_index], box_label_list[bb_index][0],
    #                      box_label_list[bb_index][1], box_label_list[bb_index][2], box_label_list[bb_index][3], 1)
    #     label_boxes.append(bb)

    ret = get_pascal_voc_metrics(label_boxes, inference_boxs)

    # print(ret)

    # for label_index in range(label_num):
    #     box_label = box_label_list[label_index]
    #     class_label = class_label_list[label_index]
    #     for pred_index in range(pred_num):
    #         box_pred = box_pred_list[pred_index]
    #         class_pred = class_pred_list[pred_index]
    #         iou, _ = box_iou_solve(box_label, box_pred, mode=False)
    #         if iou > 0 and class_label == class_pred:
    #             label_match[label_index] = 1
    #
    # match_result = (label_match == np.ones(label_num)).all()

    ap_total = 0
    class_num = 0

    for class_index in ret:
        if not np.isnan(ret[class_index].ap):  # 如果预测出了数据集以外的类，那么就会出现nan
            ap_total = ap_total + ret[class_index].ap
            class_num = class_num + 1

    m_ap = ap_total / class_num

    # print(m_ap)

    # if label.size == 0 and results.size == 0:
    #     ap = 1
    # else:
    #     ap = ret[0].ap

    # m_ap = ap

    return m_ap

    # print(ap)
    #
    # if ap > 0.6:
    #     return 1
    # else:
    #     return 0


model_list = []

for weight_index in range(len(Config.weight_list)):
    weight = Config.weight_list[weight_index]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, verbose=False)
    model_list.append(model)


class sample_node:
    def __init__(self, index, img_dir, label_dir):
        self.index = index
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.feature = None
        # 各个模型在该样本上的表现
        self.performance = np.empty(len(Config.weight_list))
        self.performance.fill(-1)
        self.model_set = Config.weight_list
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor()])

    def feature_getting(self):
        img = Image.open(self.img_dir)

        if img.mode == 'L':
            img = [img, img, img]
            img = Image.merge("RGB", img)  # 合并三通道

        img = self.transform(img)
        img = img.unsqueeze(0)

        if torch.cuda.is_available() is True:
            img = img.cuda()

        self.feature = np.array(resnet_18_encoder(img)).reshape(-1, 512)

    def sample_evalution(self):

        for weight_index in range(len(Config.weight_list)):
            model_e = model_list[weight_index]
            perfromance = performance_evaluation_map(self.img_dir, self.label_dir, model_e)
            self.performance[weight_index] = perfromance

    def sample_check(self):
        if self.feature is not None and self.performance[-1] != -1:
            return True
        else:
            return False


class sample_feature(Dataset):
    def __init__(self, sample_list, weight_index, need_image=False):
        self.sample_list = sample_list
        self.weight_index = weight_index
        self.need_image = need_image
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        feature = self.sample_list[idx].feature
        label = float(self.sample_list[idx].performance[self.weight_index])
        if self.need_image is False:
            return feature, label
        else:
            img = Image.open(self.sample_list[idx].img_dir)
            img = self.transform(img)
            return img, label

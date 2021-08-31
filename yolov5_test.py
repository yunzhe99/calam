# coding=gbk
import torch
import psutil
import os
import time
from torchstat import stat

import numpy as np


def bbox_2leftup_2rightdown(bbox):
    """����bbox���������¶���
        bbox�������ݡ���xywh
    """
    x1 = bbox[0] - bbox[2] / 2.0
    y1 = bbox[1] - bbox[3] / 2.0
    x2 = bbox[0] + bbox[2] / 2.0
    y2 = bbox[1] + bbox[3] / 2.0

    return x1, y1, x2, y2


def box_iou_solve(bbox1, bbox2, mode=True):
    """����������֮���IoUֵ
        bbox1: ������
        bbox2: ������
        mode: �����ݱ�ʾ��ʽ
              True: xyxy
              False: xywh

        IoU��intersection���������¶���: ���ϵ�Ϊ

        return IoU, (r_bbox1, r_bbox2, inter_bbox)
             PS��
                IoU�� ������ֵ
                r_bbox1��ת��Ϊxyxy��ʽ��bbox1
                r_bbox2��ת��Ϊxyxy��ʽ��r_bbox2
                inter_bbox: ��ʽΪxyxy�Ľ���λ��
    """
    if mode is True:  # bbox���ݸ�ʽ: xyxy
        # �������¶�������
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        # ��ĳ���:�����ɾ�������ظ��������������Ҫ��1
        b1_w, b1_h = bbox1[2] - bbox1[0] + 1.0, bbox1[3] - bbox1[1] + 1.0
        b2_w, b2_h = bbox2[2] - bbox2[0] + 1.0, bbox1[3] - bbox1[1] + 1.0
    else:  # bbox���ݸ�ʽ: xywh
        # �������¶�������
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox_2leftup_2rightdown(bbox1)
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox_2leftup_2rightdown(bbox2)
        # ��ĳ���
        b1_w, b1_h = bbox1[2], bbox1[3]
        b2_w, b2_h = bbox2[2], bbox2[3]

    # ���Ե����
    s1 = b1_w * b1_h
    s2 = b2_w * b2_h

    # �������
    # ������Ƕ������м��㽻��������ôӦ��ʹ��np.maximum����������λ�Ƚ�
    inter_x1 = max(b1_x1, b2_x1)  # ������������Ͻ�
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)  # ������������½�
    inter_y2 = min(b1_y2, b2_y2)

    # �����ɾ�������ظ��������������Ҫ��1����������ǹ�һ����ֵ����˲���1
    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    intersection = inter_w * inter_h

    # �������
    union_area = s1 + s2 - intersection

    # ����IoU������
    IoU = intersection / union_area

    # ����������Ϣ��������չʾ�������ӻ�
    # �������ݾ���xyxy��ʾ
    r_bbox1 = b1_x1, b1_y1, b1_x2, b1_y2
    r_bbox2 = b2_x1, b2_y1, b2_x2, b2_y2
    inter_bbox = inter_x1, inter_y1, inter_x2, inter_y2

    return IoU, (r_bbox1, r_bbox2, inter_bbox)


if __name__ == '__main__':

    t_start = time.time()
    # weight = ['base_model/model_7_2.pt']
    # print(weight[0])

    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5l', device='cpu')

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./single_v5x.pt',
                                    verbose=False)  # './all_v5s_v.pt' './single_v5s.pt'

    # stat(model, (32, 244, 244))

    img = '61_4084.jpg'
    #
    # # # Images
    # # dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
    # # imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batched list of images
    # #
    # # Inference
    results = model(img)
    results.save()

    print(u'��ǰ���̵��ڴ�ʹ�ã�%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    #
    # results.save()
    # #
    # # Results
    # # results.print()
    # # results.save()  # or .show()
    # #
    # # Data
    print(results.pandas().xywhn[0])
    #
    # results = np.array(results.pandas().xywhn[0])
    #
    # box_pred_list = results[:, 0:4]
    # confidence_list = results[:, 4]
    # class_pred_list = results[:, 5]
    #
    # # print(box_pred_list)
    # # print(confidence_list)
    # # print(class_pred_list)
    #
    # label = np.loadtxt('./base_model/000000000094.txt')
    #
    # print(label.reshape(-1, 5))
    #
    # box_label_list = label[:, 1:5]
    # class_label_list = label[:, 0]
    #
    # print(box_label_list)
    # print(class_label_list)
    #
    # label_num = len(box_label_list)
    # pred_num = len(box_pred_list)
    # label_match = np.zeros(label_num)  # ����ÿ����ǩ��ƥ���������ƥ����������Ӧλ��1
    #
    # for label_index in range(label_num):
    #     box_label = box_label_list[label_index]
    #     class_label = class_label_list[label_index]
    #     for pred_index in range(pred_num):
    #         box_pred = box_pred_list[pred_index]
    #         class_pred = class_pred_list[pred_index]
    #         iou, _ = box_iou_solve(box_label, box_pred, mode=False)
    #         print(iou)
    #         if iou > 0 and class_label == class_pred:
    #             label_match[label_index] = 1
    #
    # match_result = (label_match == np.ones(label_num)).all()
    #
    # print(match_result)

    t_end = time.time()

    print("����ʱ�䣺")
    print(t_end - t_start)

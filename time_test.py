import time
import joblib
import torch
import collections
import os
import numpy as np

from task_model import Net
from sample_node import sample_node


def feature_extration():
    file_path = '000000.png'
    label_path = '000000.txt'
    sample = sample_node(0, file_path, label_path)
    sample.feature_getting()
    feature = sample.feature
    return feature


def model_choosing(_feature):

    max_score = 0

    chooser_list = joblib.load('chooser_list_all_v.m')

    for model_index in range(len(chooser_list)):

        model_chooser = chooser_list[model_index]

        feature = _feature.reshape(-1, 512)

        maps = model_chooser.predict(feature)

        # print(maps)

        score = np.mean(maps)

        if score >= max_score:
            max_score = score
            max_index = model_index

        print(score)


if __name__ == '__main__':
    # time_start = time.time()
    #
    # feature = feature_extration()
    #
    # time_1 = time.time()
    #
    # model_choosing(feature)
    #
    # time_2 = time.time()
    #
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
    #
    # time_3 = time.time()
    #
    # results = model('000000.png')
    #
    # time_4 = time.time()
    #
    # if 1 + 1 == 2:
    #     a = 4
    #
    # time_5 = time.time()
    #
    # time_feature = time_1 - time_start
    # time_choosing = time_2 - time_1
    # time_loading = time_3 - time_2
    # time_inference_s = time_4 - time_3
    #
    # time_rules = time_5 - time_4
    #
    # time_calam = time_feature + time_choosing + time_inference_s
    # time_hms = time_inference_s + time_rules
    # time_mss = time_inference_s
    #
    # # print(time_calam)
    # # print(time_hms)
    # # print(time_mss)
    img_list = ['000000.png', '0.jpg', '3.jpg', '8.jpg', '19.png', '61_4084.jpg', '000000000802.jpg',
                '1496916404548.jpg']

    model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device=2)

    time_6 = time.time()

    results = model(img_list)

    time_7 = time.time()
    time_inference_b = time_7 - time_6
    print(time_inference_b)

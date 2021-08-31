import os
import glob
import shutil
import numpy as np
from tqdm import tqdm

from config import Config


def yolo_to_person_coco(yolo_dir, yolo_person_dir):
    yolo_label = np.loadtxt(yolo_dir).reshape(-1, 5)
    yolo_person_label = []
    for bbox in yolo_label:
        class_name = bbox[0]
        if class_name == Config.person_index_coco:
            yolo_person_label.append(bbox)
    yolo_person_label = np.array(yolo_person_label)
    np.savetxt(yolo_person_dir, yolo_person_label, fmt='%f', delimiter=' ')


def yolo_to_person_kitti(yolo_dir, yolo_person_dir):
    yolo_label = np.loadtxt(yolo_dir).reshape(-1, 5)
    yolo_person_label = []
    for bbox in yolo_label:
        class_name = bbox[0]
        if class_name == Config.person_index_kitti:
            bbox[0] = 0
            yolo_person_label.append(bbox)
    yolo_person_label = np.array(yolo_person_label)
    np.savetxt(yolo_person_dir, yolo_person_label, fmt='%f', delimiter=' ')


def yolo_to_person_crowd_human(yolo_dir, yolo_person_dir):
    print(yolo_dir)
    yolo_label = np.loadtxt(yolo_dir).reshape(-1, 5)
    yolo_person_label = []
    for bbox in yolo_label:
        class_name = bbox[0]
        if class_name == Config.person_index_crowdhuman:
            bbox[0] = 0
            yolo_person_label.append(bbox)
    yolo_person_label = np.array(yolo_person_label)
    np.savetxt(yolo_person_dir, yolo_person_label, fmt='%f', delimiter=' ')


def yolo_to_person_night(yolo_dir, yolo_person_dir):
    # print(yolo_dir)
    yolo_label = np.loadtxt(yolo_dir).reshape(-1, 5)
    yolo_person_label = []
    for bbox in yolo_label:
        class_name = bbox[0]
        if class_name == Config.person_index_night:
            bbox[0] = 0
            yolo_person_label.append(bbox)
    yolo_person_label = np.array(yolo_person_label)
    np.savetxt(yolo_person_dir, yolo_person_label, fmt='%f', delimiter=' ')


def convert_main(_yolo_label_dir, _yolo_person_label_dir):

    for label_dir in tqdm(os.listdir(_yolo_label_dir)):  # Config.coco_label_train
        yolo_label_each_dir = os.path.join(_yolo_label_dir, label_dir)  # Config.coco_label_train
        yolo_person_label_each_dir = os.path.join(_yolo_person_label_dir, label_dir)  # Config.coco_person_train
        yolo_to_person_night(yolo_label_each_dir, yolo_person_label_each_dir)


def folder_gen():

    if not os.path.exists(Config.crowdhuman_person_label_train):
        os.makedirs(Config.crowdhuman_person_label_train)

    jpg_dirs = glob.glob(Config.crowdhuman_data_label)

    for jpg_dir in jpg_dirs:
        _, img_name = os.path.split(jpg_dir)
        # print(img_name)
        new_img_dir = os.path.join(Config.crowdhuman_person_label_train, img_name)
        # print(new_img_dir)
        shutil.copyfile(jpg_dir, new_img_dir)


if __name__ == '__main__':
    # folder_gen()

    yolo_label_dir = Config.night_person_label_val
    yolo_person_label_dir = Config.night_person_label_val_converted

    if not os.path.exists(yolo_person_label_dir):
        os.makedirs(yolo_person_label_dir)

    convert_main(yolo_label_dir, yolo_person_label_dir)

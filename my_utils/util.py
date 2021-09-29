from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

def metalabel(json_file="/mnt/disk/SODA10M/SSLAD-2D/labeled/annotations/instance_train.json"):
    coco = COCO(json_file)

    for index in range(1, 5000):
        img = coco.loadImgs(index)[0]
        print(img['location'])


if __name__ == "__main__":
    metalabel()

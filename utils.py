import math
import torch
import numpy as np
from torchviz import make_dot
from torchvision.models import AlexNet
from PIL import Image
import matplotlib.pyplot as plt


def get_featuremap(model):
    model_feature = model
    model_feature.model = torch.nn.Sequential(*list(model.children())[0][:-1])
    return model_feature  # the model for feature getting


def network_show():
    model=AlexNet()
    x=torch.rand(8,3,256,512)
    y=model(x)
    print(y)
    g=make_dot(y, params=dict(model.named_parameters()))
    g.render('model_show.pdf', view=False)


def feature_visualization(x, n=32):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')

        plt.savefig('name')
        plt.close()


if __name__ == "__main__":
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='./dhd-traffic/traffic-all.pt', autoshape=False)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./dhd-traffic/traffic-all.pt')
    x = '/mnt/disk/TJU-DHD/dhd_traffic/valset/images/1496713172731.jpg'
    feature = model(x)
    feature_output1 = model.model.featuremap.cpu()
    print(feature_output1.shape)
    feature_visualization(feature_output1)

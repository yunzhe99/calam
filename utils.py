import math
import joblib
import torch
import numpy as np
from torchviz import make_dot
from torchvision.models import AlexNet
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from featuremap_classify import performance_test


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

    
def plot_perfect_choosing(sample_list):
    baseline_result = []
    perfect_result = []
    deep_result = []
    for sample in sample_list:
        baseline_result.append(sample.performance[1])
        perfect_result.append(np.max(sample.performance[1:]))
        deep_result.append(sample.performance[0])
    print('base result', np.mean(baseline_result))
    print('perfect result', np.mean(perfect_result))
    print('deep result', np.mean(deep_result))
    ecdf_baseline = sm.distributions.ECDF(baseline_result)

    x_baseline = np.linspace(min(baseline_result), max(baseline_result))

    y_baseline = ecdf_baseline(x_baseline)

    plt.step(x_baseline, y_baseline, label='shallow model baseline')

    ecdf_perfect = sm.distributions.ECDF(perfect_result)

    x_perfect = np.linspace(min(perfect_result), max(perfect_result))

    y_perfect = ecdf_perfect(x_perfect)

    plt.step(x_perfect, y_perfect, label='perfect model choosing')

    ecdf_deep = sm.distributions.ECDF(deep_result)

    x_deep = np.linspace(min(deep_result), max(deep_result))

    y_deep = ecdf_deep(x_deep)

    plt.step(x_deep, y_deep, label='deep model baseline')

    plt.legend()

    plt.savefig('perfect_choose.pdf')


def performance_heatmap(sample_list):
    performance_list = []
    for sample in sample_list[:30]:
        performance_list.append(sample.performance)
    sns.heatmap(performance_list, cmap="Greens")
    plt.savefig('heatmap.pdf')


def plot_confidence_heatmap(meta_model='best_classify_5-3.pt'):
    results = performance_test(meta_model=meta_model)
    sns.heatmap(results[:30], cmap="Greens")
    plt.savefig('confidence.pdf')


def plot_difference(difference_data='differences.tmp'):
    differences = joblib.load(difference_data)
    ecdf = sm.distributions.ECDF(differences)

    x_deep = np.linspace(min(differences), max(differences))

    y_deep = ecdf(x_deep)

    plt.step(x_deep, y_deep)
    plt.savefig('difference.pdf')


def sample_split(sample_dir='sample_list_dhd_traffic_all_with_feature_map.s', split_radio=0.5):
    sample_list = joblib.load(sample_dir)
    length = len(sample_list)
    split_index = int(length * split_radio)
    sample_part1 = sample_list[:split_index]
    sample_part2 = sample_list[split_index:]
    joblib.dump(sample_part1, 'sample_list_dhd_traffic_all_with_feature_map_part1.s')
    joblib.dump(sample_part2, 'sample_list_dhd_traffic_all_with_feature_map_part2.s')
    return True


if __name__ == "__main__":
    plot_difference()

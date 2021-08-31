import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from my_utils.data_io import load_dataset
from my_utils.preprocess import resnet_18_encoder
from my_utils.tools import get_distance_matrix


def get_features(Config):
    dataset_dir = Config.dataset_dir

    imgs = load_dataset(dataset_dir)

    img_features = resnet_18_encoder(imgs)

    img_features = np.array(img_features)

    img_features = np.squeeze(img_features)

    print(img_features.shape)

    # np.save("img_features.npy", img_features)


def context_classification_by_kmeans(img_features):
    # print(img_features)

    n_class = int(len(img_features) / 20)

    print(n_class)

    y_pred = KMeans(n_clusters=n_class, random_state=2316).fit_predict(img_features)

    # print(y_pred)

    kmeans_array = save_kmeans_array(img_features, y_pred)

    # # 画图代码
    #
    # x = np.arange(len(y_pred))
    #
    # plt.scatter(x, y_pred, alpha=0.6, s=1)
    # plt.axvline(x=255, color='r', linestyle='-')
    # plt.axvline(x=398, color='r', linestyle='-')
    # plt.axvline(x=542, color='r', linestyle='-')
    # plt.axvline(x=629, color='r', linestyle='-')
    # plt.axvline(x=909, color='r', linestyle='-')
    # plt.axvline(x=1072, color='r', linestyle='-')
    # plt.axvline(x=1194, color='r', linestyle='-')
    # plt.axvline(x=1481, color='r', linestyle='-')
    # plt.axvline(x=1582, color='r', linestyle='-')
    # plt.axvline(x=1675, color='r', linestyle='-')
    # plt.show()

    # # 保存结果
    #
    # dataframe = pd.DataFrame({'y_pred': y_pred})
    #
    # dataframe.to_csv("y_pred.csv", index=False, sep=',')

    return kmeans_array


def save_kmeans_array(img_features, cluster_result):
    array_len = len(np.unique(cluster_result))  # 数组的长度为类数

    # 初始化一个数组，其每个元素都是一个kmeans的聚类结果
    kmeans_array = [[] for _ in range(array_len)]

    for img_index in range(len(img_features)):
        kmeans_array[cluster_result[img_index]].append(img_features[img_index])

    return kmeans_array


def context_cluster_by_dbscan(kmeans_array):

    distance_matrix = get_distance_matrix(kmeans_array)

    sns.heatmap(data=distance_matrix, vmin=10, vmax=20, cmap='Blues')

    plt.show()

    clustering = DBSCAN(eps=12, min_samples=3, metric='precomputed').fit(distance_matrix)

    print(len(clustering.labels_))
    print(clustering.labels_)


def context_cluster_by_hierarchy_cluster(kmeans_array):

    distance_matrix = get_distance_matrix(kmeans_array)

    model = AgglomerativeClustering(affinity='precomputed',
                                    distance_threshold=0,
                                    n_clusters=None,
                                    linkage='average')

    model = model.fit(distance_matrix)

    plot_dendrogram(model, truncate_mode='level', p=10)
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def classifier():
    pass

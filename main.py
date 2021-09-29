import os
import joblib
import yaml
import random
import shutil
import heapq

import torch
import collections

import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from podm.podm import get_pascal_voc_metrics, BoundingBox

from config import Config
from my_utils.data_io import coco_c, Feature_set
from my_utils.preprocess import feature_getting
from task_model import model_list_train, Net, train_from_feature, train_from_image
from sample_node import sample_node, sample_feature
from test_batch_loader import test_batch_loader, load_label
from sample_node import performance_evaluation_map

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sample_list_gen(task_folder_list, task_label_list):
    # sample generation

    sample_index = 0
    sample_list = []

    # 随机抽取代表集
    for task_folder_index in range(len(task_folder_list)):
        try:
            f = open(task_folder_list[task_folder_index])
            file_list = f.readlines()
            random.shuffle(file_list)
            f.close()

            file_index = 0

            # file_list = np.loadtxt(dataset_list[task_folder_index])
            # length = Config.sample_len

            length = len(file_list)

            # for file_obj_index in tqdm(range(length)):  # 构建特征集
            for file_obj_index in range(len(file_list)):
                if file_index < length:
                    try:
                        file_obj = file_list[file_obj_index].strip("\n")
                        file_path = file_obj
                        file_dir, file_name = os.path.split(file_path)

                        label_path = os.path.join(os.path.dirname(file_dir), 'labels', file_name[:-4] + '.txt')

                        if os.path.exists(label_path) and os.path.getsize(label_path):  # 判断标签存在且非空
                            # print(os.path.getsize(label_path))
                            sample = sample_node(sample_index, file_path, label_path)
                            # sample.feature_getting()  # 抽取特征
                            sample.sample_evalution()  # 性能验证
                            # print('check result: ', sample.sample_check())
                            sample_list.append(sample)
                            sample_index = sample_index + 1
                            file_index = file_index + 1
                            print(file_index)

                    except Exception as e:
                        print(e)
                        continue

        except Exception as e:
            print(e)
            file_list = os.listdir(task_folder_list[task_folder_index])
            random.shuffle(file_list)
            length = Config.sample_len

            wrong_num = 0

            for file_obj_index in tqdm(range(length)):  # 随机选择样本构建特征集
                try:

                    file_obj = file_list[file_obj_index]
                    file_path = os.path.join(task_folder_list[task_folder_index], file_obj)
                    label_path = os.path.join(task_label_list[task_folder_index], file_obj[:-4] + '.txt')

                    # print(file_path)

                    sample = sample_node(sample_index, file_path, label_path)
                    sample.feature_getting()  # 抽取特征
                    sample.sample_evalution()  # 性能验证
                    # print('check result: ', sample.sample_check())
                    sample_list.append(sample)
                    sample_index = sample_index + 1

                except:
                    wrong_num = wrong_num + 1
                    print(wrong_num)
                    continue

        # break

    # print(sample_list[300].feature)
    joblib.dump(sample_list, 'sample_list_dhd_traffic_all.s')


def train_txt_gen():
    #  找到当前效果不好的样本
    sample_list = joblib.load('sample_list_dhd_traffic_all.s')  # 'sample_list_all_14.s'
    performance_list = []
    performance_all = []
    f = open('dhd_train_7.txt', 'w')
    for sample in sample_list:
        performance = np.max(sample.performance)
        # print(sample.performance)
        performance_list.append(performance)
        performance_all.append(sample.performance)
        if performance < 0.8:
            # print(sample.img_dir)
            f.write(sample.img_dir + '\n')

    f.close()

    print(sample_list[1020].img_dir)

    # performance_all = np.array(performance_all)

    # #字典中的key值即为csv中列名
    # dataframe = pd.DataFrame({'0':performance_all[:,0], '1':performance_all[:,1], '2':performance_all[:,2], '3':performance_all[:,3], '4':performance_all[:,4], '5':performance_all[:,5], '6':performance_all[:,6]})

    # #将DataFrame存储为csv,index表示是否显示行名，default=True
    # dataframe.to_csv("test.csv",index=False,sep=',')


    joblib.dump(performance_list, 'performance_list_6.l')
    print(len(performance_list))


def chooser_gen():
    # # 代表集加载，这一步不必须，实验期间使用同一个代表集，因为生成代表集需要很长时间
    #
    sample_list = joblib.load('sample_list_dhd_traffic_all.s')  # 'sample_list_all_14.s'
    length = len(sample_list)

    print(length)

    # 在代表集中区分训练集和验证集

    train_split_index = int(length * Config.train_val_ratio)

    print(train_split_index)

    train_sample_list = sample_list[:train_split_index]
    val_sample_list = sample_list[train_split_index:]

    print(len(val_sample_list))

    chooser_list = []

    # 生成模型选择器：神经网络方案
    for task_index in range(len(Config.weight_list)):  # len(Config.dataset_train_list)

        train_sample = sample_feature(train_sample_list, task_index, need_image=True, classify=True)
        train_sample_loader = DataLoader(train_sample, batch_size=Config.batch_size, shuffle=True)

        val_sample = sample_feature(val_sample_list, task_index, need_image=True, classify=True)
        val_sample_loader = DataLoader(val_sample, batch_size=Config.batch_size, shuffle=True)

        print(val_sample_loader.__len__())

        # chooser = train_from_feature(train_sample_loader, val_sample_loader)
        chooser = train_from_image(train_sample_loader, val_sample_loader)
        print('\n')

        chooser_list.append(chooser)

    # # 生成模型选择器，adaboost方案
    #
    # features_train = []
    #
    # for train_sample in train_sample_list:
    #     features_train.append(train_sample.feature)
    #
    # features_train = np.array(features_train)
    #
    # features_train = np.squeeze(features_train)
    #
    # # pca = PCA(n_components=32, svd_solver='full')
    # #
    # # pca.fit(features_train)
    # # features_train = pca.transform(features_train)
    #
    # # print(pca.explained_variance_ratio_)
    #
    # features_val = []
    #
    # for val_sample in val_sample_list:
    #     features_val.append(val_sample.feature)
    #
    # features_val = np.array(features_val)
    #
    # features_val = np.squeeze(features_val)
    #
    # # features_val = pca.transform(features_val)
    #
    # label_train_list = []
    #
    # for task_index in range(len(Config.weight_list)):  # len(Config.dataset_train_list)
    #     label_train = []
    #     for train_sample in train_sample_list:
    #         label_train.append(train_sample.performance[task_index])
    #     print(label_train)
    #     print("\n")
    #     label_train_list.append(label_train)
    #     label_val = []
    #     for val_sample in val_sample_list:
    #         label_val.append(val_sample.performance[task_index])
    #     regr = AdaBoostRegressor(n_estimators=50)
    #     regr.fit(features_train, label_train)
    #     # predict = regr.predict(features_val)
    #     # # score = regr.score(features_val, label_val)
    #     # mse = mean_squared_error(predict, label_val)
    #     chooser_list.append(regr)
    #     # print(mse)
    #     print(task_index)

    joblib.dump(chooser_list, 'chooser_list_all_image.m')
    # features_train = np.array(features_train, dtype=np.float32)
    # label_train_list = np.array(label_train_list, dtype=np.float32)
    # joblib.dump([features_train, label_train_list], 'feature_train.d')
    # joblib.dump(pca, 'pca.m')


def result_evaluation(results, label, image_name):
    results = results.pandas().xyxyn
    inference_boxs = []

    for result_index in range(len(results)):
        result = np.array(results[result_index])

        (filepath, tempfilename) = os.path.split(image_name[result_index])
        (filename, extension) = os.path.splitext(tempfilename)

        for obj in result:
            bb = BoundingBox(filename, obj[5], obj[0], obj[1], obj[2], obj[3], obj[4])
            inference_boxs.append(bb)

    label = list(label)
    gt_box = load_label(label)

    ret = get_pascal_voc_metrics(gt_box, inference_boxs)

    # ap_total = 0
    # class_num = 0
    #
    # for class_index in ret:
    #     if not np.isnan(ret[class_index].ap):  # 如果预测出了数据集以外的类，那么就会出现nan
    #         ap_total = ap_total + ret[class_index].ap
    #         class_num = class_num + 1
    #
    # map = ap_total / class_num

    if len(inference_boxs) == 0 and len(gt_box) == 0:
        ap = 1
    else:
        ap = ret[0].ap

    return ap


def box2ap(inference_boxs, gt_box):
    ret = get_pascal_voc_metrics(gt_box, inference_boxs, iou_threshold=0.5)

    # ap_total = 0
    # class_num = 0
    #
    # for class_index in ret:
    #     if not np.isnan(ret[class_index].ap):  # 如果预测出了数据集以外的类，那么就会出现nan
    #         ap_total = ap_total + ret[class_index].ap
    #         class_num = class_num + 1
    #
    # map = ap_total / class_num

    if len(inference_boxs) == 0 and len(gt_box) == 0:
        ap = 1
        p = 1
        r = 1
    else:
        ap = ret[0].ap

        tp = ret[0].tp
        fp = ret[0].fp
        p = tp / (fp + tp)

        num_groundtruth = ret[0].num_groundtruth

        r = tp / num_groundtruth

    return ap, p, r


def box2map(inference_boxs, gt_box):
    ret = get_pascal_voc_metrics(gt_box, inference_boxs, iou_threshold=0.5)

    # ap_total = 0
    # class_num = 0
    #
    # for class_index in ret:
    #     if not np.isnan(ret[class_index].ap):  # 如果预测出了数据集以外的类，那么就会出现nan
    #         ap_total = ap_total + ret[class_index].ap
    #         class_num = class_num + 1
    #
    # map = ap_total / class_num

    ap_total = 0
    class_num = 0

    for class_index in ret:
        if not np.isnan(ret[class_index].ap):  # 如果预测出了数据集以外的类，那么就会出现nan
            ap_total = ap_total + ret[class_index].ap
            class_num = class_num + 1

    m_ap = ap_total / class_num

    return m_ap


def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def result_test(force_reload=False):
    # # print(len(train_sample_list))
    # print(len(val_sample_list))
    #

    chooser_list = joblib.load('chooser_list_all_image.m')
    # pca = joblib.load('pca.m')
    #
    # print(chooser_list)

    # attribute_list = joblib.load('attribute.m')

    # 数据集地址
    val_image_list = Config.dataset_val_list
    val_label_list = Config.labelset_val_list

    test_batch = test_batch_loader(val_image_list, val_label_list, force_loading=force_reload)
    random_test_loader = test_batch.random_test()
    kmeans_test_loader = test_batch.kmeans_test()
    in_order_test_loader = test_batch.in_order_test()

    # model_base = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model_base = torch.hub.load('ultralytics/yolov5', 'custom', path='./dhd-traffic/traffic-all.pt',
                                verbose=False)  # './all_v5s_v.pt' './single_v5s.pt'
    model_base_big = torch.hub.load('ultralytics/yolov5', 'yolov5x')

    # model_base_big = torch.hub.load('ultralytics/yolov5', 'custom', path='./single_v5x.pt',
    #                             verbose=False)  # './all_v5s_v.pt' './single_v5s.pt'

    models = []

    for model_index in range(len(chooser_list)):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=Config.weight_list[model_index], verbose=False)
        models.append(model)

    # models = model_list

    inference_bboxs = []
    inference_mss = []
    inference_cbs = []
    inference_hp = []
    label_bboxs = []

    for data in tqdm(kmeans_test_loader):  # random_test_loader, kmeans_test_loader

        # print("开始处理批数据")
        sample, label = data
        image_dir, image, feature = sample
        image_name = list(image_dir)

        feature = np.squeeze(feature)
        # feature = pca.transform(feature)

        feature = feature.numpy()

        # image = np.squeeze(image)

        # print(image)

        # print(type(feature))

        if torch.cuda.is_available() is True:
            image = image.cuda()
        # feature = feature.float()

        max_score = 0
        max_index = 0
        min_dist = 10000000000000000

        for model_index in range(len(chooser_list)):

            # 神经网络方案
            model_chooser = chooser_list[model_index]
            # 加载模型
            model = Net()
            model.load_state_dict(model_chooser)
            if torch.cuda.is_available() is True:
                model = model.cuda()
            model.eval()

            out = model(image)

            print(out)

            out = out.cpu().detach().numpy()

            score = np.mean(out)

            if score >= max_score:
                max_score = score
                max_index = model_index

            print(score)

            # # adaboost方案
            #
            # model_chooser = chooser_list[model_index]
            #
            # feature = feature.reshape(-1, 512)
            #
            # maps = model_chooser.predict(feature)
            #
            # # print(maps)
            #
            # score = np.mean(maps)
            #
            # if score >= max_score:
            #     max_score = score
            #     max_index = model_index
            #
            # print(score)

            # # distance 方案
            #
            # attribute = attribute_list[model_index]
            #
            # attribute = np.array(attribute)
            #
            # # dist = np.linalg.norm(np.mean(attribute, axis=0) - np.mean(feature, axis=0))
            #
            # dist = mmd_linear(attribute, feature)
            #
            # if dist < min_dist:
            #     min_dist = dist
            #     max_index = model_index
            #
            # print(dist)

        # max_index = 0

        print("model chosen", max_index)

        image = list(image_dir)

        # CalaM的验证

        model_chosen = models[max_index]
        results = model_chosen(image)

        results = results.pandas().xyxyn

        for result_index in range(len(results)):
            result = np.array(results[result_index])

            (filepath, tempfilename) = os.path.split(image_name[result_index])
            (filename, extension) = os.path.splitext(tempfilename)

            for obj in result:
                bb = BoundingBox(filename, obj[5], obj[0], obj[1], obj[2], obj[3], obj[4])
                inference_bboxs.append(bb)

        # mss

        results_base_o = model_base(image)

        results_base = results_base_o.pandas().xyxyn

        for result_index in range(len(results_base)):
            result = np.array(results_base[result_index])

            (filepath, tempfilename) = os.path.split(image_name[result_index])
            (filename, extension) = os.path.splitext(tempfilename)

            for obj in result:
                bb = BoundingBox(filename, obj[5], obj[0], obj[1], obj[2], obj[3], obj[4])
                inference_mss.append(bb)

        # cbs

        results_base_big_o = model_base_big(image)

        results_base_big = results_base_big_o.pandas().xyxyn

        for result_index in range(len(results_base_big)):
            result = np.array(results_base_big[result_index])

            (filepath, tempfilename) = os.path.split(image_name[result_index])
            (filename, extension) = os.path.splitext(tempfilename)

            for obj in result:
                bb = BoundingBox(filename, obj[5], obj[0], obj[1], obj[2], obj[3], obj[4])
                inference_cbs.append(bb)

        # hp
        confidence = np.array(results_base_o.pandas().xyxy[0])[:, 4]
        if len(confidence) > 0:
            c_total = np.mean(confidence)
        else:
            c_total = 0

        # print(c_total)

        if c_total < 0.5:
            results_hp = results_base_big
        else:
            results_hp = results_base

        for result_index in range(len(results_hp)):
            result = np.array(results_hp[result_index])

            (filepath, tempfilename) = os.path.split(image_name[result_index])
            (filename, extension) = os.path.splitext(tempfilename)

            for obj in result:
                bb = BoundingBox(filename, obj[5], obj[0], obj[1], obj[2], obj[3], obj[4])
                inference_hp.append(bb)

        label_list = list(label)
        gt_box = load_label(label_list)
        label_bboxs = label_bboxs + gt_box

        # ap_calam, p_calam, r_calam = box2ap(inference_bboxs, label_bboxs)

        # print(ap_calam, p_calam, r_calam)

        #
        # model_0 = torch.hub.load('ultralytics/yolov5', 'custom', path='base_model/task0.pt')
        # results_0 = model_0(image)
        #
        # model_1 = torch.hub.load('ultralytics/yolov5', 'custom', path='base_model/task1.pt')
        # results_1 = model_1(image)
        #
        # model_2 = torch.hub.load('ultralytics/yolov5', 'custom', path='base_model/task2.pt')
        # results_2 = model_2(image)
        #
        # model_3 = torch.hub.load('ultralytics/yolov5', 'custom', path='base_model/task3.pt')
        # results_3 = model_3(image)
        #
        # model_4 = torch.hub.load('ultralytics/yolov5', 'custom', path='base_model/task4.pt')
        # results_4 = model_4(image)
        #
        # model_crowd = torch.hub.load('ultralytics/yolov5', 'custom', path='base_model/crowdhuman.pt')
        # results_crowd = model_crowd(image)

        # map_result = result_evaluation(results, label, image_name)
        # map_baseline = result_evaluation(results_base, label, image_name)
        # map_baseline_big = result_evaluation(results_base_big, label, image_name)
        # map_result_0 = result_evaluation(results_0, label, image_name)
        # map_result_1 = result_evaluation(results_1, label, image_name)
        # map_result_2 = result_evaluation(results_2, label, image_name)
        # map_result_3 = result_evaluation(results_3, label, image_name)
        # map_result_4 = result_evaluation(results_4, label, image_name)
        # map_result_crowd = result_evaluation(results_crowd, label, image_name)

        model_chosen = None

        # print(map_result)
        # print(map_baseline)
        # print(map_baseline_big)

        # print(map_result_0)
        # print(map_result_1)
        # print(map_result_2)
        # print(map_result_3)
        # print(map_result_4)
        # print(map_result_crowd)

        # model_0 = torch.hub.load('ultralytics/yolov5', 'custom', path=Config.weight_list[0], verbose=False)
        # results_0 = model_0(image)
        #
        # model_1 = torch.hub.load('ultralytics/yolov5', 'custom', path=Config.weight_list[1], verbose=False)
        # results_1 = model_1(image)
        #
        # model_2 = torch.hub.load('ultralytics/yolov5', 'custom', path=Config.weight_list[2], verbose=False)
        # results_2 = model_2(image)
        #
        # ap_0 = result_evaluation(results_0, label, image_name)
        # ap_1 = result_evaluation(results_1, label, image_name)
        # ap_2 = result_evaluation(results_2, label, image_name)
        #
        # print(ap_0)
        # print(ap_1)
        # print(ap_2)

    boxs_list = [inference_bboxs, inference_mss, inference_cbs, inference_hp, label_bboxs]

    joblib.dump(boxs_list, 'boxs_list_downtown.b')

    # ap_calam, p_calam, r_calam = box2ap(inference_bboxs, label_bboxs)
    # ap_mss, p_mss, r_mss = box2ap(inference_mss, label_bboxs)
    # ap_cbs, p_cbs, r_cbs = box2ap(inference_cbs, label_bboxs)
    #
    # print(ap_calam, p_calam, r_calam)
    # print(ap_mss, p_mss, r_mss)
    # print(ap_cbs, p_cbs, r_cbs)


def task_gen():
    # # 检查目录是否存在
    # if not os.path.exists(sample_set_images):
    #     os.makedirs(sample_set_images)
    #
    # if not os.path.exists(sample_set_labels):
    #     os.makedirs(sample_set_labels)

    all_label_dir = '/mnt/disk1/yunzhe/yolov4_crowdhuman/data/yolo/labels/train/'
    all_image_dir = '/mnt/disk1/yunzhe/yolov4_crowdhuman/data/yolo/images/train/'

    file_list = os.listdir(all_image_dir)

    random.shuffle(file_list)

    length = 8000  # 每个任务20000样本

    for file_obj_index in tqdm(range(length)):  # 随机选择样本构建特征集
        file_obj = file_list[file_obj_index]
        file_path = os.path.join(all_image_dir, file_obj)
        label_path = os.path.join(all_label_dir, file_obj[:-4] + '.txt')

        new_name_image = file_obj
        new_name_label = file_obj[:-4] + '.txt'

        newfile_path = os.path.join('/mnt/disk1/yunzhe/task_data/task9/images/', str(new_name_image))
        newlabel_path = os.path.join('/mnt/disk1/yunzhe/task_data/task9/labels/', str(new_name_label))

        try:
            shutil.copyfile(file_path, newfile_path)
            shutil.copyfile(label_path, newlabel_path)
        except IOError:
            print("文件不存在", label_path)


def metric_calculation():
    [inference_bboxs, inference_mss, inference_cbs, inference_hp, label_bboxs] = joblib.load('boxs_list_downtown.b')

    # ap_calam, p_calam, r_calam = box2ap(inference_bboxs, label_bboxs)
    # ap_mss, p_mss, r_mss = box2ap(inference_mss, label_bboxs)
    # ap_cbs, p_cbs, r_cbs = box2ap(inference_cbs, label_bboxs)
    # ap_hp, p_hp, r_hp = box2ap(inference_hp, label_bboxs)

    ap_calam = box2map(inference_bboxs, label_bboxs)
    ap_mss = box2map(inference_mss, label_bboxs)
    ap_cbs = box2map(inference_cbs, label_bboxs)
    ap_hp = box2map(inference_hp, label_bboxs)
    #
    print(ap_calam)
    print(ap_mss)
    print(ap_cbs)
    print(ap_hp)


def task_global_attribute():
    sample_list = joblib.load('sample_list_all_14.s')

    attribute_list = []
    attribute = []

    for sample in sample_list:
        if sample.index % (Config.sample_len / 2) == 0:
            attribute = [sample.feature]
        elif sample.index % (Config.sample_len / 2) == (Config.sample_len / 2) - 1:
            attribute.append(sample.feature)
            attribute_list.append(attribute)
        else:
            attribute.append(sample.feature)

    print(len(attribute_list))

    joblib.dump(attribute_list, 'attribute.m')


def ap_compare():
    sample_list = joblib.load('sample_list_dhd_traffic.s')
    model_chooser = joblib.load('chooser_list_all_v.m')

    attribute_list = joblib.load('attribute.m')

    top_num = np.zeros(10)

    p_s_list = []
    p_x_list = []
    p_best_list = []
    for sample in sample_list:
        p_s = sample.performance[5]
        p_x = sample.performance[6]
        p_best = np.max(sample.performance[0:5])

        p_s_list.append(p_s)
        p_x_list.append(p_x)
        p_best_list.append(p_best)

        confidence_list = []

        for chooser in model_chooser:
            confidence = chooser.predict(sample.feature)[0]
            confidence_list.append(confidence)

        chosen_index = np.where(confidence_list == np.max(confidence_list))[0][0]

        # for attribute in attribute_list:
        #     feature = np.array(sample.feature)
        #     attribute = np.array(attribute)
        #     dist = mmd_linear(attribute, feature)[0][0]
        #     confidence_list.append(dist)
        #
        # chosen_index = np.where(confidence_list == np.min(confidence_list))[0][0]

        if np.max(sample.performance) > 0:
            top_10 = heapq.nlargest(10, range(len(sample.performance)), sample.performance.__getitem__)
            # print(top_10)

            # print(top_5)

            for i in range(len(top_10)):
                if chosen_index == top_10[i]:
                    num = top_num[i]
                    top_num[i] = num + 1

        # print(chosen_index[0][0])
    print(top_num)
    print(len(sample_list))
    # joblib.dump([p_s_list, p_x_list, p_best_list], 'p_dhd.p')


def unseen_data_detection():
    sample_list = joblib.load('sample_list_dhd_traffic.s')
    model_chooser = joblib.load('chooser_list_all_v.m')

    attribute_list = joblib.load('attribute.m')

    label = np.zeros(len(sample_list))
    calam_detect = np.zeros(len(sample_list))
    gac_detect = np.zeros(len(sample_list))
    confidence = np.zeros(len(sample_list))

    model_base = torch.hub.load('ultralytics/yolov5', 'custom', path='./single_v5s.pt',
                                verbose=False)  # './all_v5s_v.pt' './single_v5s.pt'

    for sample_index in range(len(sample_list)):
        sample = sample_list[sample_index]
        performance = np.array(sample.performance)
        if np.max(performance) < 0.7:
            label[sample_index] = 1

        calam_list = []

        for chooser in model_chooser:
            calam = chooser.predict(sample.feature)[0]
            calam_list.append(calam)

        calam_list = np.array(calam_list)
        calam_detect[sample_index] = 1 - np.max(calam_list)

        dist_list = []
        for attribute in attribute_list:
            feature = np.array(sample.feature)
            attribute = np.array(attribute)
            dist = mmd_linear(attribute, feature)[0][0]
            dist_list.append(dist)

        dist_list = np.array(dist_list)
        gac_detect[sample_index] = np.min(dist_list)

        result = model_base(sample.img_dir)

        conf = np.array(result.pandas().xyxy[0])[:, 4]
        if len(conf) > 0:
            conf = np.mean(conf)
        else:
            conf = 0

        confidence[sample_index] = conf

    # for calam_thed in range(0, 10, 1):
    #     calam_thed = calam_thed / 10
    #     for sample_index in range(len(sample_list)):
    #         sample = sample_list[sample_index]
    #         calam_list = []
    #
    #         for chooser in model_chooser:
    #             calam = chooser.predict(sample.feature)[0]
    #             calam_list.append(calam)
    #
    #         calam_list = np.array(calam_list)
    #         if np.max(calam_list) < calam_thed:
    #             calam_detect[sample_index] = 1
    #
    #     recall = metrics.recall_score(label, calam_detect)
    #     if recall == 0:
    #         precision = 1
    #     else:
    #         precision = metrics.precision_score(label, calam_detect)
    #
    #     print(recall)
    #     print(precision)

    # print(calam_detect)
    # print(gac_detect)
    # print(confidence)

    joblib.dump([label, calam_detect, gac_detect, confidence], 'unseen_data.d')


def performance_update(sample_list, model_add):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_add, verbose=False)
    for sample in sample_list:
        performance = performance_evaluation_map(sample.img_dir, sample.label_dir, model)
        sample.performance = np.append(sample.performance, performance)
        sample.model_set = np.append(sample.model_set, model_add)


if __name__ == "__main__":

    # task_gen()

    # sample_list_gen(Config.dataset_train_list, Config.labelset_train_list)

    train_txt_gen()

    # model_add = '/home/lion/yunzhe/calam/dhd-traffic/round6.pt'
    # performance_update('sample_list_dhd_traffic_all.s', model_add)

    # ap_compare()
    
    # unseen_data_detection()

    # chooser_gen()

    # task_global_attribute()

    # result_test(force_reload=False)
    # #
    # metric_calculation()

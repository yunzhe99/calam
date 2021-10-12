import os
import re
import joblib
from matplotlib import pyplot as plt
import torch
import torchvision
import wandb
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from CalaNet import CalaNet, CalaNet1L
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# wandb.init(project="featuremap_classification")

experiment_radio = 1   # the ratio for the sample for experiment
Learning_rate = 1e-4  # learning rate for train
epoch = 20
print_iter = 10


# model_list = []

# for weight_index in range(1, len(Config.weight_list)):
#     weight = Config.weight_list[weight_index]
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, verbose=False)
#     model_list.append(model)


class sample_feature_all(Dataset):
    def __init__(self, sample_list, feature_extraction=None, classify=False, feature_exist=False, model_index="all"):
        self.sample_list = sample_list
        self.feature_extraction = feature_extraction
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        self.classify = classify
        self.feature_exist = feature_exist
        self.model_index = model_index

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        label = self.sample_list[idx].performance

        if self.classify is True:

            for label_index in range(len(label)):
                if label[label_index] > 0.8:
                    label[label_index] = float(1)
                else:
                    label[label_index] = float(0)

        if self.feature_exist is not True:
            if self.feature_extraction is None:
                img = Image.open(self.sample_list[idx].img_dir)
                img = self.transform(img)
                feature = img
            else:
                feature = self.feature_extraction(self.sample_list[idx].img_dir)
                feature = np.squeeze(feature)
        else:
            feature = self.sample_list[idx].feature
            feature = np.squeeze(feature)

        if self.model_index != "all":
            feature = feature[(512 * self.model_index):(512 * (self.model_index + 1))]
            label = np.array([label[self.model_index]])
        
        return feature, label


def feature_extractor(img_dir):
    feature_new = []
    for model in model_list:
        x = model(img_dir)
        feature_output = model.model.featuremap.cpu().numpy()
        for feature in feature_output[0]:  # for one image, there is only one feature map
            feature_new.append(feature)
    feature_new = np.array([feature_new])  # reshape to the original shape: batch * channels * height * width
    return feature_new


def data_prepare(sample_list):
    length = len(sample_list)

    print(length)
    experiment_index = int(length * experiment_radio)
    print('experiment index', experiment_index)
    sample_list = sample_list[:experiment_index]

    # 在代表集中区分训练集和验证集

    train_split_index = int(experiment_index * Config.train_val_ratio)

    print(train_split_index)

    train_sample_list = sample_list[:train_split_index]
    val_sample_list = sample_list[train_split_index:]

    print(len(val_sample_list))

    train_sample = sample_feature_all(train_sample_list, feature_extraction=feature_extractor, classify=True, feature_exist=True)
    train_sample_loader = DataLoader(train_sample, batch_size=Config.batch_size, shuffle=True)

    val_sample = sample_feature_all(val_sample_list, feature_extraction=feature_extractor, classify=True, feature_exist=True)
    val_sample_loader = DataLoader(val_sample, batch_size=Config.batch_size, shuffle=True)

    return train_sample_loader, val_sample_loader, len(val_sample_list)


# for weight_index in range(len(Config.weight_list)):
#     weight = Config.weight_list[weight_index]
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, verbose=False)
#     model_list.append(model)


def train_each(sample_list):

    length = len(sample_list)

    print(length)
    experiment_index = int(length * experiment_radio)
    print('experiment index', experiment_index)
    sample_list = sample_list[:experiment_index]

    # 在代表集中区分训练集和验证集

    train_split_index = int(experiment_index * Config.train_val_ratio)

    print(train_split_index)

    train_sample_list = sample_list[:train_split_index]
    val_sample_list = sample_list[train_split_index:]

    val_len = len(val_sample_list)
    print(len(val_sample_list))

    for model_index in range(len(sample_list[0].performance)):
        train_sample = sample_feature_all(train_sample_list, feature_extraction=feature_extractor, classify=False, feature_exist=True, model_index=model_index)
        train_sample_loader = DataLoader(train_sample, batch_size=Config.batch_size, shuffle=True)

        val_sample = sample_feature_all(val_sample_list, feature_extraction=feature_extractor, classify=False, feature_exist=True, model_index=model_index)
        val_sample_loader = DataLoader(val_sample, batch_size=Config.batch_size, shuffle=True)

        model = CalaNet1L()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = Learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
        # optimizer = torch.optim.Adagrad()
        loss_count = []
        loss_thr = None
        performance_max = 0
        acc_list = []
        for iter in range(epoch):
            if loss_thr != None:
                if loss_thr <= 0.01:
                    print('training finished.')
                    break

            print('Epoch ', iter, ': ')
            cnt = 0
            loss_sum = 0
            for i, (data_batch, labels_batch) in enumerate(train_sample_loader):
                data_batch = data_batch.float()
                y_pred = model(data_batch)
                # m = nn.Sigmoid()
                # y_pred = m(y_pred)
                labels_batch = labels_batch.float()
                loss = loss_fn(y_pred, labels_batch)
                if i % print_iter == 0:
                    loss_count.append(loss)
                    print("Iteration: ", i, ", Loss: ", loss.item())
                loss_sum += loss.item()
                # model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt = i
            loss_thr = loss_sum/(cnt+1)

            torch.save(model, 'classify' + str(model_index) + '.pt')
            for (test_x, test_y) in val_sample_loader:
                test_x = test_x.float()
                pred = model(test_x)
                loss = loss_fn(pred, test_y)

                print("Validate loss: ", loss.item())
                break

            acc = model_test_each(model, val_sample_loader, val_len)
            acc_list.append(acc)
            # if performance_on_test > performance_max:
            #     torch.save(model, 'best_classify' + str(model_index) + '.pt')
            # wandb.log({'train_loss': loss_thr, 'val_loss': loss.item(), 'acc': acc})

        plt.plot(acc_list)
        plt.savefig('train_mae.pdf')
        
        for (train_x, train_y) in train_sample_loader:
            train_x = train_x.float()
            pred = model(train_x)
            # print(pred.numpy())
            loss = loss_fn(pred, train_y)

            print("Train loss: ", loss.item())
            break
        break


def model_test_each(model, val_sample_loader, val_len):
    loss_sum = 0
    for (test_x, test_y) in val_sample_loader:
            test_x = test_x.float()
            pred = model(test_x)
            # m = nn.Sigmoid()
            # result = m(pred)
            result = pred
            result = result.detach()
            # result = result.numpy()
            # test_y = test_y.numpy()
            #  if the most possible one can be choosed, we think it is correct.
            # print(result.numpy())
            loss_fun = nn.L1Loss()
            loss = loss_fun(test_y, result)
            loss_sum += loss.item()
    print('val mae', loss_sum / len(val_sample_loader))
    return loss_sum / len(val_sample_loader)


def model_train(train_sample_loader, val_sample_loader, val_len=1000, model_name='featuremap_classify.pt'):

    model = CalaNet()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    learning_rate = Learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    # optimizer = torch.optim.Adagrad()
    loss_count = []
    loss_thr = None
    performance_max = 0
    for iter in range(epoch):
        if loss_thr != None:
            if loss_thr <= 0.01:
                print('training finished.')
                break

        print('Epoch ', iter, ': ')
        cnt = 0
        loss_sum = 0
        for i, (data_batch, labels_batch) in enumerate(train_sample_loader):
            data_batch = data_batch.float()
            y_pred = model(data_batch)
            loss = loss_fn(y_pred, labels_batch)
            if i % print_iter == 0:
                loss_count.append(loss)
                print("Iteration: ", i, ", Loss: ", loss.item())
            loss_sum += loss.item()
            # model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt = i
        loss_thr = loss_sum/(cnt+1)

        torch.save(model, model_name)
        for (test_x, test_y) in val_sample_loader:
            test_x = test_x.float()
            pred = model(test_x)
            loss = loss_fn(pred, test_y)

            print("Validate loss: ", loss.item())
            break

        acc = model_test(val_sample_loader, val_len)
        performance_on_test = performance_test()
        if performance_on_test > performance_max:
            torch.save(model, 'best_'+model_name)
        wandb.log({'train_loss': loss_thr, 'val_loss': loss.item(), 'acc': acc, 'performance_test': performance_on_test})

    for (train_x, train_y) in train_sample_loader:
        train_x = train_x.float()
        pred = model(train_x)
        # print(pred.numpy())
        loss = loss_fn(pred, train_y)

        print("Train loss: ", loss.item())
        break


def model_test(val_sample_loader, val_len, model='./featuremap_classify.pt'):
    model = torch.load('./featuremap_classify.pt')
    correct_num = 0
    for (test_x, test_y) in val_sample_loader:
            test_x = test_x.float()
            pred = model(test_x)
            m = nn.Sigmoid()
            result = m(pred)
            result = result.detach()
            result = result.numpy()
            test_y = test_y.numpy()
            # print(np.round(result))
            # print(test_y)
            #  if the most possible one can be choosed, we think it is correct.
            choose_index_list = np.argmax(result, axis=1)
            # print(choose_index_list)
            for i in range(len(choose_index_list)):
                if (test_y[i][choose_index_list[i]] == 1) or (np.max(test_y[i]) < 1):
                    correct_num += 1
    print('top 1 test accuracy', correct_num / val_len)
    return correct_num / val_len


def performance_test(sample_list_test='test_sample_with_features.s', meta_model='./featuremap_classify.pt'):
    sample_list = joblib.load(sample_list_test)
    meta_model = torch.load(meta_model)
    shallow_performance_list = []
    deep_performance_list = []
    choose_list = []
    perfect_list = []
    results = []
    for sample in sample_list:
        feature = sample.feature
        feature = torch.from_numpy(feature)
        feature = feature.float()
        pred = meta_model(feature)
        m = nn.Sigmoid()
        result = m(pred)
        result = result.detach()
        result = result.numpy()
        # print(result[0])
        choose_index = np.argmax(result[0])
        # print(choose_index)
        results.append(result[0])
        choose_performance = sample.performance[choose_index+1]
        choose_list.append(choose_performance)
        shallow_performance_list.append(sample.performance[1])
        deep_performance_list.append(sample.performance[0])
        perfect_list.append(np.max(sample.performance[1:]))
    # print("Shallow:", np.mean(shallow_performance_list))
    # print("Deep:", np.mean(deep_performance_list))
    print("Choose:", np.mean(choose_list))
    # print("Perfect:", np.mean(perfect_list))
    return results


def performance_test_each(meta_model_list, sample_list_test='test_sample_with_features.s'):
    sample_list = joblib.load(sample_list_test)
    for meta_model in meta_model_list:
        meta_model = torch.load(meta_model)
    shallow_performance_list = []
    deep_performance_list = []
    choose_list = []
    perfect_list = []
    results = []
    differences = []
    hard_sample_num = 0
    for sample in sample_list:
        choose_index = -1  # if no one can be choosed, we will turn to the deep one.
        for meta_model_index in range(len(meta_model_list)):
            feature = np.array([sample.feature[0][(512 * meta_model_index):(512 * (meta_model_index + 1))]])
            feature = torch.from_numpy(feature)
            feature = feature.float()
            pred = meta_model(feature)
            result = pred
            result = result.detach()
            result = result.numpy()
            # print(result[0])
            results.append(result[0])
            if result[0] > 1:  # the therhold that a model can be choosed
                choose_index = meta_model_index
                break
        print(choose_index)
        if choose_index == -1:
            hard_sample_num +=1
        choose_performance = sample.performance[choose_index+1]
        choose_list.append(choose_performance)
        shallow_performance_list.append(sample.performance[1])
        deep_performance_list.append(sample.performance[0])
        perfect_list.append(np.max(sample.performance[1:]))
        difference = np.max(sample.performance[1:]) - choose_performance
        differences.append(difference)
        if difference > 0.4:
            print(sample.img_dir)
    print("Shallow:", np.mean(shallow_performance_list))
    print("Deep:", np.mean(deep_performance_list))
    print("Choose:", np.mean(choose_list))
    print("Perfect:", np.mean(perfect_list))
    print("Hard sample ratio", hard_sample_num / len(sample_list))
    joblib.dump(differences, 'differences.tmp')
    return 0


if __name__ == "__main__":
    # sample_list_train = joblib.load('sample_list_dhd_traffic_all_with_feature_map_part1.s')

    # for sample in tqdm(sample_list):
    #     sample.feature_update(feature_extractor)
    
    # joblib.dump(sample_list, 'test_sample_with_features.s')

    # for sample in sample_list_train:
    #     print(sample.feature)

    # train without features

    # resnet_model = torchvision.models.resnet18(pretrained=True)

    # feature = feature_extractor(sample_list[0].img_dir)

    # print(dir(resnet_model))

    # chooser = train_from_feature(train_sample_loader, val_sample_loader)

    # train with features existing

    # print("Begin data preparation")

    # train_sample_loader, val_sample_loader, val_len = data_prepare(sample_list_train)

    # print("Begin model training")

    # chooser = model_train(train_sample_loader, val_sample_loader, val_len=val_len, model_name='classify_5-4.pt')
    # model_test(val_sample_loader, val_len, model='./featuremap_classify_best_1008.pt')

    # performance_test(meta_model='./featuremap_classify_best_1008.pt')

    # train_each(sample_list_train)

    meta_model_list = ['classify0.pt', 'classify1.pt', 'classify2.pt', 'classify3.pt', 'classify4.pt', 'classify5.pt', 'classify6.pt']
    performance_test_each(meta_model_list)

import os
import joblib
import torch
import torchvision

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from CalaNet import CalaNet
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

experiment_radio = 0.1   # the ratio for the sample for experiment
Learning_rate = 1e-3  # learning rate for train
epoch = 100
print_iter = 10

model_list = []


class sample_feature_all(Dataset):
    def __init__(self, sample_list, feature_extraction=None, classify=False, feature_exist=False):
        self.sample_list = sample_list
        self.feature_extraction = feature_extraction
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        self.classify = classify
        self.feature_exist = feature_exist

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


def data_prepare():
    sample_list = joblib.load('sample_list_dhd_traffic_all_with_feature_map.s')  # 'sample_list_all_14.s'
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


def model_train(train_sample_loader, val_sample_loader):
    
    model = CalaNet()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    learning_rate = Learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    # optimizer = torch.optim.Adagrad()
    loss_count = []
    loss_thr = None
    for iter in range(epoch):
        if loss_thr != None:
            if loss_thr <= 0.1:
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

        torch.save(model, './featuremap_classify_all.pt')
        for (test_x, test_y) in val_sample_loader:
            test_x = test_x.float()
            pred = model(test_x)
            loss = loss_fn(pred, test_y)

            print("Validate loss: ", loss.item())
            break

    for (train_x, train_y) in train_sample_loader:
        train_x = train_x.float()
        pred = model(train_x)
        # print(pred.numpy())
        loss = loss_fn(pred, train_y)

        print("Train loss: ", loss.item())
        break


def model_test(val_sample_loader, val_len):
    model = torch.load('./featuremap_classify_all.pt')
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
            for i in choose_index_list:
                if test_y[i][choose_index_list[i]] == 1:
                    correct_num += 1
    print('top 1 test accuracy', correct_num / val_len)


if __name__ == "__main__":
    # sample_list = joblib.load('sample_list_dhd_traffic_all.s')

    # for sample in tqdm(sample_list):
    #     sample.feature_update(feature_extractor)
    
    # joblib.dump(sample_list, 'sample_list_dhd_traffic_all_with_feature_map.s')

    # resnet_model = torchvision.models.resnet18(pretrained=True)

    # feature = feature_extractor(sample_list[0].img_dir)

    # print(dir(resnet_model))

    # chooser = train_from_feature(train_sample_loader, val_sample_loader)

    train_sample_loader, val_sample_loader, val_len = data_prepare()

    chooser = model_train(train_sample_loader, val_sample_loader)
    model_test(val_sample_loader, val_len)

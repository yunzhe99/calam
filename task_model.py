import os
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import joblib

from torch.utils.data import DataLoader

from my_utils.data_io import coco_c, Feature_set
from my_utils.preprocess import resnet_18_encoder
from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Net(nn.Module):

    def __init__(self, image=False):
        super(Net, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view([-1, 512])
        x = self.fc(x)
        return x


def train():
    wandb.init(project='task_model')

    batch_size = Config.batch_size
    learning_rate = Config.learning_rate
    epoch_num = Config.epoch_num

    config = wandb.config

    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.epoch_num = epoch_num
    config.train_set = Config.sample_train
    config.val_set = Config.sample_val

    coco_c_train = coco_c(Config.sample_train)

    train_loader = DataLoader(coco_c_train, batch_size=batch_size, shuffle=True)

    model = Net()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):

        print_loss_list = []

        for data in tqdm(train_loader):
            img, label = data

            img = img.cuda()
            label = torch.as_tensor(label).cuda()

            img_feature = resnet_18_encoder(img)

            img_feature = torch.as_tensor(img_feature).cuda()
            out = model(img_feature)

            loss = criterion(out, label)

            print_loss = loss.data.item()

            print_loss_list.append(print_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc = test(weight=model.state_dict())

        wandb.log({"train_loss": np.mean(np.array(print_loss_list)), "test_acc": test_acc})

        print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(np.array(print_loss_list))))

    torch.save(model.state_dict(), './classify_mini.pt')
    wandb.save('./classify_mini.pt')


def test(weight=None):
    model = Net()
    if weight is None:
        model.load_state_dict(torch.load('classify.pt'))

    model.load_state_dict(weight)

    model = model.cuda()

    model.eval()

    coco_c_val = coco_c(Config.sample_val)
    val_loader = DataLoader(coco_c_val, batch_size=Config.batch_size, shuffle=True)

    eval_acc = 0
    for data in tqdm(val_loader):
        img, label = data

        img = img.cuda()
        label = torch.as_tensor(label).cuda()

        img_feature = resnet_18_encoder(img)

        img_feature = torch.as_tensor(img_feature).cuda()
        out = model(img_feature)

        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    print('Acc: {:.6f}'.format(
        eval_acc / (coco_c_val.__len__())
    ))

    return eval_acc / (coco_c_val.__len__())


def train_from_image(train_loader, val_loader):

    learning_rate = Config.learning_rate
    epoch_num = Config.epoch_num
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):

        print_loss_list = []
        model.train()

        for data in tqdm(train_loader):
            feature, label = data

            feature = feature.cuda()
            label = torch.as_tensor(label).cuda()

            feature = feature.float()

            out = model(feature)

            _, pred = torch.max(out, 1)

            # print(torch.sum(pred))
            # print(pred, label)

            loss = criterion(pred.float(), label.float())

            print_loss = loss.data.item()

            print_loss_list.append(print_loss)

            optimizer.zero_grad()
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

        test_acc = test_from_image(val_loader, model)

        # wandb.log({"train_loss": np.mean(np.array(print_loss_list)), "test_acc": test_acc})

        print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(np.array(print_loss_list))))

    # torch.save(model.state_dict(), './classify_mini.pt')
    return model.state_dict()


def train_from_feature(train_loader, val_loader):
    # wandb.init(project='task_model_from_feature')

    learning_rate = Config.learning_rate
    epoch_num = Config.epoch_num

    # config = wandb.config
    #
    # config.batch_size = batch_size
    # config.learning_rate = learning_rate
    # config.epoch_num = epoch_num

    model = Net()

    # model = torchvision.models.resnet18(pretrained=True)

    if torch.cuda.is_available() is True:
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):

        print_loss_list = []

        for data in tqdm(train_loader):

            feature, label = data

            # print(feature.shape)

            if torch.cuda.is_available() is True:
                feature = feature.cuda()
                label = torch.as_tensor(label).cuda()

            feature = feature.float()
            label = label.float()

            out = model(feature)

            _, pred = torch.max(out, 1)

            # print(torch.sum(pred))

            # loss = criterion(out, label) + 10/(torch.sum(pred) + 1)

            loss = criterion(out, label)

            print_loss = loss.data.item()

            print_loss_list.append(print_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test_acc = test_from_feature(val_loader, model.state_dict())

        # wandb.log({"train_loss": np.mean(np.array(print_loss_list)), "test_acc": test_acc})

        print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(np.array(print_loss_list))))
        # print('test_acc', test_acc)

    # torch.save(model.state_dict(), './classify_mini.pt')
    return model.state_dict()


def test_from_feature(val_loader, weight):

    model = torchvision.models.resnet18(pretrained=False)
    model.load_state_dict(weight)
    if torch.cuda.is_available() is True:
        model = model.cuda()
    model.eval()

    eval_acc = 0
    for data in tqdm(val_loader):
        feature, label = data
        if torch.cuda.is_available() is True:
            feature = feature.cuda()
            label = torch.as_tensor(label).cuda()

        feature = feature.float()

        out = model(feature)

        _, pred = torch.max(out, 1)

        # print(torch.sum(pred))

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    # print('test acc: {:.6f}'.format(
    #     eval_acc / Config.val_set_len
    # ))

    return eval_acc / Config.val_set_len * Config.batch_size


def test_from_image(val_loader, model):

    model.eval()

    eval_acc = 0
    for data in tqdm(val_loader):
        feature, label = data

        feature = feature.cuda()
        label = torch.as_tensor(label).cuda()

        feature = feature.float()

        out = model(feature)

        _, pred = torch.max(out, 1)

        # print(torch.sum(pred))
        # print(pred, label)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    print('test acc: {:.6f}'.format(
        eval_acc / val_loader.__len__() * Config.batch_size
    ))

    return eval_acc / val_loader.__len__() * Config.batch_size


def model_list_train(model_map, feature_set_train, label_set_train, feature_set_val, label_set_val):
    # ?????????model?????????????????????

    model_chooser_list = []

    for model_index in range(len(model_map)):
        Feature_train = Feature_set(feature_set_train, label_dir=label_set_train,
                                    model_performance=model_map[model_index], map_threshold=Config.map_threshold)
        Feature_val = Feature_set(feature_set_val, label_dir=label_set_val,
                                  model_performance=model_map[model_index], map_threshold=Config.map_threshold)

        Feature_train_loader = DataLoader(Feature_train, batch_size=Config.batch_size, shuffle=True)
        Feature_val_loader = DataLoader(Feature_val, batch_size=Config.batch_size, shuffle=True)

        model_chooser = train_from_feature(Feature_train_loader, Feature_val_loader)
        print("\n\n")
        model_chooser_list.append(model_chooser)

    joblib.dump(model_chooser_list, 'model_chooser_list.m')

    return model_chooser_list


if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load(Config.weight_list[0]))
    print(model.state_dict().keys())

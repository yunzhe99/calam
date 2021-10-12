import torch.nn as nn
import torch.nn.functional as F
import torch

drop_p = 0.2
Output_dim = 7

class CalaNet(torch.nn.Module):
    def __init__(self):
        super(CalaNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3584, out_channels=8, kernel_size=3, stride=2, padding=1), # input [batchsize,1,6,52], output [batchsize,8,3,26]
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_p),
            # torch.nn.MaxPool2d(kernel_size=3, stride=3) # input [batchsize,8,156], output [batchsize,8,52]
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=(0,1)), # input [batchsize,8,3,26], output [batchsize,16,1,13]
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_p),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2) # input [batchsize,16,26], output [batchsize,16,13]
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(120, 100),
            torch.nn.Dropout(drop_p),
            torch.nn.ReLU(),
            torch.nn.Linear(100, Output_dim)
        )
        # self.dp = torch.nn.Dropout(drop_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


class CalaNet1L(torch.nn.Module):
    def __init__(self):
        super(CalaNet1L, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=2, padding=1), # input [batchsize,1,6,52], output [batchsize,8,3,26]
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_p),
            # torch.nn.MaxPool2d(kernel_size=3, stride=3) # input [batchsize,8,156], output [batchsize,8,52]
        )
        self.conv1_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1), # input [batchsize,1,6,52], output [batchsize,8,3,26]
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_p),
            # torch.nn.MaxPool2d(kernel_size=3, stride=3) # input [batchsize,8,156], output [batchsize,8,52]
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=(0,1)), # input [batchsize,8,3,26], output [batchsize,16,1,13]
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_p),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2) # input [batchsize,16,26], output [batchsize,16,13]
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(24, 256),
            torch.nn.Dropout(drop_p),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)

        )
        # self.dp = torch.nn.Dropout(drop_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_5(x)
        x = self.conv2(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


import datetime
import torch
import torchvision
import requests

from torch.nn import Module
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

print(f"Last updated: {datetime.datetime.now()}")

numChannels     = 3 #1 for greyscale - 3 for RGB 
hiddenChannels  = 20
outChannels     = 50
in_features     = 800
hidden_units    = 500
classes         = 26 #total number of unique class labels
kernal_size     = (5,5)
stride          = (2,2)

class ec_model(Module):
    #initiaize the parant constructor
    def __init__(self,in_features=in_features,
                 out_features=classes,
                 hidden_units=hidden_units,
                 in_channels=numChannels,
                 out_channels=outChannels,
                 hidden_channels=hiddenChannels,
                 stride=stride,
                 kernal_size=kernal_size):
        super().__init__()
        #initialize teh fist CONV => ReLu => pool layers
        self.conv1 = Conv2d(in_channels=in_channels,
                        out_channels=hidden_channels,
                        kernel_size=kernal_size)
        self.relu1 = ReLU()
        self.maxPool1 = MaxPool2d(kernal_size=stride,
                                  stride=stride)

        #inititalize the second CONV => ReLu => pool layers
        self.conv2 = Conv2d(in_channels=hidden_channels,
                            out_channels=out_channels,
                            kernel_size=kernal_size)
        self.relu2 = ReLU()
        self.maxPool2 = MaxPool2d(kernel_size=stride,
                                  stride=stride)
        
        #initialize set of FC => ReLu layers 
        self.fc1 = Linear(in_features=in_features,out_features=hidden_units)
        self.relu3 = ReLU()

        #initialize our softmax
        self.fc2 = Linear(in_features=hidden_units,out_features=out_features)
        self.logSoftMax = LogSoftmax(dim=1)

        def forward(self,x:torch.Tensor):
            return




#Trainning loop
epochs = 1000


for epoch in range(epochs):
    ec_model.train()
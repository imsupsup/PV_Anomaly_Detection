import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper import *

class CAE_Network(nn.Module):
    def __init__(self, kernel_size, out_size, hidden_size):
        super(CAE_Network, self).__init__()
        self.k = kernel_size
        self.o = out_size
        self.h = hidden_size

        # Encoder Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.o, self.k, padding=1, stride=1),
            nn.BatchNorm2d(self.o),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.o, (self.o)**2, self.k, padding=1, stride=1),
            nn.BatchNorm2d((self.o)**2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Decoder Layers
        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d((self.o)**2, self.o,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.o, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        latent = out
        out = self.trans_conv1(out)
        # print(out.shape)
        out = self.trans_conv2(out)
        # print(out.shape)
        return latent, out



class CAE_Network_3layer(nn.Module):
    def __init__(self, kernel_size, out_size, hidden_size):
        super(CAE_Network_3layer, self).__init__()
        self.k = kernel_size
        self.o = out_size
        self.h = hidden_size

        # Encoder Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.o, self.k, padding=1, stride=1),
            nn.BatchNorm2d(self.o),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.o, (self.o)**2, self.k, padding=1, stride=1),
            nn.BatchNorm2d((self.o)**2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d((self.o)**2, (self.o)**2, self.k, padding=1, stride=1),
            nn.BatchNorm2d((self.o)**2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fcn_enc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim = -1),
            nn.Linear((self.o**2)*(W//8)*(H//8),self.h),
            nn.ReLU(inplace=True)
        )


        # Decoder Layers


        self.fcn_dec = nn.Sequential(
            nn.Linear(self.h,(self.o**2)*(W//8)*(H//8)),
            nn.ReLU(inplace=True)
        )

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d((self.o)**2, (self.o)**2,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        
        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d((self.o)**2, self.o,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.o, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        out = self.conv3(out)
        # print(out.shape)
        out = self.fcn_enc(out)
        latent = out
        out = self.fcn_dec(out)
        out = out.view([-1,(self.o**2),W//8,H//8])
        out = self.trans_conv1(out)
        # print(out.shape)
        out = self.trans_conv2(out)
        # print(out.shape)
        out = self.trans_conv3(out)
        # print(out.shape)
        return latent, out
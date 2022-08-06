# modified from torchvision
import torch.nn as nn
from torch.nn import BatchNorm2d
import torch
import torch.nn.functional as F
import torchvision
resnet = torchvision.models.resnet18(pretrained=True)



class MyNet(nn.Module):
    def __init__(self, class_num):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = nn.Sequential(
            resnet.layer1[0].conv1,
            resnet.layer1[0].bn1,
            nn.ReLU(),
            resnet.layer1[0].conv2,
            resnet.layer1[0].bn2,
            nn.ReLU()
        ) 

        self.droppout = nn.Dropout(0.5)
        self.cls = nn.Linear(64*16*2, class_num)

    def forward(self, x):
        b, c, h, w = x.size()
        cx = torch.linspace(0, 1, h).type_as(x)
        cy = torch.linspace(0, 1, w).type_as(x)
        cx = cx.unsqueeze(1).repeat(1, w)
        cy = cy.unsqueeze(0).repeat(h, 1)
        coord = torch.stack((cx, cy), dim=0)
        x = torch.cat((x, coord.unsqueeze(0).repeat(b, 1, 1, 1)), dim=1) 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.layer1(x)

        x = x.view(x.size(0), -1)
        x = self.droppout(x)
        x = self.cls(x)
        return x




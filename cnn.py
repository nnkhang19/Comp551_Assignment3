import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, args):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(in_channels[0]), kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)
        self.conv2 = nn.Conv2d(in_channels=int(in_channels[0]), out_channels=int(in_channels[1]), kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)
        
        img_size = args.img_size // 4
        self.fc1 = nn.Linear(int(in_channels[1]) * (img_size) * (img_size), 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.args = args
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Resnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = models.resnet50(pretrained = True)
        fc_in_dim = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.eval() 

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_in_dim),
            nn.Linear(fc_in_dim, args.hidden_dim), 
            nn.ReLU(),
            #nn.Dropout(0.5), 
            nn.Linear(args.hidden_dim, args.num_class)
        )
    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
        return self.fc(x)
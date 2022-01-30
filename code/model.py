import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        x = self.bn1(self.pool(torch.nn.functional.relu(self.conv1(x))))
        x = self.bn2(self.pool(torch.nn.functional.relu(self.conv2(x))))
        x = self.bn3(self.pool(torch.nn.functional.relu(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

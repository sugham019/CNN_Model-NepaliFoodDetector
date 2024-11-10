from dataclasses import dataclass
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, image_res: tuple[int, int], outputClasses: int):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=4)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.fc1 = nn.Linear(64 * ((image_res[0]//2)//2) * ((image_res[1]//2)//2), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, outputClasses)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.leaky_relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) 
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.output(x)
        return x
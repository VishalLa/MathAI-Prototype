import torch
import torch.nn as nn 
import sys
import os

'''
output_size = ((input_size - kernel_size + 2*padding)/stride) + 1
input_size = 28
'''

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_classes=18):
        super(CNN, self).__init__()

        # Input: 1x28x28

        self.conv_layers = nn.Sequential(
            # 1st conv layer
            # 1X28X28 -> 16X10X10
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(5,5),
                padding=2,
                stride=3,
                bias=True
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            # 2nd conv layer
            # Conv2D: 16X10X10 -> 32X10X10
            # MaxPool2D: 32X10X10 -> 32X5X5
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3,3),
                padding=1,
                stride=1,
                bias=True
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),

            # 3rd conv layer
            # Conv2d: 32x5x5 -> 64x7x7
            # MaxPool2d: 64x7x7 -> 64x3x3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                padding=2,
                stride=1,
                bias=True
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.3),

            # 4th conv layer
            # 64X14X14 -> 128X14X14
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                padding=1,
                stride=1,
                bias=True
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.4),
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
            output_size = self.conv_layers(dummy_input).view(1, -1).shape[1]

        self.flatten = nn.Flatten(),

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=output_size, out_features=128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(in_features=64, out_features=num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(f"Shape after conv layers: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Shape after flatten: {x.shape}")
        x = self.fc_layers(x)
        return x 
    

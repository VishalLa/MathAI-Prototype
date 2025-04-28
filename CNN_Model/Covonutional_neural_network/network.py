import torch
import torch.nn as nn 
import sys
import os


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3,3),
                padding=1,
                stride=1,
                bias=True
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1),

            # 2nd conv layer 
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3,3),
                padding=1,
                stride=1,
                bias=True
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2, padding=1),

            # 3rd conv layer 
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
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1),

            # 4th conv layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                padding=1,
                stride=1,
                bias=True
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1),

            # 5th conv layer
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
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1)
        )

        # Calculate flattened size dynamically
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
        #     output_size = self.conv_layers(dummy_input).view(1, -1).shape[1]

        nn.Flatten(),

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=258, bias=True),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=258, out_features=124, bias=True),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=124, out_features=64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=64, out_features=20)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(f"Shape after conv layers: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Shape after flatten: {x.shape}")
        x = self.fc_layers(x)
        return x 
    

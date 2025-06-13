import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from required_variables import classes

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------
# Squeeze-and-Excitation (SE) Block
# ------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.se(x)
        return x * weight


# ------------------------------
# CNN Model
# ------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=classes):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=3, bias=True),  # -> 16x22x22
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.4),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=True),  # -> 32x22x22
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x11x11
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=2, stride=1, bias=True),  # -> 64x13x13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.4),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True),  # -> 128x13x13
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 128x4x4
            nn.Dropout2d(0.4),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),  # -> 256x2x2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.4),
        )

        self.se = SEBlock(channel=256)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)
            dummy_output = self.se(self.conv_layers(dummy_input))
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ------------------------------
# Focal Loss
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

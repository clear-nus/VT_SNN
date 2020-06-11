"""CNN3D model definitions."""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class CNN3D(nn.Module):
    def __init__(self):

        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=4,
            kernel_size=(7, 3, 3),
            stride=(4, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=4,
            out_channels=8,
            kernel_size=(5, 3, 3),
            stride=(3, 1, 1),
        )

    def forward(self, x):
        batch_size, C, H, W, sequence_size = x.size()
        x = x.view([batch_size, C, sequence_size, H, W])
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view([batch_size, np.prod([8, 26, 5, 3])])

        return out


class TactCNN3D(nn.Module):
    def __init__(self, output_size):
        super(TactCNN3D, self).__init__()

        self.cnn_left = CNN3D()
        self.cnn_right = CNN3D()

        # 8, 35, 5, 3
        self.fc = nn.Linear(np.prod([8, 26, 5, 3]) * 2, output_size)

        self.drop = nn.Dropout(0.5)

    def forward(self, x_left, x_right):
        out_left = self.cnn_left(x_left)
        out_right = self.cnn_right(x_right)
        out = torch.cat([out_left, out_right], dim=1)
        out = self.fc(out)
        # out = self.drop(out)

        return out


class VisCNN3D(nn.Module):
    def __init__(self, output_size):
        super(VisCNN3D, self).__init__()
        self.input_size = 8 * 5 * 3

        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=2,
            kernel_size=(10, 5, 5),
            stride=(5, 2, 2),
        )
        self.conv2 = nn.Conv3d(
            in_channels=2,
            out_channels=4,
            kernel_size=(5, 3, 3),
            stride=(3, 2, 2),
        )
        self.conv3 = nn.Conv3d(
            in_channels=4,
            out_channels=8,
            kernel_size=(5, 3, 3),
            stride=(3, 2, 2),
        )

        # Define the output layer
        self.fc = nn.Linear(np.prod([8, 6, 6, 5]), output_size)

    def forward(self, x):
        batch_size, C, H, W, sequence_size = x.size()
        x = x.view([batch_size, C, sequence_size, H, W])
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = out.view([batch_size, np.prod([8, 6, 6, 5])])
        out = self.fc(out)
        # out = self.drop(out)

        return out


class MmCNN3D(nn.Module):
    def __init__(self, output_size):

        super(MmCNN3D, self).__init__()

        self.cnn_left = CNN3D()
        self.cnn_right = CNN3D()

        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=2,
            kernel_size=(10, 5, 5),
            stride=(5, 2, 2),
        )
        self.conv2 = nn.Conv3d(
            in_channels=2,
            out_channels=4,
            kernel_size=(5, 3, 3),
            stride=(3, 2, 2),
        )
        self.conv3 = nn.Conv3d(
            in_channels=4,
            out_channels=8,
            kernel_size=(5, 3, 3),
            stride=(3, 2, 2),
        )

        # 8, 35, 5, 3
        # Define the output layer
        self.fc = nn.Linear(
            np.prod([8, 26, 5, 3]) * 2 + np.prod([8, 6, 6, 5]), output_size,
        )

        self.drop = nn.Dropout(0.5)

    def forward(self, tact_left, tact_right, vision):
        tact_left = self.cnn_left(tact_left)
        tact_right = self.cnn_right(tact_right)

        batch_size, C, H, W, sequence_size = vision.size()
        vision = vision.view([batch_size, C, sequence_size, H, W])
        vision = self.conv1(vision)
        vision = F.relu(vision)
        vision = self.conv2(vision)
        vision = F.relu(vision)
        vision = self.conv3(vision)
        vision = F.relu(vision)
        vision = vision.view([batch_size, np.prod([8, 6, 6, 5])])

        out = torch.cat([tact_left, tact_right, vision], dim=1)
        out = self.fc(out)

        # out = self.drop(out)
        return out

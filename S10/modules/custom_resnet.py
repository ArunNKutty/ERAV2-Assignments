"""Module to define the model."""

import torch.nn as nn
import torch.nn.functional as F
import torchinfo


############# Assignment 10 Model #############
class CustomResNet(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.02

    def __init__(self):
        super().__init__()

        #  Model Notes

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        # 1. Input size: 32x32x3
        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer1: X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.layer1_x = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer1: R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.layer1_r1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer 2: Conv 3x3 [256k], MaxPooling2D, BN, ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer 3: X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.layer3_x = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer 3: R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.layer3_r2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # MaxPooling with Kernel Size 4
        # If stride is None, it is set to kernel_size
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)

        # FC Layer
        self.fc = nn.Linear(512, 10)

    def print_view(self, x, msg=""):
        """Print shape of the model"""
        if self.print_shape:
            if msg != "":
                print(msg, "\n\t", x.shape, "\n")
            else:
                print(x.shape)

    def forward(self, x):
        """Forward pass"""

        # PrepLayer
        x = self.prep(x)
        self.print_view(x, "PrepLayer")

        # Layer 1
        x = self.layer1_x(x)
        self.print_view(x, "Layer 1, X")
        r1 = self.layer1_r1(x)
        self.print_view(r1, "Layer 1, R1")
        x = x + r1
        self.print_view(x, "Layer 1, X + R1")

        # Layer 2
        x = self.layer2(x)
        self.print_view(x, "Layer 2")

        # Layer 3
        x = self.layer3_x(x)
        self.print_view(x, "Layer 3, X")
        r2 = self.layer3_r2(x)
        self.print_view(r2, "Layer 3, R2")
        x = x + r2
        self.print_view(x, "Layer 3, X + R2")

        # MaxPooling
        x = self.maxpool(x)
        self.print_view(x, "Max Pooling")

        # FC Layer
        # Reshape before FC such that it becomes 1D
        x = x.view(x.shape[0], -1)
        self.print_view(x, "Reshape before FC")
        x = self.fc(x)
        self.print_view(x, "After FC")

        # Softmax
        return F.log_softmax(x, dim=-1)

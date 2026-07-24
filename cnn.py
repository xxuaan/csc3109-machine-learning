import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BilinearCNN(nn.Module):
    """Symmetric Bilinear CNN (Lin et al., 2015) with a VGG16-BN backbone."""

    def __init__(self, num_classes, pretrained=False, freeze_backbone=False):
        super().__init__()
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16_bn(weights=weights)

        # Drop the final MaxPool2d (last layer in vgg.features) so we keep the
        # full-resolution conv5_3 feature map (512 x 14 x 14 for 224x224 input).
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])
        self.feature_dim = 512  # output channels of vgg16_bn conv5_3

        if freeze_backbone:
            self.freeze_backbone()

        # Bilinear descriptor size = feature_dim ** 2
        self.classifier = nn.Linear(self.feature_dim * self.feature_dim, num_classes)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def bilinear_pool(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        x = x.view(B, C, H * W)                          # [B, C, HW]
        x = torch.bmm(x, x.transpose(1, 2)) / (H * W)     # [B, C, C] outer product, avg pooled
        x = x.view(B, C * C)                              # flatten
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-5)  # signed sqrt
        x = F.normalize(x, p=2, dim=1)                    # L2 norm
        return x

    def forward(self, x):
        x = self.features(x)          # [B, 512, 14, 14]
        x = self.bilinear_pool(x)     # [B, 262144]
        x = self.classifier(x)        # [B, num_classes]
        return x
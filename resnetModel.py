# Define the ResNet model
import torch.nn as nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')  # Load ResNet18 with ImageNet weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 256)  # Change output layer for similarity

    def forward(self, x):
        return self.model(x)
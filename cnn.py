import torch.nn as nn
from torchvision import models

# Pre-trained ResNet50 as a base model for embedding extraction
class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        # Load a pre-trained ResNet50 model
        self.base_model = models.resnet50(pretrained=True)
        # Replace the last fully connected layer with a new one for embeddings
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)  # Embedding size
        )

        # Unfreeze deeper layers for fine-tuning
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)

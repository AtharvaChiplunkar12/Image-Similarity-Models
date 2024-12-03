from torchvision import models
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load a pre-trained ResNet model and remove the last fully connected layer
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1])) 
        
        # Adding a new fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),  
            nn.Sigmoid()
        )

    def forward_once(self, x):
        x = self.resnet(x)  
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
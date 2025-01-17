import torch
import torch.nn as nn
from torchvision import models

# https://www.mathworks.com/help/deeplearning/ug/convert-classification-network-into-regression-network.html
# https://github.com/pytorch/vision/tree/main/torchvision/models
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch

class ResNet(nn.Module):
    def __init__(self, model='RESNET_18'):
        super().__init__()

        self.resnet = None
        if model == 'RESNET_34':
            self.resnet = models.resnet34()
        elif model == 'RESNET_50':
            self.resnet = models.resnet50()
        else: # ResNet18
            self.resnet = models.resnet18()
        
        layers = list(self.resnet.children())[:-1] # exclude the last fc layer
        self.features = nn.Sequential(*layers)  
        self.fc = nn.Linear(self.resnet.fc.in_features, 1) # map to one
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
  
class MobileNet(nn.Module):
    def __init__(self, model='MOBILENET_V3_SMALL'):
        super().__init__()

        self.mobilenet = None
        if model == 'MOBILENET_V2':
            self.mobilenet = models.mobilenet_v2()
        elif model == 'MOBILENET_V3_LARGE':
            print('MNV3L')
            self.mobilenet = models.mobilenet_v3_large()
        else: 
            self.mobilenet = models.mobilenet_v3_small()

        # Modify the classifier to output a single value
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.mobilenet(x)
    
def get_model(model):
    if model.find('RESNET') != -1:
        print('RESNET')
        return ResNet(model=model)
    elif model.find('MOBILENET') != -1:
        print('MOBILENET')
        return MobileNet(model=model)
    return None
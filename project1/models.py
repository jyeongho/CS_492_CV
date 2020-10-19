import torch
import torch.nn as nn
from torch.nn import init

import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

__all__ = ['contrastive_model']


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)  


class contrastive_model(nn.Module):
    def __init__(self, class_num, out_dim):
        super(contrastive_model, self).__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b0')
        feature_dim = self.base_model._fc.in_features
        self.base_model._fc = nn.Identity()

        self.linear1 = nn.Linear(feature_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.classifier = nn.Linear(out_dim, class_num)

        self.linear1.apply(weights_init_classifier)
        self.linear2.apply(weights_init_classifier)
        self.classifier.apply(weights_init_classifier)
    
    def forward(self, x):
        h = self.base_model(x)
        z = self.linear2(F.relu(self.linear1(h)))
        pred = self.classifier(z)
        return z, pred
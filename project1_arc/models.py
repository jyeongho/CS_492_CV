import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

__all__ = ['contrastive_model_b0', 'ArcMargin']


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)


class contrastive_model_b0(nn.Module):
    def __init__(self, class_num, out_dim):
        super(contrastive_model_b0, self).__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b0')
        feature_dim = self.base_model._fc.in_features
        self.base_model._fc = nn.Identity()

        self.linear1 = nn.Linear(feature_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.classifier = nn.Linear(out_dim, class_num, bias=False)

        self.linear1.apply(weights_init_classifier)
        self.linear2.apply(weights_init_classifier)
        self.classifier.apply(weights_init_classifier)
    
    def forward(self, x):
        h = self.base_model(x)
        z = self.linear2(F.relu(self.linear1(h)))
        z = F.normalize(z)
        pred = self.classifier(z)
        return z, pred

# forward embed feature from contrastive_model_b0 into ArcMargin model and return modified pred value for calculating arcface loss
# reference : ArcMarginProduct class in "https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py"
# To calculate arcface loss, we need to do additional process to model prediction which is output of the classifier layer.
# However, at test time, we need to get original model prediction value to calculate accuracy
# Therefore, use ArcMargin model to update parameter of classifier layer. After updating, copy the value of params into classifier of contrastive_model_b0.
class ArcMargin(nn.Module):
    def __init__(self, out_dim, num_classes, s=30.0, m=0.5):
        super(ArcMargin, self).__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.classifier = nn.Linear(out_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, z_xi, z_xj, target_x):
        cos_xi = F.linear(z_xi, F.normalize(self.classifier.weight))
        sin_xi = torch.sqrt((1.0 - torch.pow(cos_xi, 2)).clamp(0, 1))
        phi_xi = cos_xi * self.cos_m - sin_xi * self.sin_m
        phi_xi = torch.where(cos_xi > self.threshold, phi_xi, cos_xi - self.mm)

        cos_xj = F.linear(z_xj, F.normalize(self.classifier.weight))
        sin_xj = torch.sqrt((1.0 - torch.pow(cos_xj, 2)).clamp(0, 1))
        phi_xj = cos_xj * self.cos_m - sin_xj * self.sin_m
        phi_xj = torch.where(cos_xj > self.threshold, phi_xj, cos_xj - self.mm)

        one_hot_x = torch.zeros(cos_xi.size(), requires_grad=True).cuda()
        one_hot_x.scatter_(1, target_x.view(-1, 1).long(), 1)

        pred_xi = (one_hot_x * phi_xi) + ((1.0 - one_hot_x) * cos_xi)
        pred_xj = (one_hot_x * phi_xj) + ((1.0 - one_hot_x) * cos_xj)
        pred_xi *= self.s
        pred_xj *= self.s

        return pred_xi, pred_xj
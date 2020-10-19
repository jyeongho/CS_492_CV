import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math

__all__ = ['ContrastiveLoss']

# We use NTXentLoss from "https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py".
class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool).cuda()
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_label = torch.zeros_like(pred).cuda()
            smooth_label.fill_(self.smoothing / (self.num_classes - 1))
            smooth_label.scatter_(dim=1, index=target.data.unsqueeze(1), value=self.confidence)
        return torch.mean(torch.sum(-smooth_label * pred, dim=-1))


class ContrastiveLoss():
    def __init__(self, temperature, consistency_epoch, batchsize_label, batchsize_unlabel, num_classes, smoothing, use_cosine_similarity=True):
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.consistency_epoch = consistency_epoch
        self.batch_size = batchsize_label + batchsize_unlabel
        self.criterion = NTXentLoss(self.batch_size, self.temperature, self.use_cosine_similarity).cuda()
        self.num_calsses = num_classes
        self.smoothing = smoothing

    def __call__(self, z_xi, z_xj, pred_xi, pred_xj, target_x, z_ui, z_uj, epoch):
        z_xi = F.normalize(z_xi, dim=1)
        z_xj = F.normalize(z_xj, dim=1)
        z_ui = F.normalize(z_ui, dim=1)
        z_uj = F.normalize(z_uj, dim=1)
        
        i_repr = torch.cat([z_xi, z_ui], dim=0)
        j_repr = torch.cat([z_xj, z_uj], dim=0)
        loss = self.criterion(i_repr, j_repr)
        if epoch <= self.consistency_epoch:
            return loss
        else:
            cross_entropy_criterion = LabelSmoothingLoss(self.num_calsses, self.smoothing).cuda()
            pred_repr = torch.cat([pred_xi, pred_xj], dim=0)
            label_repr = torch.cat([target_x, target_x], dim=0)
            cross_entropy_loss = cross_entropy_criterion(pred_repr, label_repr)
            return loss + 100 * cross_entropy_loss


            



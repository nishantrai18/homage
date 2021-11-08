import torch
import sys
import time
import math

import torch.nn as nn
import torch.nn.functional as F
import model_utils as mu

sys.path.append('../utils')
from utils import calc_topk_accuracy
from random import random


eps = 1e-5
INF = 1000.0


class MemoryBank(nn.Module):

    def __init__(self, size):
        super(MemoryBank, self).__init__()
        self.maxlen = size
        self.dim = None
        self.bank = None

    def bootstrap(self, X):
        self.dim = X.shape[1:]
        gcd = math.gcd(X.shape[0], self.maxlen)
        self.bank = torch.cat([X[:gcd]] * (self.maxlen // gcd), dim=0).detach().to(X.device)
        assert self.bank.shape[0] == self.maxlen, "Invalid shape: {}".format(self.bank.shape)
        self.bank.requires_grad = False

    def update(self, X):
        # Initialize the memory bank
        N = X.shape[0]
        if self.dim is None:
            self.bootstrap(X)

        assert X.shape[1:] == self.dim, "Invalid size: {} {}".format(X.shape, self.dim)
        self.bank = torch.cat([self.bank[N:], X.detach().to(X.device)], dim=0).detach()

    def fetchBank(self):
        if self.bank is not None:
            assert self.bank.requires_grad is False, "Bank grad not false: {}".format(self.bank.requires_grad)
        return self.bank

    def fetchAppended(self, X):
        if self.bank is None:
            self.bootstrap(X)
            return self.fetchAppended(X)
        assert X.shape[1:] == self.bank.shape[1:], "Invalid shapes: {}, {}".format(X.shape, self.bank.shape)
        assert self.bank.requires_grad is False, "Bank grad not false: {}".format(self.bank.requires_grad)
        return torch.cat([X, self.bank], dim=0)


class WeightNormalizedMarginLoss(nn.Module):
    def __init__(self, target):
        super(WeightNormalizedMarginLoss, self).__init__()

        self.target = target.float().clone()

        # Parameters for the weight loss
        self.f = 0.5
        self.one_ratio = self.target[self.target == 1].numel() / (self.target.numel() * 1.0)

        # Setup weight mask
        self.weight_mask = target.float().clone()
        self.weight_mask[self.weight_mask >= 1.] = self.f * (1 - self.one_ratio)
        self.weight_mask[self.weight_mask <= 0.] = (1. - self.f) * self.one_ratio

        # Normalize the weight accordingly
        self.weight_mask = self.weight_mask.to(self.target.device) / (self.one_ratio * (1. - self.one_ratio))

        self.hinge_target = self.target.clone()
        self.hinge_target[self.hinge_target >= 1] = 1
        self.hinge_target[self.hinge_target <= 0] = -1

        self.dummy_target = self.target.clone()

        self.criteria = nn.HingeEmbeddingLoss(margin=((1 - self.f) / (1 - self.one_ratio)))

    def forward(self, value):
        distance = 1.0 - value
        return self.criteria(self.weight_mask * distance, self.hinge_target)


class SimHandler(nn.Module):

    def __init__(self):
        super(SimHandler, self).__init__()

    def verify_shape_for_dot_product(self, mode0, mode1):

        B, N, D = mode0.shape
        assert (B, N, D) == tuple(mode1.shape), \
            "Mismatch between mode0 and mode1 features: {}, {}".format(mode0.shape, mode1.shape)

        # dot product in mode0-mode1 pair, get a 4d tensor. First 2 dims are from mode0, the last from mode1
        nmode0 = mode0.view(B * N, D)
        nmode1 = mode1.view(B * N, D)

        return nmode0, nmode1, B, N, D

    def get_feature_cross_pair_score(self, mode0, mode1):
        """
            Gives us all pair wise scores
            (mode0/mode1)features: [B, N, D], [B2, N2, D]
            Returns 4D pair score tensor
        """

        B1, N1, D1 = mode0.shape
        B2, N2, D2 = mode1.shape

        assert D1 == D2, "Different dimensions: {} {}".format(mode0.shape, mode1.shape)
        nmode0 = mode0.view(B1 * N1, D1)
        nmode1 = mode1.view(B2 * N2, D2)

        score = torch.matmul(
            nmode0.reshape(B1 * N1, D1),
            nmode1.reshape(B2 * N2, D1).transpose(0, 1)
        ).view(B1, N1, B2, N2)

        return score

    def get_feature_pair_score(self, mode0, mode1):
        """
            Returns aligned pair scores
            (pred/gt)features: [B, N, D]
            Returns 2D pair score tensor
        """

        nmode0, nmode1, B, N, D = self.verify_shape_for_dot_product(mode0, mode1)
        score = torch.bmm(
            nmode0.view(B * N, 1, D),
            nmode1.view(B * N, D, 1)
        ).view(B, N)

        return score

    def l2NormedVec(self, x, dim=-1):
        assert x.shape[dim] >= 256, "Invalid dimension for reduction: {}".format(x.shape)
        return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class AlignSimHandler(SimHandler):

    def __init__(self, instance_label):
        super(AlignSimHandler, self).__init__()

        self.target = instance_label.clone()
        self.criterion_base = nn.CrossEntropyLoss()
        self.criterion = lambda x, y: self.criterion_base(x, y.float().argmax(dim=1))

        self.accuracyKList = [1, 3]

    def forward(self, mode0, mode1):
        score = self.get_feature_cross_pair_score(mode0, mode1)

        B, NS, _, _ = score.shape

        score_flattened = score.view(B * NS, B * NS)

        assert self.target.shape == (B, NS, B, NS), "Invalid shape: {}, {}".format(self.target.shape, score.shape)

        instance_target_flattened = self.target.view(B * NS, B * NS)
        instance_target_lbl = instance_target_flattened.float().argmax(dim=1)

        # Compute and log performance metrics
        stats = {}
        topKs = calc_topk_accuracy(score_flattened, instance_target_lbl, self.accuracyKList)
        for i in range(len(self.accuracyKList)):
            stats["acc" + str(self.accuracyKList[i])] = topKs[i]

        return self.criterion(score_flattened, instance_target_flattened), stats


class CosSimHandler(SimHandler):

    def __init__(self):
        super(CosSimHandler, self).__init__()

        self.target = None
        self.criterion = nn.MSELoss()

    def score(self, mode0, mode1):
        cosSim = self.get_feature_pair_score(self.l2NormedVec(mode0), self.l2NormedVec(mode1))

        assert cosSim.min() >= -1. - eps, "Invalid value for cos sim: {}".format(cosSim)
        assert cosSim.max() <= 1. + eps, "Invalid value for cos sim: {}".format(cosSim)

        return cosSim

    def forward(self, mode0, mode1):
        score = self.score(mode0, mode1)

        if self.target is None:
            self.target = torch.ones_like(score)

        stats = {"m": score.mean()}

        return self.criterion(score, self.target), stats


class CorrSimHandler(SimHandler):

    def __init__(self):
        super(CorrSimHandler, self).__init__()

        self.shapeMode0, self.shapeMode1 = None, None
        self.runningMeanMode0 = None
        self.runningMeanMode1 = None

        self.retention = 0.7
        self.target = None

        self.criterion = nn.L1Loss()

        self.noInitYet = True

    @staticmethod
    def get_ovr_mean(mode):
        return mode.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).detach().cpu()

    def init_vars(self, mode0, mode1):

        self.shapeMode0 = mode0.shape
        self.shapeMode1 = mode1.shape

        assert len(self.shapeMode0) == 3

        self.runningMeanMode0 = self.get_ovr_mean(mode0)
        self.runningMeanMode1 = self.get_ovr_mean(mode1)

        self.noInitYet = False

    def update_means(self, mean0, mean1):

        self.runningMeanMode0 = (self.runningMeanMode0 * self.retention) + (mean0 * (1. - self.retention))
        self.runningMeanMode1 = (self.runningMeanMode1 * self.retention) + (mean1 * (1. - self.retention))

    def get_means_on_device(self, device):
        return self.runningMeanMode0.to(device), self.runningMeanMode1.to(device)

    def score(self, mode0, mode1):

        if self.noInitYet:
            self.init_vars(mode0, mode1)

        meanMode0 = self.get_ovr_mean(mode0)
        meanMode1 = self.get_ovr_mean(mode1)
        self.update_means(meanMode0.detach().cpu(), meanMode1.detach().cpu())
        runningMean0, runningMean1 = self.get_means_on_device(mode0.device)

        corr = self.get_feature_pair_score(
            self.l2NormedVec(mode0 - runningMean0),
            self.l2NormedVec(mode1 - runningMean1)
        )

        assert corr.min() >= -1. - eps, "Invalid value for correlation: {}".format(corr)
        assert corr.max() <= 1. + eps, "Invalid value for correlation: {}".format(corr)

        return corr

    def forward(self, mode0, mode1):
        score = self.score(mode0, mode1)

        if self.target is None:
            self.target = torch.ones_like(score)

        stats = {"m": score.mean()}

        return self.criterion(score, self.target), stats


class DenseCorrSimHandler(CorrSimHandler):

    def __init__(self, instance_label):
        super(DenseCorrSimHandler, self).__init__()

        self.target = instance_label.float().clone()
        # self.criterion = WeightNormalizedMSELoss(self.target)
        self.criterion = WeightNormalizedMarginLoss(self.target)

    def get_feature_pair_score(self, mode0, mode1):
        return self.get_feature_cross_pair_score(mode0, mode1)

    def forward(self, mode0, mode1):
        score = self.score(mode0, mode1)

        B, N, B2, N2 = score.shape
        assert (B, N) == (B2, N2), "Invalid shape: {}".format(score.shape)
        assert score.shape == self.target.shape, "Invalid shape: {}, {}".format(score.shape, self.target.shape)

        stats = {
            "m": (self.criterion.weight_mask * score).mean(),
            "m-": score[self.target <= 0].mean(),
            "m+": score[self.target > 0].mean(),
        }

        return self.criterion(score), stats


class DenseCosSimHandler(CosSimHandler):

    def __init__(self, instance_label):
        super(DenseCosSimHandler, self).__init__()

        self.target = instance_label.float()
        # self.criterion = WeightNormalizedMSELoss(self.target)
        self.criterion = WeightNormalizedMarginLoss(self.target)

    def get_feature_pair_score(self, mode0, mode1):
        return self.get_feature_cross_pair_score(mode0, mode1)

    def forward(self, mode0, mode1):
        score = self.score(mode0, mode1)
        assert score.shape == self.target.shape, "Invalid shape: {}, {}".format(score.shape, self.target.shape)

        stats = {
            "m": (self.criterion.weight_mask * score).mean(),
            "m-": score[self.target <= 0].mean(),
            "m+": score[self.target > 0].mean(),
        }

        return self.criterion(score), stats


class InterModeDotHandler(nn.Module):

    def __init__(self, last_size=1):
        super(InterModeDotHandler, self).__init__()

        self.cosSimHandler = CosSimHandler()
        self.last_size = last_size

    def contextFetHelper(self, context):
        context = context[:, -1, :].unsqueeze(1)
        context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
        return context

    def fetHelper(self, z):
        B, N, D, S, S = z.shape
        z = z.permute(0, 1, 3, 4, 2).contiguous().view(B, N * S * S, D)
        return z

    def dotProdHelper(self, z, zt):
        return self.cosSimHandler.get_feature_cross_pair_score(
            self.cosSimHandler.l2NormedVec(z), self.cosSimHandler.l2NormedVec(zt)
        )

    def get_cluster_dots(self, feature):
        fet = self.fetHelper(feature)
        return self.dotProdHelper(fet, fet)

    def forward(self, context=None, comp_pred=None, comp_fet=None):
        cdot = self.fetHelper(comp_fet)
        return self.dotProdHelper(cdot, cdot), cdot

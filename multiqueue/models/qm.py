import collections
from copy import deepcopy

import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.nn import init


class QM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, queues, momentum, wd=0.05):
        ctx.queues = queues
        ctx.wd = wd
        ctx.features = queues.get_features()
        ctx.m = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        # outputs = torch.einsum("nk,ck->nc", inputs, ctx.features)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        ctx.queues.update(inputs, targets, ctx.wd)
        return grad_inputs, None, None, None, None


def qm(inputs, indexes, queues, momentum, weight_decay=0.05):
    return QM.apply(inputs, indexes, queues, momentum, weight_decay)



class QueueMemory(nn.Module, ABC):
    def __init__(
        self, centroids, temp=0.05, momentum=0.2, batch_size=128
    ):
        super(QueueMemory, self).__init__()

        self.momentum = torch.nn.Parameter(torch.tensor(momentum))
        self.temp = torch.nn.Parameter(torch.tensor(temp))
        self.groups = batch_size // 16
        self.weight = None

        self.centroids = centroids

    def refresh(self, centroids):
        self.centroids = centroids

    def forward(self, inputs, targets, weight_decay=0.1):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = qm(inputs, targets, self.centroids, weight_decay)

        outputs /= self.temp

        loss = F.cross_entropy(outputs, targets)
        return loss

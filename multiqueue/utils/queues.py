import collections
import random
import math

import torch
from torch import nn
import torch.nn.functional as F


class Queues(nn.Module):
    def __init__(self, queue_max=16, momentum=0.9, num_features=2048, sample_num=8, id_sample_num=16, batch_size=256,
                 lr=0.01,random_centroid=0.2):
        """

        Args:
            labels: input labels
            queues: cluster queues
            threshold:the least number of queues
        """
        super(Queues, self).__init__()
        self.queue_max = queue_max
        self.momentum = momentum
        self.sample_num = sample_num
        self.weights = self.__create_momentum_weights()
        self.id_sample_num = id_sample_num
        self.ids_num = batch_size // id_sample_num
        self.lr = lr
        self.num_features = num_features

        self.one_weight = 0.5
        self.p=random_centroid
        # self.weights=None
        # raise RuntimeError('weights: ',self.weights)

    # @torch.no_grad()
    def __create_momentum_weights(self) -> torch.Tensor:
        return torch.tensor([(1 - self.momentum) * (self.momentum ** i)
                             for i in range(self.sample_num)]).unsqueeze(0).t()

    @torch.no_grad()
    def init_queue(self, features, labels, optimizer=None, lr=0.01):

        self.feature_queues = collections.defaultdict(list)
        for i, label in enumerate(labels):
            # print('label: ', label)
            if label == -1:
                continue
            # print('feat_: {}, i: {}'.format(len(features),i))
            self.feature_queues[labels[i]].append(features[i])

        self.feature_queues = [
            torch.stack(self.feature_queues[idx], dim=0).mean(0).unsqueeze(0) for idx in
            sorted(self.feature_queues.keys())
        ]
        self.__centroids = torch.cat([i.mean(0).unsqueeze(0) for i in self.feature_queues], dim=0).cuda()
        self.features = torch.stack(self.feature_queues, dim=0)
        for i in range(len(self.feature_queues)):
            self.feature_queues[i]=self.__centroids[i].unsqueeze(0)
        
        # multi queue
        self.current_flags = torch.zeros(len(self.feature_queues), dtype=torch.int)

        for i in range(len(self.feature_queues)):
            while self.feature_queues[i].size()[0] < self.queue_max:
                self.feature_queues[i] = torch.cat(
                    [self.feature_queues[i],
                     self.feature_queues[i][0:self.queue_max - self.feature_queues[i].size()[0]]])
        
        # cut queue feature
        for i in range(len(self.feature_queues)):
            self.feature_queues[i] = self.feature_queues[i][:self.queue_max]
        # self.feature_queues=torch.tensor(self.feature_queues)

        if optimizer is not None:
            params = [{"params": [value]} for value in self.feature_queues]
            optimizer.add_param_group({"params": params, "lr": lr})

    @torch.no_grad()
    def update(self, input_features, labels, weight_decay=0.05, centroid_weight=0.15):

        for input_feature, label in zip(input_features, labels):
            queue_length = self.feature_queues[label].shape[0]
            current_flag = self.current_flags[label]
            self.current_flags[label] = (self.current_flags[label] + 1) % queue_length
            # old_feature = self.feature_queues[label][current_flag]
            self.feature_queues[label][current_flag] = centroid_weight * self.feature_queues[label][current_flag] + \
                                                       (1 - centroid_weight) * input_feature
            self.feature_queues[label][current_flag]/=self.feature_queues[label][current_flag].norm()
            self.feature_queues[label][current_flag] = input_feature.cpu()

    def __momentum_feature(self, tensors: torch.Tensor) -> torch.Tensor:
        tensors = tensors * self.weights
        return tensors.sum(0)

    def get_features(self) -> torch.Tensor:

       
        out_features = [feature_queue[random.randint(0, feature_queue.shape[0] - 1)]
                        for feature_queue in self.feature_queues]
        out_features = torch.stack(out_features).cuda()

       
        return out_features
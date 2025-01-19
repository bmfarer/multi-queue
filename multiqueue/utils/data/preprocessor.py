from __future__ import absolute_import

import copy
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None,transform_key=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform_k=transform_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        # img_k=copy.deepcopy(img)

        if self.transform is not None:
            img_q = self.transform(img)
            if self.transform_k is not None:
                img_k=self.transform_k(img)
                return img_q,img_k, fname, pid, camid, index
            return img_q, fname, pid, camid, index

        return img, fname, pid, camid, index

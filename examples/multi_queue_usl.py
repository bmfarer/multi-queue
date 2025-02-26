# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import argparse
# import collections
import os.path as osp
import random
import re
import sys
import time
from datetime import timedelta

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.models.qm import QueueMemory
from clustercontrast.trainers_temp import ClusterContrastTrainer
from clustercontrast.utils.queues import Queues
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T2
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler,RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint

start_epoch = best_mAP = 0
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4'



def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T2.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create auto_encoder
    last_epoch = 0
    model = create_model(args)
    if args.load_last:
        print('==> load checkpoint...')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']
        print('==> load success!')

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if
              value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,
                                                   last_epoch=last_epoch - 1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    # DBSCAN cluster
    eps = args.eps
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    centroids = Queues(queue_max=args.queue_max, momentum=args.momentum, sample_num=args.sample_num)
    memory = QueueMemory(centroids, temp=args.temp,
                         momentum=args.momentum).cuda()
    optimizer.add_param_group({"params": memory.temp, "lr": args.lr * args.tlr, "weight_decay": args.weight_decay})

    for epoch in range(last_epoch, args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0).cpu()
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        centroids.init_queue(features, pseudo_labels)
        del cluster_loader, features

        memory.refresh(centroids)

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader), is_writer=True, out_path=args.tpath)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            SummaryWriter(osp.join(args.tpath, 'baseline')) \
                .add_scalar('mAP/cluster_contrast', mAP * 100, global_step=epoch)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  encoder mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best auto_encoder:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/multi_usl'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")

    parser.add_argument('--load-last', type=bool, default=False,
                        help="whether loaded the last training best auto_encoder")
    working_file = re.sub('.py', '', osp.basename(__file__))
    parser.add_argument('--tpath', help='the output path of tensorboard', type=str,
                        default='./tensorboard_output/{}'.format(working_file))
    parser.add_argument('--gamma', type=float, help='learning gamma', default=0.1)
    parser.add_argument('--sample_num', default=16, type=int, help='centroid sample number')
    parser.add_argument('--tlr', default=0.01, type=float, help='temp learning rate')
    parser.add_argument('--q-noise', default=0.1, type=float, help='noise weight of multi-queue')
    parser.add_argument('-qm', '--queue-max', default=16, type=int, help='the max length of multi queue')

    main()

from __future__ import print_function, absolute_import
import time

import torch

from .utils.meters import AverageMeter
from tensorboardX import SummaryWriter
import os.path as osp
import torch.nn.functional as F


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, is_writer=False, out_path=None):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        temps = AverageMeter()


        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs_q
            inputs, labels, indexes = self._parse_data(inputs)
            f_out = self._forward(inputs)
           
            loss = self.memory(f_out, labels)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           

            losses.update(loss.item())
            temps.update(self.memory.temp.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if is_writer == True and (epoch + 1) % 10 == 0:
                # print('output loss_avg')
                writer = SummaryWriter(osp.join(out_path, 'epoch_{}'.format(epoch)))
                writer.add_scalar('temp_avg', temps.avg, global_step=i)
                writer.add_scalar('losses_avg', losses.avg, global_step=i)

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Temp {:.5f} ({:.5f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              temps.val, temps.avg))
    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

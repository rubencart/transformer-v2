# -*- coding: utf-8 -*-
# date: 2018-12-02 21:29
import torch

import torch.nn as nn
from torch.autograd import Variable


class MultiGPULossCompute(object):
    """
    A multi-gpu loss compute and train function.
    """

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.criterion = criterion
        self.generator = generator
        self.criteria = [None for _ in range(len(devices) - 1)] \
                        + [nn.parallel.replicate(criterion, devices=devices)]
        self.generators = [None for _ in range(len(devices) - 1)] \
                          + [nn.parallel.replicate(generator, devices=devices)]

        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, target, normalize):
        total = 0.0
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(target, target_gpus=self.devices)

        self.replicate_modules(len(out_scatter))
        generator = self.generators[len(out_scatter) - 1]
        criterion = self.criteria[len(out_scatter) - 1]

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data, requires_grad=self.opt is not None)] for o in
                          out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)), t[:, i:i + chunk_size].contiguous().view(-1)) for g, t in
                 zip(gen, targets)]
            loss = nn.parallel.parallel_apply(criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()
            if l.shape == torch.Size([]):  # handle 1 GPU case
                total += l.item() / normalize
            else:
                l = l[0] / normalize
                total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return total * normalize

    def replicate_modules(self, var_length):
        if var_length != len(self.devices):
            idx = var_length - 1
            if not self.criteria[idx] or not self.generators[idx]:
                self.criteria[idx] = nn.parallel.replicate(self.criterion, devices=self.devices[:var_length])
                self.generators[idx] = nn.parallel.replicate(self.generator, devices=self.devices[:var_length])
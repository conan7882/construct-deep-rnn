#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

from tensorflow.examples.tutorials.mnist import input_data


class MNIST(object):
    def __init__(self, data_path, batch_size, data_type):
        if data_type == 'test':
            self.mnist = input_data.read_data_sets('data_path', one_hot=True).test
        else:
            self.mnist = input_data.read_data_sets('data_path', one_hot=True).train
        self._b_size = batch_size

    def next_batch(self):
        batch_x, batch_y = self.mnist.next_batch(self._b_size)
        batch_x = batch_x.reshape((self._b_size, 28, 28))

        return {'data': batch_x, 'label': batch_y}

    @property
    def epochs_completed(self):
        return self.mnist.epochs_completed

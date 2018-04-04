#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from lib.net.rnn import RNNClassification
from lib.dataflow.mnist import MNIST
from lib.helper.train import Train
from lib.model.deep.stack import StackedRNN
from lib.model.deep.densrnn import DenseRNN


class Config(object):
    n_class = 10

    n_step = 28
    n_feature = 28

    hidden_size_list = [10, 10]
    max_grad_norm = 5
    keep_prob = 1
    lr = 0.001
    max_epoch = 10000
    batch_size = 128

    data_dir = '/Users/gq/workspace/Dataset/MNIST_data/'
    train_data = MNIST(data_dir, batch_size, 'train')
    valid_data = MNIST(data_dir, batch_size, 'test')
    test_data = MNIST(data_dir, batch_size, 'test')

if __name__ == '__main__':
    config = Config()

    data = tf.placeholder(tf.float32,
                          [None, config.n_step, config.n_feature],
                          name='data')
    label = tf.placeholder(tf.float32,
                          [None, config.n_class],
                          name='label')
    lr = tf.placeholder(tf.float32, name='lr')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    input_plh_dict = {'data': data,
                      'label': label,
                      'lr': lr,
                      'keep_prob': keep_prob}

    model = RNNClassification(
        config.n_class,
        'lstm',
        config.hidden_size_list,
        config.max_grad_norm,
        rnn_construction=StackedRNN)
    model.create_model(input_plh_dict)

    trainer = Train(config, model, input_plh_dict)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch_cnt = 0
        while epoch_cnt < config.max_epoch:
            trainer.train_epoch(sess)
            epoch_cnt += 1


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cells.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class BasicLSTM(RNNCell):
    def __init__(self, hidden_size, forget_bias=0.0):
        print('*** LSTM ***')
        self._n_hidden = hidden_size
        self._forget_b = forget_bias

    @property
    def state_size(self):
        # return (self._n_hidden, self._n_hidden)
        return tf.contrib.rnn.LSTMStateTuple(self._n_hidden, self._n_hidden)

    @property
    def output_size(self):
        return self._n_hidden

    def __call__(self, x, state, scope='basic_lstm',
                 Wx_initializer=None,
                 Wh_initializer=None,
                 bias_initializer=None):
        # reference code: https://github.com/OlavHN/bnlstm/blob/master/lstm.py
        with tf.variable_scope('basic_lstm'):
            hidden_size = self._n_hidden
            c = state.c
            h = state.h

            size_x = x.get_shape().as_list()[-1]
            size_h = h.get_shape().as_list()[-1]
            Wx = tf.get_variable(
                        name='Wx',
                        shape=[size_x, 4 * hidden_size],
                        initializer=Wx_initializer,
                        dtype=tf.float32,
                        trainable=True)
            Wh = tf.get_variable(
                        name='Wh',
                        shape=[size_h, 4 * hidden_size],
                        initializer=Wh_initializer,
                        dtype=tf.float32,
                        trainable=True)
            biases = tf.get_variable(
                        name='biases',
                        shape=[4 * hidden_size],
                        initializer=bias_initializer,
                        dtype=tf.float32,
                        trainable=True)
            # f_biases = tf.get_variable(
            #             name='f_biases',
            #             shape=[hidden_size],
            #             initializer=tf.constant_initializer(
            #                 self._forget_b * np.ones_like([hidden_size], dtype=float)),
            #             # dtype=tf.float32,
            #             trainable=True)
            # biases = tf.concat([iog_biases, f_biases], axis=0, name='biases')

            input_concat = tf.concat([x, h], axis=-1, name='input')
            W_concat = tf.concat([Wx, Wh], axis=0, name='weights')
            iogf = tf.matmul(input_concat, W_concat) + biases
            i, o, g, f = tf.split(iogf, num_or_size_splits=4, axis=1)
            f = tf.add(f, self._forget_b * np.ones_like([hidden_size], dtype=float))

            next_c = tf.add(tf.multiply(tf.sigmoid(f), c),
                            tf.multiply(tf.sigmoid(i), tf.tanh(g)))
            next_h = tf.multiply(tf.sigmoid(o), tf.tanh(next_c))

            next_state = tf.contrib.rnn.LSTMStateTuple(c=next_c, h=next_h)
            return next_h, next_state

class BasicGRU(RNNCell):
    def __init__(self, hidden_size):
        print('*** GRU ***')
        self._n_hidden = hidden_size

    @property
    def state_size(self):
        return self._n_hidden

    @property
    def output_size(self):
        return self._n_hidden

    def __call__(self, x, state, scope='basic_gru',
                 Wx_initializer=None,
                 Wh_initializer=None,
                 bias_initializer=None):
        hidden_size = self._n_hidden
        h = state
        # hidden_size = 40
        # print(x.get_shape()[-1])

        size_x = x.get_shape().as_list()[-1]
        size_h = h.get_shape().as_list()[-1]
        Wxrz = tf.get_variable(
            name='Wxrz',
            shape=[size_x, 2 * hidden_size],
            initializer=Wx_initializer,
            dtype=tf.float32,
            trainable=True)
        Whrz = tf.get_variable(
            name='Whrz',
            shape=[size_h, 2 * hidden_size],
            initializer=Wh_initializer,
            dtype=tf.float32,
            trainable=True)
        rz_biases = tf.get_variable(
            name='rz_biases',
            shape=[2 * hidden_size],
            initializer=bias_initializer,
            dtype=tf.float32,
            trainable=True)
        input_concat = tf.concat([x, h], axis=-1)
        Wrz_concat = tf.concat([Wxrz, Whrz], axis=0, name='rz_weights')
        rz = tf.matmul(input_concat, Wrz_concat) + rz_biases
        r, z = tf.split(rz, num_or_size_splits=2, axis=1)
        r = tf.sigmoid(r)
        z = tf.sigmoid(z)

        Wxh = tf.get_variable(
            name='Wxh',
            shape=[size_x, hidden_size],
            initializer=Wx_initializer,
            dtype=tf.float32,
            trainable=True)
        Whh = tf.get_variable(
            name='Whh',
            shape=[size_h, hidden_size],
            initializer=Wh_initializer,
            dtype=tf.float32,
            trainable=True)
        h_biases = tf.get_variable(
            name='biases',
            shape=[hidden_size],
            initializer=bias_initializer,
            dtype=tf.float32,
            trainable=True)
        input_concat = tf.concat([x, tf.multiply(r, h)], axis=-1)
        Wh_concat = tf.concat([Wxh, Whh], axis=0, name='h_weights')
        next_h = tf.tanh(tf.matmul(input_concat, Wh_concat) + h_biases)
        next_h = tf.add(tf.multiply(z, h), tf.multiply((1 - z), next_h))
        return next_h, next_h

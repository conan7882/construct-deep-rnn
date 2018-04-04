#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from .cells import BasicLSTM, BasicGRU

def make_cell(hidden_size, cell_type, forget_bias=0.0,
              is_training=True, keep_prob=1):
    if cell_type == 'gru':
        cell = BasicGRU(hidden_size=hidden_size)
    else:
        cell = BasicLSTM(hidden_size=hidden_size,
                         forget_bias=forget_bias)

    if is_training is True:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32)
    return cell


def rnn_layer(inputs,
              hidden_size_list,
              forget_bias=0.0,
              init_state=None,
              is_training=True,
              keep_prob=1,
              cell_type='lstm',
              rnn_construction=tf.contrib.rnn.MultiRNNCell,
              name='rnn_layer'):
    
    def _make_cell(hidden_size):
        return make_cell(hidden_size,
                         cell_type,
                         forget_bias=forget_bias,
                         is_training=is_training,
                         keep_prob=keep_prob)

    with tf.variable_scope(name):
        try:
            cell = rnn_construction(
                [_make_cell(hidden_size)
                 for hidden_size in hidden_size_list])
        except TypeError:
            cell = rnn_construction(
                [_make_cell(hidden_size)
                 for hidden_size in hidden_size_list],
                 input_size=inputs.get_shape()[-1])

        state = init_state 
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=state, dtype=tf.float32)

        out_size = cell.output_size
    return outputs, state, out_size

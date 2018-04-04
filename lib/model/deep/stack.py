#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: stack.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference:
# https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/rnn_cell_impl.py

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class StackedRNN(RNNCell):
    def __init__(self, cells):
        print('*** StackedRNN ***')
        self._cells = cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        # with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

    def __call__(self, inputs, state):
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            with tf.variable_scope('cell_{}'.format(i)):
                cur_state = state[i]
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)

        new_states = (tuple(new_states))

        return cur_inp, new_states


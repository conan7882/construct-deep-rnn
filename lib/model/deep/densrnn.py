#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: densrnn.py
# Author: Qian Ge <geqian1001@gmail.com>
# Improving Language Modeling using Densely Connected Recurrent Neural Networks

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class DenseRNN(RNNCell):
    def __init__(self, cells, input_size):
        self._n_layer = len(cells)
        self._cells = cells
        self._n_hidden = self._cells[-1].output_size

        self._outsize = input_size
        for cell in self._cells:
            self._outsize += cell.output_size
        
    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._outsize

    def __call__(self, inputs, state, scope='DenseRNN'):
        with tf.variable_scope(scope):
            cell_inputs = inputs
            state_output_list = []
            for cell_id in range(0, self._n_layer):
                with tf.variable_scope('cell_{}'.format(cell_id)):
                    cell_state = state[cell_id]
                    cell_output, new_state = self._cells[cell_id](
                        cell_inputs, cell_state)
                    cell_inputs = tf.concat([cell_inputs, cell_output], axis=-1)
                    state_output_list.append(new_state)
            state_output_list = (tuple(state_output_list))
        return cell_inputs, state_output_list

    def zero_state(self, batch_size, dtype):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

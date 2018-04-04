#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rnn.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.base import BaseModel
import tensorcv.models.layers as L

from ..model.layers import rnn_layer


class RNNClassification(BaseModel):
    def __init__(self, n_class,
                 cell_type,
                 hidden_size_list,
                 max_grad_norm,
                 rnn_construction=tf.contrib.rnn.MultiRNNCell):
        self._n_class = n_class
        self._cell_type = cell_type
        if not isinstance(hidden_size_list, list):
            hidden_size_list = [hidden_size_list]
        self._hidden_size_list = hidden_size_list
        self._max_grad_norm = max_grad_norm
        self._construction = rnn_construction

        self.set_is_training(True)
        self.layer = {}

    def create_model(self, input_dict):
        self._input_dict = input_dict
        self._create_model()

    def _create_model(self):
        with tf.name_scope('input'):
            inputs = self._input_dict['data']
            keep_prob = self._input_dict['keep_prob']

        outputs, state, out_size = rnn_layer(
              inputs,
              self._hidden_size_list,
              forget_bias=1.0,
              init_state=None,
              is_training=self.is_training,
              keep_prob=keep_prob,
              cell_type=self._cell_type,
              rnn_construction=self._construction,
              name='rnn_layer')

        outputs = outputs[:, -1, :]
        outputs = tf.reshape(outputs, [-1, out_size])
        logits = L.fc(outputs, self._n_class, name='softmax')
        prediction = tf.nn.softmax(logits)

        self.layer['logits'] = logits
        self.layer['prediction'] = prediction

    def _get_loss(self):
        label = self._input_dict['label']
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.layer['logits'], labels=label))
        return self.loss

    def _get_optimizer(self):
        lr = self._input_dict['lr']
        return tf.train.RMSPropOptimizer(learning_rate=lr)

    def get_train_op(self):
        with tf.name_scope('train_op'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self._get_loss(), tvars),
                self._max_grad_norm)
            grads = zip(grads, tvars)

            opt = self._get_optimizer()
            train_op = opt.apply_gradients(grads)
        return train_op

    def get_accuracy(self):
        label = tf.argmax(self._input_dict['label'], axis=1)
        pred = tf.argmax(self.layer['prediction'], axis=1)
        correct_pred = tf.equal(label, pred)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


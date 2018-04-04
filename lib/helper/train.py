#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Qian Ge <geqian1001@gmail.com>

class Train(object):
    def __init__(self, config, model, input_plh_dict):
        self.global_step = 0

        self._config = config
        self._model = model

        self._train_data = config.train_data
        self._valid_data = config.valid_data

        self.get_plh(input_plh_dict)
        self._setup_op()

    def get_plh(self, input_plh_dict):
        for key in input_plh_dict:
            setattr(self, '_{}_plh'.format(key), input_plh_dict[key])

    def _setup_op(self):
        self._train_op = self._model.get_train_op()
        self._accuracy_op = self._model.get_accuracy()

    def train_epoch(self, sess):
        dataflow = self._config.train_data
        print('Start epoch: {}'.format(dataflow.epochs_completed + 1))
        self._model.set_is_training(True)

        cur_epoch = dataflow.epochs_completed

        step = 0
        acc_sum = 0
        while True:
            batch_data = dataflow.next_batch()
            if dataflow.epochs_completed > cur_epoch:
                break
            step += 1
            self.global_step += 1

            _, acc = sess.run(
                [self._train_op, self._accuracy_op],
                feed_dict={self._data_plh: batch_data['data'],
                           self._label_plh: batch_data['label'],
                           self._keep_prob_plh: self._config.keep_prob,
                           self._lr_plh: self._config.lr})
            acc_sum += acc

            if step % 100 == 0:
                print('epoch: {}, step: {}, accuarcy: {:0.2f}'.
                      format(cur_epoch, step, acc_sum / step))

        print('Finish epoch: {}, accuarcy: {:0.2f}'.
              format(dataflow.epochs_completed, acc_sum / step))

# -*- coding: utf-8 -*-
########################################################
# Copyright (c) 2019, Baidu Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
########################################################
"""
This module to train a so labeling model
"""

import json
import os
import time
import sys
import argparse
import ConfigParser

import paddle
import paddle.fluid as fluid
import six

import spo_data_reader
import spo_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib



def train(conf_dict, data_reader, use_cuda=False):
    """
    Training of so labeling model
    """
    # input data layer
    word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)
    postag = fluid.layers.data(
        name='token_pos', shape=[1], dtype='int64', lod_level=1)
    p_word = fluid.layers.data(
        name='p_word', shape=[1], dtype='int64', lod_level=1)
    # label
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)

    # embedding + lstm
    feature_out = spo_model.db_lstm(data_reader, word, \
            postag, p_word, conf_dict)

    # loss function
    # crf layer
    mix_hidden_lr = float(conf_dict['mix_hidden_lr'])
    crf_cost = fluid.layers.linear_chain_crf(
        input=feature_out,
        label=target,
        param_attr=fluid.ParamAttr(name='crfw', learning_rate=mix_hidden_lr))
    avg_cost = fluid.layers.mean(crf_cost)

    # optimizer
    sgd_optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=2e-3, )

    sgd_optimizer.minimize(avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    train_batch_reader = paddle.batch(
        paddle.reader.shuffle(data_reader.get_train_reader(), buf_size=8192),
        batch_size=conf_dict['batch_size'])

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    feeder = fluid.DataFeeder(feed_list=[word, postag, p_word, target], place=place)
    exe = fluid.Executor(place)

    save_dirname = conf_dict['spo_model_save_dir']

    def train_loop(main_program, trainer_id=0):
        """start train loop"""
        exe.run(fluid.default_startup_program())

        start_time = time.time()
        batch_id = 0
        for pass_id in six.moves.xrange(conf_dict['pass_num']):
            pass_start_time = time.time()
            cost_sum, cost_counter = 0, 0
            for data in train_batch_reader():
                cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                cost = cost[0]
                cost_sum += cost
                cost_counter += 1
                if batch_id % 10 == 0 and batch_id != 0:
                    print >> sys.stderr, "batch %d finished, second per batch: %02f" % (
                        batch_id, (time.time() - start_time) / batch_id)

                # cost expected, training over
                if float(cost) < 1:
                    save_path = os.path.join(save_dirname, 'final')
                    fluid.io.save_inference_model(save_path, ['word_data', 'token_dist', 'p_word'],
                                                  [feature_out], exe, params_filename='params')
                    return
                batch_id = batch_id + 1

            # save the model once each pass ends
            pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
            print >> sys.stderr, "%d pass end, cost time: %02f, avg_cost: %f" % (
                    pass_id, time.time() - pass_start_time, pass_avg_cost)
            save_path = os.path.join(save_dirname, 'pass_%04d-%f' %
                                    (pass_id, pass_avg_cost))
            fluid.io.save_inference_model(save_path, ['word_data', 'token_pos', 'p_word'],
                                          [feature_out], exe, params_filename='params')

        else:
            # pass times complete and the training is over
            save_path = os.path.join(save_dirname, 'final')
            fluid.io.save_inference_model(save_path, ['word_data', 'token_pos', 'p_word'],
                                          [feature_out], exe, params_filename='params')
        return

    train_loop(fluid.default_main_program())


def main(conf_dict, use_cuda=False):
    """Train main function"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    data_generator = spo_data_reader.DataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['so_label_dict_path'],
        p_eng_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['spo_train_data_path'],
        test_data_list_path=conf_dict['spo_test_data_path'])
    
    train(conf_dict, data_generator, use_cuda=use_cuda)


if __name__ == '__main__':
    # Load the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str,
        help="conf_file_path_for_model. (default: %(default)s)",
        required=True)
    args = parser.parse_args()
    conf_dict = conf_lib.load_conf(args.conf_path)
    use_gpu = True if conf_dict.get('use_gpu', 'False') == 'True' else False
    main(conf_dict, use_cuda=use_gpu)

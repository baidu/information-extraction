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
This module to infer with a p classification model
"""

import json
import os
import sys
import argparse
import ConfigParser
import math

import numpy as np
import paddle
import paddle.fluid as fluid

import p_data_reader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib


def predict_infer(conf_dict, data_reader, predict_data_path, \
        predict_result_path, model_path):
    """
    Predict with trained models 
    """
    if len(predict_result_path) > 0:
        result_writer = open(predict_result_path, 'w')
    else:
        result_writer = sys.stdout

    np.set_printoptions(precision=3)
    if len(model_path) == 0:
        return

    place = fluid.CPUPlace()
    word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)
    postag = fluid.layers.data(
        name='token_pos', shape=[1], dtype='int64', lod_level=1)
    feeder = fluid.DataFeeder(feed_list=[word, postag], place=place)
    exe = fluid.Executor(place)

    test_batch_reader = paddle.batch(
        paddle.reader.buffered(data_reader.get_predict_reader\
                (predict_data_path, need_input=True, need_label=False),
                size=8192),
        batch_size=conf_dict["batch_size"])
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(
                model_path, exe, params_filename='params')

        # batch
        batch_id = 0
        for data in test_batch_reader():
            feeder_data = []
            input_data = []
            for item in data:
                input_dic = json.loads(item[0])
                input_data.append(input_dic)
                feeder_data.append(item[1:])
            results = exe.run(inference_program, feed=feeder.feed(feeder_data),
                              fetch_list=fetch_targets, return_numpy=False)
            label_scores = np.array(results[0]).tolist()
            #infer a batch
            infer_a_batch(label_scores, input_data, result_writer, data_reader)
            
            batch_id += 1


def infer_a_batch(label_scores, input_data, result_writer, data_reader):
    """Infer the results of a batch"""
    for sent_idx, label in enumerate(label_scores):
        p_label = []
        label = map(float, label)
        for p_idx, p_score in enumerate(label):
            if sigmoid(p_score) > 0.5:
                p_label.append(data_reader.get_label_output(p_idx))
        for p in p_label:
            output_fields = [json.dumps(input_data[sent_idx], ensure_ascii=False), p]
            result_writer.write('\t'.join(output_fields).encode('utf-8'))
            result_writer.write('\n')


def sigmoid(x):
    """sigmode function"""
    return math.exp(x) / (1 + math.exp(x))


def main(conf_dict, model_path, predict_data_path, 
            predict_result_path, use_cuda=False):
    """Predict main function"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    data_generator = p_data_reader.RcDataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['train_data_path'],
        test_data_list_path=conf_dict['test_data_path'])
    
    predict_infer(conf_dict, data_generator, predict_data_path, \
            predict_result_path, model_path)


if __name__ == '__main__':
    # Load configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str,
            help="conf_file_path_for_model. (default: %(default)s)",
            required=True)
    parser.add_argument("--model_path", type=str,
            help="model_path", required=True)
    parser.add_argument("--predict_file", type=str,
            help="the_file_to_be_predicted", required=True)
    parser.add_argument("--result_file", type=str,
            default='', help="the_file_of_predicted_results")
    args = parser.parse_args()
    conf_dict = conf_lib.load_conf(args.conf_path)
    model_path = args.model_path
    predict_data_path = args.predict_file
    predict_result_path = args.result_file
    for input_path in [model_path, predict_data_path]:
        if not os.path.exists(input_path):
            raise ValueError("%s not found." % (input_path))
    main(conf_dict, model_path, predict_data_path, predict_result_path)

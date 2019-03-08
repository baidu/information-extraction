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
This module to infer with a trained so model
"""

import json
import os
import sys
import argparse
import ConfigParser
import codecs

import numpy as np
import paddle
import paddle.fluid as fluid

import spo_data_reader

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
    p_word = fluid.layers.data(
        name='p_word', shape=[1], dtype='int64', lod_level=1)
    feeder = fluid.DataFeeder(feed_list=[word, postag, p_word], place=place)
    exe = fluid.Executor(place)

    test_batch_reader = paddle.batch(
        paddle.reader.buffered(data_reader.get_predict_reader\
            (predict_data_path, need_input=True, need_label=False),
            size=8192),
        batch_size=conf_dict['batch_size'])
    inference_scope = fluid.core.Scope()
    text_spo_dic = {}  #final triples
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
                feeder_data.append(item[1:])
                input_data.append(item[0])
            results = exe.run(inference_program, feed=feeder.feed(feeder_data),
                              fetch_list=fetch_targets, return_numpy=False)
            tag_split_idx = results[0].lod()[0]
            label_tag_scores = np.array(results[0])
            # sentence
            print >> sys.stderr, 'batch_id=', batch_id
            for sent_idx, tag_idx in enumerate(tag_split_idx[:-1]):
                input_sent = input_data[sent_idx].split('\t')[0]
                input_p = input_data[sent_idx].split('\t')[1]
                tag_scores = label_tag_scores[tag_idx: tag_split_idx[sent_idx + 1]]
                # token
                tag_list = []
                for token_idx, token_tags in enumerate(tag_scores):
                    tag = data_reader.get_label_output(token_tags)
                    tag_list.append(tag)
                predicted_s_list, predicted_o_list = refine_predict_seq(input_sent, tag_list) 
                tag_list_str = json.dumps(tag_list, ensure_ascii=False)
                if len(predicted_s_list) == 0 or len(predicted_o_list) == 0:
                    continue
                else:
                    text = json.loads(input_sent)["text"]
                    predicted_s_list = list(set(predicted_s_list))
                    predicted_o_list = list(set(predicted_o_list))
                    for predicted_s in predicted_s_list:
                        for predicted_o in predicted_o_list:
                            if text not in text_spo_dic:
                                text_spo_dic[text] = set()
                            text_spo_dic[text].add((predicted_s, input_p, predicted_o))
                    
            batch_id += 1
    output(text_spo_dic, result_writer)


def refine_predict_seq(input_sent, tag_list):
    """
    Generate s-o list based on the annotation results 
    predicted by the model
    """
    sent_info = json.loads(input_sent)
    word_seq = [item["word"] for item in sent_info["postag"]]
    s_list, o_list= [], []
    token_idx = 0
    while token_idx < len(tag_list):
        if tag_list[token_idx] == 'O':
            token_idx += 1
        elif tag_list[token_idx].endswith('SUB') and tag_list[token_idx].startswith('B'):
            cur_s = word_seq[token_idx]
            token_idx += 1
            while token_idx < len(tag_list) and tag_list[token_idx].endswith('SUB'):
                cur_s += word_seq[token_idx]
                token_idx += 1
            s_list.append(cur_s)
        elif tag_list[token_idx].endswith('OBJ') and tag_list[token_idx].startswith('B'):
            cur_o = word_seq[token_idx]
            token_idx += 1
            while token_idx < len(tag_list) and tag_list[token_idx].endswith('OBJ'):
                cur_o += word_seq[token_idx]
                token_idx += 1
            o_list.append(cur_o)
        else:
            token_idx += 1
    return s_list, o_list


def get_schemas(schema_file):
    """"Read the original schema file"""
    schema_dic = {}
    with codecs.open(schema_file, 'r', 'utf-8') as fr:
        for line in fr:
            dic = json.loads(line.strip())
            predicate = dic["predicate"]
            subject_type = dic["subject_type"]
            object_type = dic["object_type"]
            schema_dic[predicate] = (subject_type, object_type)
    return schema_dic


def output(text_spo_dic, result_writer):
    """
    Output final SPO triples
    """
    schema_dic = {}
    schema_dic = get_schemas('./data/all_50_schemas')
    for text in text_spo_dic:
        text_dic = {"text": text}
        text_dic["spo_list"] = []
        for spo in text_spo_dic[text]:
            dic = {"subject": spo[0], "predicate": spo[1], \
                    "object": spo[2], "subject_type": schema_dic[spo[1]][0], \
                    "object_type": schema_dic[spo[1]][1]}
            text_dic["spo_list"].append(dic)
        result_writer.write(json.dumps(text_dic, ensure_ascii=False).encode('utf-8'))
        result_writer.write('\n')


def main(conf_dict, model_path, predict_data_path, predict_result_path, \
        use_cuda=False):
    """Predict main function"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    data_generator = spo_data_reader.DataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['so_label_dict_path'],
        p_eng_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['spo_train_data_path'],
        test_data_list_path=conf_dict['spo_test_data_path'])
    
    predict_infer(conf_dict, data_generator, predict_data_path, \
            predict_result_path, model_path)


if __name__ == '__main__':
    # Load the configuration file
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

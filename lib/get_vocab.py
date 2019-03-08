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
This module to generate vocabulary list
"""

import random
import os
import codecs
import sys
import json
reload(sys)
sys.setdefaultencoding('utf-8')

def load_word_file(f_input):
    """
    Get all words in files
    :param string: input file
    """
    file_words = {}
    with codecs.open(f_input, 'r', 'utf-8') as fr:
        words = []
        for line in fr:
            try:
                dic = json.loads(line.decode('utf-8').strip())
                postag = dic['postag']
                words = [item["word"].strip() for item in postag]
            except:
                continue
            for word in words:
                file_words[word] = file_words.get(word, 0) + 1
    return file_words


def get_vocab(train_file, dev_file):
    """
    Get vocabulary file from the field 'postag' of files
    :param string: input train data file
    :param string: input dev data file
    """
    word_dic = load_word_file(train_file)
    if len(word_dic) == 0:
        raise ValueError('The length of train word is 0')
    dev_word_dic = load_word_file(dev_file)
    if len(dev_word_dic) == 0:
        raise ValueError('The length of dev word is 0')
    for word in dev_word_dic:
        if word in word_dic:
            word_dic[word] += dev_word_dic[word]
        else:
            word_dic[word] = dev_word_dic[word]
    print '<UNK>'
    vocab_set = set()
    value_list = sorted(word_dic.iteritems(), key=lambda d:d[1], reverse=True)
    for word in value_list[:30000]:
        print word[0]
        vocab_set.add(word[0])

    #add predicate in all_50_schemas
    if not os.path.exists('./data/all_50_schemas'):
        raise ValueError("./data/all_50_schemas not found.")
    with codecs.open('./data/all_50_schemas', 'r', 'utf-8') as fr:
        for line in fr:
            dic = json.loads(line.decode('utf-8').strip())
            p = dic['predicate']
            if p not in vocab_set:
                vocab_set.add(p)
                print p

    
if __name__ == '__main__':
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    get_vocab(train_file, dev_file)

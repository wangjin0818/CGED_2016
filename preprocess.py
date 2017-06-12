from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import codecs
import random

import numpy as np
import xml.dom.minidom

import jieba
# jieba.set_dictionary(os.path.join('dict', 'dict.txt.big'))

from collections import defaultdict

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

def serialize(file_name):
    logging.info('Loading data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')

    ret_id, ret_text, ret_label = [], [], []
    for doc in docs:
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
        text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

        correction = doc.getElementsByTagName('CORRECTION')[0].childNodes[0].nodeValue.replace('\n', '')
        errs = doc.getElementsByTagName('ERROR')

        locat_dict = {}
        missing_len = 0
        text_len = len(text)

        inc = -1
        err_array = []
        for err in errs:
            start_off = err.getAttribute('start_off')
            end_off = err.getAttribute('end_off')
            err_type = err.getAttribute('type')

            for i in range(int(start_off), int(end_off)+1):
                locat_dict[i+inc] = error_dict[err_type]

                if err_type == 'M':
                    text = text[:i+inc] + 'M' + text[i+inc:]
                    inc = inc + 1
                    # missing_len = missing_len + 1

        text_array = []
        label_array = []
        for i in range(text_len + missing_len):
            if locat_dict.has_key(i):
                text_array.append(text[i])
                label_array.append(locat_dict[i])
            else:
                text_array.append(text[i])
                label_array.append(0)

        # print(text_id)
        # print(text_array)
        # print(label_array)

        ret_id.append(text_id)
        ret_text.append(text_array)
        ret_label.append(label_array)

    return ret_id, ret_text, ret_label

def position_serialize(file_name, bucket_size=5, position='right'):
    with codecs.open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')

    ret_text, ret_label = [], []
    for doc in docs:
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
        text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

        errs = doc.getElementsByTagName('ERROR')

        locat_dict = {}
        missing_len = 0
        text_len = len(text)

        inc = -1
        err_array = []
        for err in errs:
            start_off = err.getAttribute('start_off')
            end_off = err.getAttribute('end_off')
            err_type = err.getAttribute('type')

            for i in range(int(start_off), int(end_off)+1):
                locat_dict[i+inc] = error_dict[err_type]

                if err_type == 'M':
                    text = text[:i+inc] + 'M' + text[i+inc:]
                    inc = inc + 1
                    # missing_len = missing_len + 1

        text_array = []
        label_array = []
        for i in range(text_len + missing_len):
            if locat_dict.has_key(i):
                text_array.append(text[i])
                label_array.append(locat_dict[i])
            else:
                text_array.append(text[i])
                label_array.append(0)
        
        for i in range(len(label_array)):
            line_text = []
            line_label = 0 

            if position == 'center':
                # if the left side are not enough
                if i <= (bucket_size - 1) / 2:
                    continue

                elif i >= len(label_array) - ((bucket_size - 1) / 2):
                    continue
                    
                else:
                    line_text = text_array[i - ((bucket_size - 1) / 2): i + 1 + ((bucket_size - 1) / 2)]

                if label_array[i] == 0:
                    if random.uniform(0, 1) <= 0.15:
                        ret_text.append(line_text)
                        ret_label.append(label_array[i])
                else:
                    ret_text.append(line_text)
                    ret_label.append(label_array[i])

            if position == 'right':
                # if the left side are not enough
                if i <= bucket_size:
                    continue
                else:
                    line_text = text_array[i - bucket_size:i]

                if label_array[i] == 0:
                    if random.uniform(0, 1) <= 0.15:
                        ret_text.append(line_text)
                        ret_label.append(label_array[i])
                else:
                    ret_text.append(line_text)
                    ret_label.append(label_array[i])


    return ret_text, ret_label

def detect_serialize(file_name):
    logging.info('Loading data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')

    ret_id, ret_text, ret_label = [], [], []
    for doc in docs:
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
        text = list(jieba.cut(text, cut_all=False))
        ret_text.append(text)
        ret_label.append(0)

        text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

        correction = doc.getElementsByTagName('CORRECTION')[0].childNodes[0].nodeValue.replace('\n', '')
        correction = list(jieba.cut(correction, cut_all=False))
        ret_text.append(correction)
        ret_label.append(1)

        errs = doc.getElementsByTagName('ERROR')

    return ret_id, ret_text, ret_label


def identification_serialize(file_name): 
    logging.info('Loading data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')

    ret_id, ret_text = [], []
    R_label, M_label, S_label, W_label = [], [], [], []
    for doc in docs:
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
        text = list(jieba.cut(text, cut_all=False))
        ret_text.append(text)
        
        text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')


        errs = doc.getElementsByTagName('ERROR')
        cur_err_dict = defaultdict(int)
        for err in errs:
            err_type = err.getAttribute('type')
            cur_err_dict[err_type] = cur_err_dict[err_type] + 1

        if cur_err_dict['R'] >= 1:
            R_label.append(1)
        else:
            R_label.append(0)

        if cur_err_dict['M'] >= 1:
            M_label.append(1)
        else:
            M_label.append(0)

        if cur_err_dict['S'] >= 1:
            S_label.append(1)
        else:
            S_label.append(0)

        if cur_err_dict['W'] >= 1:
            W_label.append(1)
        else:
            W_label.append(0)

    return ret_id, ret_text, R_label, M_label, S_label, W_label

if __name__ == '__main__':
    data = os.path.join('data', 'CGED16_TOCFL_Train_All.txt.bak')
    # ret_text, ret_label = position_serialize(data, position='right')
    # print(len(ret_text))
    # print(len(ret_label))
    # print(ret_label.count(0))
    # print(ret_label.count(1))
    # print(ret_label.count(2))
    # print(ret_label.count(3))
    # print(ret_label.count(4))

    # ret_id, ret_text, R_label, M_label, S_label, W_label = identification_serialize(data)
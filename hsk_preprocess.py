from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import codecs
import random
import cPickle

import numpy as np
import xml.dom.minidom

import jieba
# jieba.set_dictionary(os.path.join('dict', 'dict.txt.big'))

from collections import defaultdict

prng = np.random.RandomState(1224)

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

def hsk_detect_train_serialize(file_name):
    logging.info('Loading train data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')

    ret_id, ret_text, ret_label = [], [], []
    for doc in docs:
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
        text = list(jieba.cut(text, cut_all=False))
        ret_text.append(text)
        ret_label.append(1)

        # text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

        correction = doc.getElementsByTagName('CORRECTION')[0].childNodes[0].nodeValue.replace('\n', '')
        correction = list(jieba.cut(correction, cut_all=False))
        ret_text.append(correction)
        ret_label.append(0)

        errs = doc.getElementsByTagName('ERROR')

    return ret_text, ret_label

def hsk_detect_test_serialize(file_name):
    logging.info('Loading test data from %s' % (file_name))

    test_sid = []; test_text = []
    with codecs.open(file_name, 'r') as my_file:
        for line in my_file.readlines():
            line = line.strip().split('\t')
            
            sid = line[0].replace('(sid=', '').replace(')', '')
            text = list(jieba.cut(line[1], cut_all=False))

            test_sid.append(sid)
            test_text.append(text)

    return test_sid, test_text

def hsk_identification_train_serialize(file_name):
    logging.info('Loading train data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')

    ret_text = []
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

    return ret_text, R_label, M_label, S_label, W_label

def hsk_identification_test_serialize(file_name):
    logging.info('Loading test data from %s' % (file_name))

    test_sid = []; test_text = []
    with codecs.open(file_name, 'r') as my_file:
        for line in my_file.readlines():
            line = line.strip().split('\t')
            
            sid = line[0].replace('(sid=', '').replace(')', '')
            text = list(jieba.cut(line[1], cut_all=False))

            test_sid.append(sid)
            test_text.append(text)

    return test_sid, test_text

def hsk_position_train_serialize(file_name, bucket_size=7, position='right'):
    logging.info('Loading train data from %s, bucket_size: %d, position mode: %s' % (file_name, bucket_size, position))

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
        
        line_text = ['0'] * (bucket_size - 1)
        line_text.extend(text_array)
        for i in range(bucket_size, len(label_array)):
            if i < bucket_size:
                continue

            if label_array[i] == 0:
                if random.uniform(0, 1) <= 0.05:
                    ret_text.append(line_text[(i-bucket_size):i])
                    ret_label.append(label_array[i])
            else:
                ret_text.append(line_text[(i-bucket_size):i])
                ret_label.append(label_array[i])

    return ret_text, ret_label

def hsk_position_train_sample(file_name, bucket_size=7, position='right'):
    logging.info('Loading train data from %s, bucket_size: %d, position mode: %s' % (file_name, bucket_size, position))

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

        line_text = ['0'] * (bucket_size - 1)
        line_text.extend(text_array)
        for i in range(bucket_size, len(label_array)):
            if i < bucket_size:
                continue

            if label_array[i] == 0:
                if random.uniform(0, 1) <= 0.07:
                    ret_text.append(line_text[(i-bucket_size):i])
                    ret_label.append(label_array[i])
            else:
                ret_text.append(line_text[(i-bucket_size):i])
                ret_label.append(label_array[i])

    pickle_file = os.path.join('pickle', 'hsk_position_train.pickle')
    cPickle.dump((ret_text, ret_label), open(pickle_file, 'wb'))
    logging.info('dataset created!')

def hsk_position_test_serialize(file_name, bucket_size=7, position='right'):
    logging.info('Loading test data from %s, bucket_size: %d, position mode: %s' % (file_name, bucket_size, position))

    ret_position = []; ret_text = []
    # with codecs.open(file_name, 'r') as my_file:
    my_file = codecs.open(file_name, 'r', 'utf-8')
    # my_file = open(file_name, 'r')
    for line in my_file.readlines():
        line = line.strip().split('\t')

        sid = line[0].replace('(sid=', '').replace(')', '')

        text_line = ['0'] * (bucket_size - 1)
        text_line.extend(line[1])
        # print(text_line)
        for i in range(bucket_size, len(text_line)+1):
            # print(text_line[(i-bucket_size):i])
            ret_position.append((sid, i-bucket_size))
            ret_text.append(text_line[(i-bucket_size):i])

    return ret_position, ret_text

def hsk_final_serialize(file_name):
    logging.info('Loading test data from %s' % (file_name))

    test_sid = []; test_length = []
    with codecs.open(file_name, 'r', 'utf-8') as my_file:
        for line in my_file.readlines():
            line = line.strip().split('\t')
            sid = line[0].replace('(sid=', '').replace(')', '')
            length = len(line[1])

            test_sid.append(sid)
            test_length.append(length)

    return test_sid, test_length


if __name__ == '__main__':
    # train_file = os.path.join('data', 'CGED16_HSK_Train_All_Revised.txt')
    # train_text, R_label, M_label, S_label, W_label = hsk_identification_train_serialize(train_file)
    # print(len(train_text), len(R_label))

    # test_file = os.path.join('data', 'CGED16_HSK_Test_Input.txt')
    # test_sid, test_text = hsk_identification_test_serialize(test_file)
    # print(len(test_sid), len(test_text))

    bucket_size = 10

    train_file = os.path.join('data', 'CGED16_HSK_Train_All_Revised.txt')
    hsk_position_train_sample(train_file, bucket_size=bucket_size, position='right')

    test_file = os.path.join('data', 'CGED16_HSK_Test_Input.txt')
    test_position, test_text = hsk_position_test_serialize(test_file, bucket_size=bucket_size, position='right')
    print(len(test_position), len(test_text))
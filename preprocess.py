from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import codecs

import numpy as np
import xml.dom.minidom

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


if __name__ == '__main__':
    data = os.path.join('data', 'CGED16_HSK_Train_All.txt')
    ret_id, ret_text, ret_label = serialize(data)
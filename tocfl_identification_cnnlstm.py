from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import math

import cPickle
import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Adadelta
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

pickle_file = os.path.join('pickle', 'tocfl_identification_train.pickle')

prng = np.random.RandomState(1224)

batch_size = 50
maxlen = 60

hidden_dim = 120

kernel_size = 3
nb_filter = 120
nb_epoch = 10

option = 'S_label'

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    return x

def make_idx_data(revs, word_idx_map):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, y_train, y_sid = [], [], [], []
    i = 0; j =0
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        if rev['split'] == 1:
            y = rev[option]

            if option == 'R_label':
                if not y == 1:
                    rand = prng.rand(1)[0]
                    if rand >= 0.6:
                        continue
                else:
                    i = i + 1

            if option == 'W_label':
                if not y == 1:
                    rand = prng.rand(1)[0]
                    if rand >= 0.18:
                        continue
                else:
                    i = i + 1

            if option == 'S_label':
                if y == 1:
                    rand = prng.rand(1)[0]
                    if rand >= 0.5:
                        continue
                    else:
                        i = i + 1

            if option == 'M_label' and y == 1:
                i = i + 1

            X_train.append(sent)
            y_train.append(y)

        elif rev['split'] == 0:
            X_test.append(sent)
            y_sid.append(rev['sid'])

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)

    logging.info('target class number: %d' % (i))

    return (X_train, X_test, y_train, y_sid)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    revs, W, word_idx_map, vocab = cPickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, y_train, y_sid = make_idx_data(revs, word_idx_map)
    logging.info('X_train shape:' + str(X_train.shape))
    logging.info('X_test shape:' + str(X_test.shape))
    logging.info('sid shape:' + str(len(y_sid)))

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("max features of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension num of word vector [num_features]: %d" % num_features)

    # Keras Model
    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=kernel_size,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1
                            ))
    model.add(MaxPooling1D(pool_length=2))

    # lstm layer:
    model.add(LSTM(hidden_dim))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)

    y_pred = model.predict(X_test, batch_size=batch_size).flatten()
    for i in range(len(y_pred)):
        if y_pred[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    pred_dict = {}
    # print(len(y_sid))
    for i in range(len(y_sid)):
        if pred_dict.has_key(y_sid[i]):
            pred_dict[y_sid[i]] = pred_dict[y_sid[i]] + y_pred[i]
        else:
            pred_dict[y_sid[i]] = y_pred[i]

    positive_label = 0
    negative_label = 0
    detect_save_file = os.path.join('result', 'track2', ('identification_%s.txt' % (option)))
    with open(detect_save_file, 'wb') as my_file:
        for key in pred_dict.keys():
            if pred_dict[key] >= 1:
                my_file.write('%s\t%d\n' % (key, 1))
                positive_label = positive_label + 1
            else:
                my_file.write('%s\t%d\n' % (key, 0))
                negative_label = negative_label + 1

    logging.info('result saved!')
    logging.info('positive_label: %d' % (positive_label))
    logging.info('negative_label: %d' % (negative_label))
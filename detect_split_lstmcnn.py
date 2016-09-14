from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import math

import cPickle
import numpy as np

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

pickle_file = os.path.join('pickle', 'detect_HSK_split.pickle')

batch_size = 10
maxlen = 100

hidden_dim = 120

kernel_size = 3
nb_filter = 120
nb_epoch = 10

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
    X_train, X_test, y_train, y_test = [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['label']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_test.append(sent)
            y_test.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)

    return [X_train, X_test, y_train, y_test]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    revs, W, word_idx_map, vocab = cPickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, y_train, y_test = make_idx_data(revs, word_idx_map)
    logging.info('X_train shape:' + str(X_train.shape))
    logging.info('X_test shape:' + str(X_test.shape))

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
    model.add(LSTM(hidden_dim, return_sequences=True))

    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=kernel_size,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1
                            ))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(70))    # best: 120
    model.add(Dropout(0.25))    # best: 0.25
    model.add(Activation('relu'))

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

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)

    logging.info('precision score: %.3f' % (precision))
    logging.info('recal score: %.3f' % (recall))
    logging.info('accuracy score: %.3f' % (accuracy))
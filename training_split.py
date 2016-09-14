from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging

import numpy as np

from collections import defaultdict
from preprocess import serialize, error_dict
from sklearn.cross_validation import train_test_split

from keras.layers.embeddings import Embedding
from keras.models import Sequential

from seq2seq.models import SimpleSeq2seq
from keras.layers.core import Dropout
# from keras.layers import Input
# from keras.layers.recurrent import LSTM
# from keras.layers.core import Dense, Dropout
# from keras.layers.wrappers import TimeDistributed
# from keras.models import Model

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

bucket_size = 10
random_state = 0
embedding_dim = 200

batch_size = 20
nb_epoch = 3
hidden_dim = 100

def encode(ret_text, ret_label):
    vocab = defaultdict(int)

    for i in range(len(ret_text)):
        for j in range(len(ret_text[i])):
            vocab[ret_text[i][j]] = vocab[ret_text[i][j]] + 1

    # for debug
    # ret_text = ret_text[:200]
    # ret_label = ret_label[:200]

    # train, test split
    train_text, test_text, train_label, test_label = train_test_split(ret_text, ret_label, 
        random_state=random_state, test_size=0.1)
    # print(len(train_text), len(test_text))
    logging.info('training samples: %d' % len(train_text))
    logging.info('testing samples: %d' % len(test_text))

    word_idx_map = dict()
    i = 1
    for word in vocab.keys():
        word_idx_map[word] = i
        i = i + 1

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(train_text)):
        cur_line = []
        cur_label = []
        move_point = 1

        for j in range(len(train_text[i])):
            cur_line.append(word_idx_map[train_text[i][j]])
            cur_label.append(train_label[i][j])

            move_point = move_point + 1
            if move_point > bucket_size:
                X_train.append(cur_line)
                y_train.append(cur_label)

                cur_line = []
                cur_label = []
                move_point = 1

        if len(cur_line) > 0:
            for k in range(len(cur_line), 10):
                cur_line.append(0)
                cur_label.append(0)

            X_train.append(cur_line)
            y_train.append(cur_label)

    # y_train = np.array(y_train)
    new_y_train = np.zeros((len(y_train), bucket_size, len(error_dict.keys())+1))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            new_y_train[i, j, y_train[i][j]] = 1
            # print(new_y_train[i][j])

    X_train = np.array(X_train)

    for i in range(len(test_text)):
        cur_line = []
        cur_label = []
        move_point = 1

        for j in range(len(test_text[i])):
            cur_line.append(word_idx_map[test_text[i][j]])
            cur_label.append(test_label[i][j])

            move_point = move_point + 1
            if move_point > bucket_size:
                X_test.append(cur_line)
                y_test.append(cur_label)

                cur_line = []
                cur_label = []
                move_point = 1

        if len(cur_line) > 0:
            for k in range(len(cur_line), 10):
                cur_line.append(0)
                cur_label.append(0)

            X_test.append(cur_line)
            y_test.append(cur_label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, new_y_train, X_test, y_test, vocab

def decode(X_test, y_test, predict_array):
    y_pred = []
    y_true = []
    for i in range(len(X_test)):
        for j in range(len(X_test[i])):

            if X_test[i][j] != 0:
                y_true.append(y_test[i][j])

                print(predict_array[i][j], np.argmax(predict_array[i][j]))
                cur_pred = np.argmax(predict_array[i][j])
                y_pred.append(cur_pred)

    print(y_true, y_pred)
    # print(len(y_true), len(y_pred))

    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)

    print(precision, recall, accuracy)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    input_file = os.path.join('data', 'CGED16_HSK_Train_All.txt')
    ret_id, ret_text, ret_label = serialize(input_file)

    X_train, y_train, X_test, y_test, vocab = encode(ret_text, ret_label)

    feature_length = len(vocab.keys()) + 1

    model = Sequential()
    # model.add(Input(shape=(bucket_size, ), dtype='int32'))
    model.add(Embedding(feature_length, embedding_dim, input_length=bucket_size))
    model.add(Dropout(0.25))
    seq2seq = SimpleSeq2seq(
        input_dim=embedding_dim,
        input_length=bucket_size, 
        hidden_dim=100, 
        output_dim=len(error_dict.keys()) + 1,
        output_length=bucket_size,
        depth=3
    )
    model.add(seq2seq)

    # sequence = Input(shape=(bucket_size, ), dtype='int32')

    # embedded = Embedding(input_dim=feature_length, output_dim=embedding_dim, input_length=bucket_size) (sequence)
    # embedded = Dropout(0.25) (embedded)

    # # encoder
    # encoder = LSTM(hidden_dim, return_sequences=True) (embedded)
    # encoder = Dropout(0.25) (encoder)

    # decoder = LSTM(hidden_dim, return_sequences=True) (encoder)
    # decoder = Dropout(0.25) (decoder)

    # output = TimeDistributed(Dense(1)) (decoder)
    # model = Model(input=sequence, output=output)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    predict_array = model.predict(X_test, batch_size=batch_size)

    print(predict_array.shape)

    decode(X_test, y_test, predict_array)
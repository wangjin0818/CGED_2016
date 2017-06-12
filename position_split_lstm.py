from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import gensim

import numpy as np

from preprocess import position_serialize
from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    batch_size = 50
    nb_epoch = 20
    bucket_size = 10
    embedding_dim = 300

    input_file = os.path.join('data', 'CGED16_HSK_Train_All_Revised.txt')
    text, label = position_serialize(input_file, bucket_size=bucket_size, position='right')

    model_file = os.path.join('embedding', 'wiki.zh_CN.single.vector')
    model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False)

    train_text, test_text, train_label, test_label = train_test_split(text, label, \
        test_size=0.2, random_state=0)

    X_train, X_test = [], []

    combine_text = []
    for i in range(len(train_text)):
        new_vector = []
        tt = train_text[i]
        for j in range(len(tt)):
            try:
                vect = model[tt[j]]
                new_vector.append(vect)
            except:
                new_vector.append(np.zeros(embedding_dim))

        X_train.append(new_vector)
    
    X_train = np.array(X_train)
    y_train = np_utils.to_categorical(train_label, 5)
    print(X_train.shape)
    print(y_train.shape)

    for i in range(len(test_text)):
        new_vector = []
        tt = test_text[i]
        for j in range(len(tt)):
            try:
                vect = model[tt[j]]
                new_vector.append(vect)
            except:
                new_vector.append(np.zeros(embedding_dim))

        X_test.append(new_vector)

    X_test = np.array(X_test)
    y_test = test_label
    print(X_test.shape)

    # keras model
    model = Sequential()
    model.add(LSTM(120, input_shape=(bucket_size, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    pred_result = model.predict(X_test, batch_size=batch_size)

    print(pred_result.shape)

    y_pred = []
    for i in range(len(pred_result)):
        # print(pred_result[i])
        pred = np.argmax(pred_result[i], axis=0)
        y_pred.append(pred)

    print(y_test)
    print(y_pred)

    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    logging.info('precision score: %.3f' % (precision))
    logging.info('recal score: %.3f' % (recall))
    logging.info('f1 score: %.3f' % (f1))
    logging.info('accuracy score: %.3f' % (accuracy))
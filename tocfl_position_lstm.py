from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import gensim
import codecs
import cPickle
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation

from tocfl_preprocess import tocfl_position_train_serialize, tocfl_position_test_serialize, tocfl_position_train_sample

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    nb_epoch = 10
    bucket_size = 10
    embedding_dim = 300
    batch_size = 50

    train_file = os.path.join('pickle', 'tocfl_position_train.pickle')

    train_text, train_label = cPickle.load(open(train_file, 'rb'))
    logging.info('train text sample: %d' % (len(train_text)))
    logging.info('train label sample: %d' % (len(train_label)))

    print(train_label.count(0))
    print(train_label.count(1))
    print(train_label.count(2))
    print(train_label.count(3))
    print(train_label.count(4))

    model_file = os.path.join('embedding', 'wiki.zh_TW.single.vector')
    model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False)

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
                new_vector.append(np.random.uniform(-0.25, 0.25, embedding_dim))

        X_train.append(new_vector)

    X_train = np.array(X_train)
    y_train = np_utils.to_categorical(train_label, 5)
    print(X_train.shape)
    print(y_train.shape)

    test_file = os.path.join('data', 'CGED16_TOCFL_Test_Input.txt')
    test_position, test_text = tocfl_position_test_serialize(test_file, bucket_size=bucket_size, position='right')
    logging.info('test sid sample: %d' % (len(test_position)))
    logging.info('test label sample: %d' % (len(test_text)))

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
    print(X_test.shape)

    # keras model
    model = Sequential()
    model.add(LSTM(120, input_shape=(bucket_size, embedding_dim), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(120))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error', optimizer='adadelta')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    pred_result = model.predict(X_test, batch_size=batch_size)
    print(pred_result.shape)

    y_pred = []
    for i in range(len(pred_result)):
        # print(pred_result[i])
        pred = np.argmax(pred_result[i], axis=0)
        y_pred.append(pred)

    print(y_pred.count(0))
    print(y_pred.count(1))
    print(y_pred.count(2))
    print(y_pred.count(3))
    print(y_pred.count(4))

    position_save_file = os.path.join('result', 'track2', 'position_result.txt')
    with open(position_save_file, 'wb') as my_file:
        for i in range(len(y_pred)):
            my_file.write('%s\t%d\t%d\n' % (test_position[i][0], test_position[i][1], y_pred[i]))

    logging.info('result saved!')

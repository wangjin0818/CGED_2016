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

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    input_file = os.path.join('data', 'CGED16_HSK_Train_All_Revised.txt')
    text, label = position_serialize(input_file, bucket_size=7, position='right')

    model_file = os.path.join('embedding', 'wiki.zh_CN.single.vector')
    model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False)

    combine_text = []
    for i in range(len(text)):
        new_vecter = []
        tt = text[i]
        for j in range(len(tt)):
            try:
                vect = model[tt[j]]
                new_vecter.append(vect)
            except:
                continue

        new_vecter = np.array(new_vecter)
        new_vecter = np.mean(new_vecter, axis=0)

        try:
            new_vector = list(new_vecter)
        except:
            print(new_vecter)
        combine_text.append(new_vecter)

    X_train, X_test, y_train, y_test = train_test_split(combine_text, label, \
        test_size=0.2, random_state=0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import gensim
import cPickle

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from collections import defaultdict

from preprocess import detect_serialize
from preprocess import detect_single_serialize

# configuration
num_feature = 300

random_state = 0
test_size = 0.1

def build_data_train_test(text, label):
    """
    loads data and split into train and test sets.
    """
    revs = []
    vocab = defaultdict(float)

    X_train, X_test, y_train, y_test = train_test_split(ret_text, ret_label, random_state=random_state, test_size=test_size)
    logging.info('training sample size: %d' % (len(X_train)))
    logging.info('training label size: %d' % (len(y_train)))
    logging.info('testing sample size: %d' % (len(X_test)))
    logging.info('testing label size: %d' % (len(y_test)))

    for i in xrange(len(X_train)):
        line = X_train[i]

        rev = []
        orig_rev = ' '.join(line)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {
            'label': y_train[i],
            'text': orig_rev,
            'num_words': len(orig_rev.split()),
            'split': 1
        }
        revs.append(datum)

    for i in xrange(len(X_test)):
        line = X_test[i]

        rev = []
        orig_rev = ' '.join(line)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {
            'label': y_test[i],
            'text': orig_rev,
            'num_words': len(orig_rev.split()),
            'split': 0
        }
        revs.append(datum)

    return revs, vocab

def load_bin_vec(model, vocab, k=num_feature):
    """
    loads 400 x 1 word vecs from Google (Mikolov) pre-trained word2vec
    """
    word_vecs = {}

    for word in vocab.keys():
        try:
            word_vec = model[word]
        except:
            word_vec = np.random.uniform(-0.25, 0.25, k)
            logging.info('word %s cannot be found in embedding model' % (word))

        word_vecs[word] = word_vec

    return word_vecs

def get_W(word_vecs, k=num_feature):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    # position 0 was not used
    W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)

    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # load competition file
    input_file = os.path.join('data', 'CGED16_HSK_Train_All.txt')
    # ret_id, ret_text, ret_label = detect_serialize(input_file)
    ret_id, ret_text, ret_label = detect_single_serialize(input_file)

    revs, vocab = build_data_train_test(ret_text, ret_label)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    mean_l = np.mean(pd.DataFrame(revs)['num_words'])
    std_l = np.std(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))
    logging.info('mean sentence length: ' + str(mean_l))
    logging.info('std sentence length: ' + str(std_l))

    logging.info('loading word2vec...')
    embedding_file = os.path.join('embedding', 'wiki.zh_CN.single.vector')
    emvedding = gensim.models.Word2Vec.load_word2vec_format(embedding_file, binary=False)
    w2v = load_bin_vec(emvedding, vocab)
    logging.info('word2vec loaded!')
    logging.info('num words in word2vec: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v)
    logging.info('extracted index from word2vec! ')

    pickle_file = os.path.join('pickle', 'detect_HSK_single.pickle')
    cPickle.dump([revs, W, word_idx_map, vocab], open(pickle_file, 'wb'))
    logging.info('dataset created!')

    # +0.293, -0.293

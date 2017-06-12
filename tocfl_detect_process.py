from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import cPickle
import gensim

import numpy as np
import pandas as pd

from collections import defaultdict
from tocfl_preprocess import tocfl_detect_train_serialize, tocfl_detect_test_serialize

num_feature = 400
bucket_size = 40

def build_data_train_test(train_text, train_label, test_sid, test_text):
    """
    loads data and split into train and test sets.
    """
    revs = []
    vocab = defaultdict(float)

    for i in xrange(len(train_text)):
        line = train_text[i]

        if len(line) > bucket_size:
            continue

        rev = []
        orig_rev = ' '.join(line)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {
            'label': train_label[i],
            'text': orig_rev,
            'num_words': len(orig_rev.split()),
            'split': 1
        }
        revs.append(datum)

    for i in xrange(len(test_text)):
        line = test_text[i]
        text_length = len(line)

        for j in range((text_length / bucket_size) + 1):
            new_line = line[j * bucket_size: (j + 1) * bucket_size]

            rev = []
            orig_rev = ' '.join(new_line)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                'sid': test_sid[i],
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

    train_file = os.path.join('data', 'CGED16_TOCFL_Train_All.txt')
    train_text, train_label = tocfl_detect_train_serialize(train_file)
    logging.info('train text sample: %d' % (len(train_text)))
    logging.info('train label sample: %d' % (len(train_label)))

    test_file = os.path.join('data', 'CGED16_TOCFL_Test_Input.txt')
    test_sid, test_text = tocfl_detect_test_serialize(test_file)
    logging.info('test sid sample: %d' % (len(test_sid)))
    logging.info('test label sample: %d' % (len(test_text)))

    revs, vocab = build_data_train_test(train_text, train_label, test_sid, test_text)
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
    embedding_file = os.path.join('embedding', 'wiki.zh_TW.vector')
    emvedding = gensim.models.Word2Vec.load_word2vec_format(embedding_file, binary=False)
    w2v = load_bin_vec(emvedding, vocab)
    logging.info('word2vec loaded!')
    logging.info('num words in word2vec: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v)
    logging.info('extracted index from word2vec! ')

    pickle_file = os.path.join('pickle', 'tocfl_detect_train.pickle')
    cPickle.dump([revs, W, word_idx_map, vocab], open(pickle_file, 'wb'))
    logging.info('dataset created!')

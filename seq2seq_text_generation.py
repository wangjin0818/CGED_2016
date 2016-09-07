from __future__ import print_function

from keras.utils.data_utils import get_file

import os
import sys
import logging

import random
# import seq2seq
# from seq2seq.models import SimpleSeq2seq


from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
from seq2seq.layers.bidirectional import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.engine.topology import Layer


import numpy as np

class SimpleSeq2seq(Sequential):
    '''
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decoder the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence elements.
    The input sequence and output sequence may differ in length.

    Arguments:

    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3, 
            there will be 3 LSTMs on the enoding side and 3 LSTMs on the 
            decoding side. You can also specify depth as a tuple. For example,
            if depth = (4, 5), 4 LSTMs will be added to the encoding side and
            5 LSTMs will be added to the decoding side.
    dropout : Dropout probability in between layers.

    '''
    def __init__(self, output_dim, hidden_dim, output_length, depth=1, dropout=0.25, **kwargs):
        super(SimpleSeq2seq, self).__init__()
        if type(depth) not in [list, tuple]:
            depth = (depth, depth)
        self.encoder = LSTM(hidden_dim, **kwargs)
        self.decoder = LSTM(hidden_dim, return_sequences=True, **kwargs)
        for i in range(1, depth[0]):
            self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
            self.add(Dropout(dropout))
        self.add(self.encoder)
        self.add(Dropout(dropout))
        # self.add(RepeatVector(output_length))
        self.add(self.decoder)
        for i in range(1, depth[1]):
            self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
            self.add(Dropout(dropout))
        self.add(TimeDistributed(Dense(output_dim)))

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model
print('Build model...')
model = SimpleSeq2seq(input_dim=len(chars), hidden_dim=128, output_length=200287, output_dim=len(chars))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
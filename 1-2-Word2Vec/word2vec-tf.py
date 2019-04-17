#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

sentences = ["i like dog", "i love coffe", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
index2word = {i: w for w, i in enumerate(word_list)}
n_class = len(word2index)

n_step = 2
n_embedding = 100
n_hidden = 2


def make_batch(sentences):
    input_batch = []
    target_batch = []
    for s in sentences:
        word = s.split()
        input = [word2index[i] for i in word[:-1]]
        target = word2index[word[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch, target_batch


X = tf.placeholder(tf.float32, [None, ])

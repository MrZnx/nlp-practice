#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

seq_data = ['make', 'need', 'coal', 'word',
            'love', 'hate', 'live', 'home', 'hash', 'star']
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
char2index = {c: i for i, c in enumerate(char_arr)}
index2char = {i: c for i, c in enumerate(char_arr)}
num_char = len(char2index)

n_step = 3
n_hidden = 128


def make_batch(seq_data):
    input_batch = []
    target_batch = []
    for s in seq_data:
        input = [char2index[i] for i in s[:-1]]
        target = char2index[s[-1]]
        input_batch.append(np.eye(num_char)[input])
        target_batch.append(np.eye(num_char)[target])
    return input_batch, target_batch


X = tf.placeholder(tf.float32, [None, n_step, num_char])
Y = tf.placeholder(tf.float32, [None, num_char])

W = tf.Variable(tf.random_normal([n_hidden, num_char]))
b = tf.Variable(tf.random_normal([num_char]))

cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction = tf.cast(tf.argmax(model, 1), tf.int32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    input_batch, target_batch = make_batch(seq_data)
    for epoch in range(5000):
        _, loss = sess.run([optimizer, cost], feed_dict={
            X: input_batch,
            Y: target_batch
        })
        if (epoch + 1) % 1000 == 0:
            print('Epoch: ', epoch + 1, ' cost =', loss)

    input = [s[:-1] for s in seq_data]
    predict = sess.run([prediction], feed_dict={
        X: input_batch
    })
    print(input, '->', [index2char[i] for i in predict[0]])

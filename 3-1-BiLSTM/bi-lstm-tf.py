#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

word_list = list(set(sentence.split()))
word2index = {w: i for i, w in enumerate(word_list)}
index2word = {i: w for i, w in enumerate(word_list)}
n_class = len(word2index)
n_step = len(sentence.split())
n_hidden = 5


def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i in range(len(words) - 1):
        input = [word2index[j] for j in words[:i + 1]]
        input += [0] * (n_step - len(input))
        target = word2index[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch, target_batch


X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden)

outputs, states = tf.nn.bidirectional_dynamic_rnn(
    lstm_fw_cell, lstm_bw_cell, X, dtype=tf.float32)

outputs = tf.concat(outputs, 2)
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
    input_batch, target_batch = make_batch(sentence)
    for epoch in range(10000):
        _, loss = sess.run([optimizer, cost], feed_dict={
            X: input_batch,
            Y: target_batch
        })
        if (epoch + 1) % 1000 == 0:
            print('Epoch: ', epoch + 1, ' cost =', loss)
    predict = sess.run([prediction], feed_dict={
        X: input_batch
    })
    print(sentence)
    print([index2word[i] for i in [pre for pre in predict[0]]])

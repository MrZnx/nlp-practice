#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentences = ["i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
index2word = {i: w for i, w in enumerate(word_list)}
n_class = len(word2index)

n_step = 2
n_hidden = 5


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


X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_hidden)
# outputs: [batch_size, n_step, n_hidden] -> [?, 2, 5]
# states: [batch_size, n_hidden] -> [?, 5]
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
    input_batch, target_batch = make_batch(sentences)
    for epoch in range(5000):
        _, loss = sess.run([optimizer, cost], feed_dict={
            X: input_batch,
            Y: target_batch
        })
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', epoch + 1, 'cost =', loss)
    input = [s.split()[:-1] for s in sentences]
    predict = sess.run([prediction], feed_dict={
        X: input_batch
    })
    print(input, '->', [index2word[i] for i in predict[0]])

#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentences = ["i like dog", "i love coffe", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
index2word = {i: w for i, w in enumerate(word_list)}
# 词汇量
n_class = len(word2index)

n_step = 2
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


X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

input = tf.reshape(X, shape=[-1, n_step * n_class])
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

hidden = tf.nn.tanh(tf.matmul(input, H) + d)
model = tf.matmul(hidden, U) + b

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction = tf.argmax(model, 1)

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
            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))
    predict = sess.run([prediction], feed_dict={
        X: input_batch
    })
    input = [s.split()[:2] for s in sentences]
    print(input, '->', [index2word[i] for i in predict[0]])

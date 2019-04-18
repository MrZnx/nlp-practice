#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# params
embedding_size = 2
sequence_length = 3
num_classes = 2  # {0, 1}
filter_sizes = [2, 2, 2]
num_filters = 3

sentences = ["i love you", "he loves me", "she likes baseball",
             "i hate you", "sorry for that", "this is awful"]
# 1 is good, 0 is not good.
labels = [1, 1, 1, 0, 0, 0]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word2index)

# ex: [i love you] => [2, 4, 9], inputs shape -> (6, 3)
inputs = []
for s in sentences:
    inputs.append(np.asarray([word2index[i] for i in s.split()]))


# one-hot, outputs shape -> (6, 2)
outputs = []
for out in labels:
    outputs.append(np.eye(num_classes)[out])

# (6, 3)
X = tf.placeholder(tf.int32, [None, sequence_length])
# (6, 2)
Y = tf.placeholder(tf.int32, [None, num_classes])

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
# [batch_size, sequence_length, embedding_size]
embedded_chars = tf.nn.embedding_lookup(W, X)
# [batch_size, sequence_length, embedding_size, 1]
embedded_chars = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    # [batch_size, sequence_length - filter_size + 1, 1, num_filters] -> [?, 2, 1, 3]
    conv = tf.nn.conv2d(embedded_chars,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    # [batch_size, 1, 1, num_filters] -> [?, 1, 1, 3]
    pooled = tf.nn.max_pool(h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID')
    pooled_outputs.append(pooled)

num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, -1)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

Weight = tf.get_variable('W', shape=[num_filters_total, num_classes],
                         initializer=tf.contrib.layers.xavier_initializer())
Bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model = tf.nn.xw_plus_b(h_pool_flat, Weight, Bias)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

hypothesis = tf.nn.softmax(model)
predictions = tf.argmax(hypothesis, 1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5000):
        _, loss = sess.run([optimizer, cost], feed_dict={
            X: inputs,
            Y: outputs
        })
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', epoch + 1, 'cost =', loss)

    test_text = 'sorry hate you'
    tests = []
    tests.append(np.asarray([word2index[i] for i in test_text.split()]))

    predict = sess.run([predictions], feed_dict={
        X: tests
    })
    result = predict[0][0]
    if result == 0:
        print(test_text, ' Bad')
    else:
        print(test_text, ' Good')

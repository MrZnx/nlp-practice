#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_seq = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
index2word = {i: w for i, w in enumerate(word_list)}
n_class = len(word2index)

batch_size = 20
embedding_size = 2


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        random_inputs.append(np.eye(n_class)[data[i][0]])
        random_labels.append(np.eye(n_class)[data[i][1]])
    return random_inputs, random_labels


skip_grams = []
for i in range(1, len(word_seq) - 1):
    target = word2index[word_seq[i]]
    skip_grams.append([target, word2index[word_seq[i - 1]]])
    skip_grams.append([target, word2index[word_seq[i + 1]]])

inputs = tf.placeholder(tf.float32, [None, n_class])
labels = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_uniform([n_class, embedding_size], -1, 1))
WT = tf.Variable(tf.random_uniform([embedding_size, n_class], -1, 1))

hidden = tf.matmul(inputs, W)
output = tf.matmul(hidden, WT)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=output, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(5000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={
            inputs: batch_inputs,
            labels: batch_labels
        })

        if (epoch + 1) % 1000 == 0:
            print('Epoch: ', epoch + 1, ' cost =', loss)

    training_embedding = W.eval()
    for i, label in enumerate(word_list):
        x, y = training_embedding[i]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.show()

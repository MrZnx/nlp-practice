#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_seq = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
index2word = {i: w for w, i in enumerate(word_list)}
n_class = len(word2index)

batch_size = 20
embedding_size = 2
neg_sampled_size = 10


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        random_inputs.append(data[i][0])
        random_labels.append([data[i][1]])
    return random_inputs, random_labels


skip_grams = []
for i in range(1, len(word_seq) - 1):
    # [..., context1, target, context2, ...] window size = 1
    target = word2index[word_seq[i]]
    context = [word2index[word_seq[i - 1]], word2index[word_seq[i + 1]]]
    for w in context:
        skip_grams.append([target, w])

inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([n_class, embedding_size], -1, 1))
select_embed = tf.nn.embedding_lookup(embeddings, inputs)

nce_weights = tf.Variable(tf.random_uniform([n_class, embedding_size], -1, 1))
nce_bias = tf.Variable(tf.zeros([n_class]))

cost = tf.reduce_mean(tf.nn.nce_loss(
    nce_weights, nce_bias, labels, select_embed, neg_sampled_size, n_class))
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
            print('Epoch:', (epoch + 1), 'cost =', loss)

    trained_embeddings = embeddings.eval()

    for i, label in enumerate(word_list):
        x, y = trained_embeddings[i]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.show()

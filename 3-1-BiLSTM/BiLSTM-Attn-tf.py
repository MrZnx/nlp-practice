#!usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball",
             "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2index = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_list)

embedding_dim = 2
n_hidden = 5
n_step = 3
n_class = 2

input_batch = []
for s in sentences:
    input_batch.append(np.asarray([word2index[i] for i in s.split()]))

target_batch = []
for l in labels:
    target_batch.append(np.eye(n_class)[l])

X = tf.placeholder(tf.int32, [None, n_step])
Y = tf.placeholder(tf.int32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

embedding = tf.Variable(tf.random_uniform([vocab_size, n_hidden]))
input = tf.nn.embedding_lookup(embedding, X)

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

# outputs: [[batch_size, n_step, n_hidden], batch_size, n_step, n_hidden]
# states: (2, 2), states[0][0]: [batch_size, n_hidden]
outputs, states = tf.nn.bidirectional_dynamic_rnn(
    lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)
# outputs: [batch_size, n_step, n_hidden * 2]
outputs = tf.concat(outputs, 2)
# states: [batch_size, n_hidden * 2]
states = tf.concat([states[1][0], states[1][1]], 1)
# states: [batch_size, n_hidden * 2, 1]
states = tf.expand_dims(states, 2)

# attn_weights: [batch_size, n_step]
attn_weights = tf.squeeze(tf.matmul(outputs, states), 2)
soft_attn_weights = tf.nn.softmax(attn_weights, 1)
# outputs: [batch_size, n_hidden * 2, n_step]
# soft_attn_weights: [batch_size, n_step, 1]
# context: [batch_size, n_hidden * 2, 1]
context = tf.matmul(tf.transpose(
    outputs, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2))
# context: [batch_size, n_hidden * 2]
context = tf.squeeze(context, 2)

model = tf.matmul(context, W)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()

hypothesis = tf.nn.softmax(model)
prediction = tf.argmax(hypothesis, 1)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5000):
        _, loss, attention = sess.run([optimizer, cost, soft_attn_weights], feed_dict={
            X: input_batch,
            Y: target_batch
        })
        if (epoch + 1) % 1000 == 0:
            print('Epoch: ', epoch + 1, ' cost: ', loss)

    test_text = 'sorry hate you'
    tests = [np.asarray([word2index[i] for i in test_text.split()])]

    predict = sess.run([prediction], feed_dict={
        X: tests
    })
    result = predict[0][0]
    if result == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")

    fig = plt.figure(figsize=(6, 3))  # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels([''] + ['first_word', 'second_word',
                               'third_word'], fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + ['batch_1', 'batch_2', 'batch_3',
                               'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
    plt.show()

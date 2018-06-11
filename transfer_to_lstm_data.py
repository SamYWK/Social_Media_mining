# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 12:41:34 2018

@author: pig84
"""

import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

texts = []
y = []

#read from csv
with open ('train.csv','r', newline='',encoding='IBM437') as train_file:
    rows = csv.reader(train_file)
    for row in rows:
        y.append(row[1])
        texts.append(row[2])
texts = texts[1:]
y = np.array(y[1:]).reshape(-1, 1)
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

sentences =[]
dictionary = {}
dict_i = 0
max_length = 0

#encode each word
for text in texts:
    words = text.split(" ")
    words = [w.replace('.', '') for w in words]
    sentence = []
    word_count = 0
    for word in words:
        if word not in dictionary:
            dictionary[word] = dict_i
            dict_i += 1
        sentence.append(dictionary[word])
        word_count += 1
    if word_count > max_length:
        max_length = word_count
    sentences.append(np.asarray(sentence))

# fill 0 to sentences
for i in range(len(sentences)):
    while(sentences[i].size != max_length):
        sentences[i] = np.append(sentences[i], [0])
        
n = len(sentences)
df = pd.DataFrame(sentences)
df2 = pd.DataFrame(y)
df3 = pd.concat([df2, df], axis = 1)
df3.to_csv('./final_data_lstm.csv', index = False, header = False)
#tensorflow

## hyperparameters
#lr = 0.001                  # learning rate
#training_iters = 100000     # train step 上限
#batch_size = 1            
#n_inputs = 1               
#n_steps = max_length        # time steps
#n_hidden_units = 128        # neurons in hidden layer
#n_classes = 2              

#g_1 = tf.Graph()
#with g_1.as_default():
#    
#    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#    y = tf.placeholder(tf.float32, [None, n_classes])
#    
#    # 对 weights biases 初始值的定义
#    weights = {
#        # shape (28, 128)
#        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
#        # shape (128, 10)
#        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
#    }
#    biases = {
#        # shape (128, )
#        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
#        # shape (10, )
#        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
#    }
#    
#    def RNN(X, weights, biases):
#        # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
#        # X ==> (128 batches * 28 steps, 28 inputs)
#        X = tf.reshape(X, [-1, n_inputs])
#    
#        # X_in = W*X + b
#        X_in = tf.matmul(X, weights['in']) + biases['in']
#        # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
#        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
#        # 使用 basic LSTM Cell.
#        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
#        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
#        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
#        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
#        results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
#        return results
#    
#    pred = RNN(x, weights, biases)
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
#    
#    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#    
#    with tf.Session() as sess:
#        
#        init = tf.global_variables_initializer()
#        sess.run(init)
#        step = 0
#        for batch in range(int (n / batch_size)):
#            batch_xs = sentences[batch]
#            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#            batch_ys = y[batch]
#            print(type(batch_xs), type(batch_ys))
#            sess.run([train_op], feed_dict={
#                x: batch_xs,
#                y: batch_ys,
#            })
#            if step % 20 == 0:
#                print(sess.run(accuracy, feed_dict={
#                x: batch_xs,
#                y: batch_ys,
#                }))
#            step += 1
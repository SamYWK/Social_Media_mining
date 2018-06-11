# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:38:41 2018

@author: pig84
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('final_data_lstm.csv', header = None)
X = df.drop(df.columns[0], axis = 1)
X = X.drop(df.columns[1], axis = 1).values
Y = df.iloc[:, 0:2].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
#tensorflow

# hyperparameters
n = X.shape[0]
lr = 0.0001                  # learning rate
epochs = 100    # train step 上限
batch_size = 128            
n_inputs = 1               
n_steps = 128        # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 2              

g_1 = tf.Graph()
with g_1.as_default():
    
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    
    # 对 weights biases 初始值的定义
    weights = {
        # shape (28, 128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
        # shape (128, 10)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
    biases = {
        # shape (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        # shape (10, )
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }
    
    def RNN(X, weights, biases):
        # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
        # X ==> (128 batches * 28 steps, 28 inputs)
        X = tf.reshape(X, [-1, n_inputs])
    
        # X_in = W*X + b
        X_in = tf.matmul(X, weights['in']) + biases['in']
        # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
        # 使用 basic LSTM Cell.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
        return results
    
    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epochs):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                if batch % 20 == 0:
                    print(sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                    }))
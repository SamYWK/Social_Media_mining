import csv
import gensim
from gensim.models.doc2vec import TaggedDocument
import logging
import tensorflow as tf
import numpy as np
import pandas as pd

texts = []
y = []

with open ('processed_emoji.csv','r', newline='',encoding='IBM437') as train_file:
    rows = csv.reader(train_file)
    for row in rows:
        y.append(row[1])
        texts.append(row[2])

sentences =[]
T = []

for text in texts:
    words = text.split(" ")
    sentence = []
    for word in words:
        sentence.append(word)
    sentences.append(sentence)
n = len(sentences)
a = 1
for s in sentences:
    train = TaggedDocument(s,[str(a)])
    a = a + 1
    T.append(train)
    
model = gensim.models.Doc2Vec(T[1:],dm=1,alpha=0.1,window=5, min_alpha=0.025, min_count=1,size=784)
X = model[0].reshape(1, -1)

for m in range(1, (n-1)):
    print(m)
    X = np.concatenate((X, model[m].reshape(1, -1)), axis = 0)
    
df = pd.DataFrame(X)
df2 = pd.DataFrame(np.array(y[1:]).reshape(-1, 1))
df3 = pd.concat([df2, df], axis = 1)
df3.to_csv('./final_data_cnn.csv', index = False, header = False)
#n = 100000
#batch_size = 200
#learning_rate = 0.0001
#epochs = 100
#g_1 = tf.Graph()
#    
#with g_1.as_default():
#    with tf.device('/device:GPU:0'):
#        with tf.name_scope('X_placeholder'):
#            X_placeholder = tf.placeholder(tf.float32, [None, 100], name = 'x_inputs')
#        with tf.name_scope('y_placeholder'):
#            y_placeholder = tf.placeholder(tf.float32, [None, 1], name = 'y_inputs')
#        
#        #forward
#        x1 = tf.layers.dense(X_placeholder, 100, activation = tf.nn.relu, name = 'xlayer_1')
#        x2 = tf.layers.dense(x1, 100, activation = tf.nn.relu, name = 'xlayer_2')
#        x3 = tf.layers.dense(x2, 1, activation = None, name = 'xlayer_3')
#        
#        with tf.name_scope('loss_1'):
#            loss_1 = tf.nn.softmax_cross_entropy_with_logits(logits = x3, labels = y_placeholder)
#        with tf.name_scope('train_step_1'):
#            train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
#        #prediction
#        with tf.name_scope('accuracy'):
#            correct_prediction = tf.equal(tf.argmax(x3, 1), tf.argmax(y_placeholder, 1))
#            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        
#    #initializer
#    init = tf.global_variables_initializer()
#    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
#    config.gpu_options.allow_growth = True
#    
#    #saver
#    saver = tf.train.Saver()
#    with tf.Session(config = config) as sess:
#        sess.run(init)
#        #saver.restore(sess, "./saver/model.ckpt")
#        print(sess.run(accuracy, feed_dict = {X_placeholder : model, y_placeholder : y}))
#        #writer
#        #writer = tf.summary.FileWriter('logs/', sess.graph)
#        for epoch in range(epochs):
#            for batch in range(int (n / batch_size)):
#                batch_xs = model[(batch*batch_size) : (batch+1)*batch_size]
#                batch_ys = y[(batch*batch_size) : (batch+1)*batch_size]
#                sess.run(train_step_1, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
#                
#                if batch % 500 == 0:
#                    print(sess.run(accuracy, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
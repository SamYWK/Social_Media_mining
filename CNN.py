from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_name):
    df = pd.read_csv(file_name, header = None)
    X = df.drop(df.columns[0], axis = 1).values.astype(np.float32)
    
    #normalize X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df.iloc[:, 0].values.reshape(-1, 1)
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray()
    return train_test_split(X, y, test_size = 0.2, random_state = 1)

def main():
    X_train, X_test, y_train, y_test = load_data('final_data_cnn.csv')
    
    n = X_train.shape[0]
    batch_size = 200
    learning_rate = 0.0001
    epochs = 25
    g_1 = tf.Graph()
        
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            with tf.name_scope('X_placeholder'):
                X_placeholder = tf.placeholder(tf.float32, [None, 784], name = 'x_inputs')
            with tf.name_scope('y_placeholder'):
                y_placeholder = tf.placeholder(tf.float32, [None, 2], name = 'y_inputs')
                
            keep_prob = tf.placeholder(tf.float32)
            
            #forward
            X_resahpe = tf.reshape(X_placeholder, [-1, 28, 28, 1])
            
            #forward
            x1 = tf.layers.conv2d(
                    inputs = X_resahpe,
                    filters = 32,
                    kernel_size = 5,
                    padding = 'same',
                    activation = tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs = x1, pool_size=[2, 2], strides=2)
            
            x2 = tf.layers.conv2d(
                    inputs = pool1,
                    filters = 64,
                    kernel_size = 5,
                    padding = 'same',
                    activation = tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs = x2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
            dense_1 = tf.layers.dense(pool2_flat, 1024, activation = tf.nn.relu)
            dense_1_drop = tf.nn.dropout(dense_1, keep_prob)
            dense_2 = tf.layers.dense(dense_1_drop, 2, activation = None)
            
            with tf.name_scope('loss_1'):
                loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = dense_2, labels = y_placeholder))
#                loss_1 = tf.losses.mean_squared_error(predictions = x3, labels = y_placeholder)
            with tf.name_scope('train_step_1'):
                train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
            #prediction
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(dense_2, 1), tf.argmax(y_placeholder, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        #initializer
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        #saver
        saver = tf.train.Saver()
        output_accuracy = np.array([])
        with tf.Session(config = config) as sess:
            sess.run(init)
            #saver.restore(sess, "./saver/model.ckpt")
            print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test, keep_prob: 1}))
            #writer
            #writer = tf.summary.FileWriter('logs/', sess.graph)
            
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step_1, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys, keep_prob: 0.5})
                    
                    if batch % 500 == 0:
                        output_accuracy = np.append(output_accuracy, sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: y_test, keep_prob: 1}))
                        print('Epoch', epoch,
                        'Loss :', sess.run(loss_1, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys, keep_prob: 1}),
                        'Accuracy :', sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test, keep_prob: 1}))
        np.savetxt('CNN_Accuracy.csv', output_accuracy, delimiter=',')
                    
if __name__ == '__main__':
    main()
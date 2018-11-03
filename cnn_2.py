import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import matplotlib.pyplot as plt
dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
print('.............Data Loaded...............')
# %% Load Data

#Variable Initialization
learning_rate = 0.003
batch_size = 225
seq_len = 4096
input_num_units = 23
output_num_units = 1
epoch = 10
cost_plot = []


#def my_leaky_relu(x):
#    return tf.nn.leaky_relu(x, alpha=0.01)

tf_x = tf.placeholder(tf.float64, shape=[None, seq_len ,1],name="Input")
tf_exp_y =  tf.placeholder(tf.float64,[None,output_num_units],name="Expected_Output")

conv1 = tf.layers.conv1d(inputs = tf_x,
                filters = 4,
            kernel_size = (6),
            padding = 'valid',
            activation =tf.nn.leaky_relu,name='conv1')
#conv1 = tf.layers.conv2d(tf_x, filters=conv1_fmaps, kernel_size = conv1_ksize,
#                         strides = conv1_stride, padding=conv1_pad,
#                         activation = tf.nn.relu, name="conv1")

pool1 = tf.layers.max_pooling1d(inputs = conv1,
                pool_size = 2 ,
                strides = 2)
conv2 = tf.layers.conv1d(inputs = pool1,
            filters = 4,
            kernel_size = 5,
            padding = 'valid',
            activation =tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling1d(inputs = conv2,
                pool_size = 2 ,
                strides = 2)
conv3 = tf.layers.conv1d(inputs = pool2,
            filters = 10,
            kernel_size = 2,
            padding = 'valid',
            activation =tf.nn.leaky_relu)
pool3 = tf.layers.max_pooling1d(inputs = conv3,
                pool_size = 1,
                strides = 1)
conv4 = tf.layers.conv1d(inputs = pool3,
            filters = 10,
            kernel_size = 1,
            padding = 'valid',
            activation =tf.nn.leaky_relu)
pool4 = tf.layers.max_pooling1d(inputs = conv4,
                pool_size = 1 ,
                strides = 1)
conv5 = tf.layers.conv1d(inputs = pool4,
            filters = 15,
            kernel_size = 1,
            padding = 'valid',
            activation =tf.nn.leaky_relu)
pool5 = tf.layers.max_pooling1d(inputs = conv5,
                pool_size = 1,
                strides = 1)
P = tf.contrib.layers.flatten(pool5)
fc1 = tf.contrib.layers.fully_connected(P, 50, activation_fn =tf.nn.leaky_relu)
fc2 = tf.contrib.layers.fully_connected(fc1, 20, activation_fn = tf.nn.leaky_relu)
fc3 = tf.contrib.layers.fully_connected(fc2, output_num_units, activation_fn = tf.nn.softmax)
output_layer = fc3
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_exp_y, logits=output_layer)
cost = tf.reduce_mean(cross_entropy)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# predicted = tf.nn.sigmoid(output_layer)
# correct_pred = tf.equal(tf.round(predicted), tf_exp_y)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print ('...................CNN created...................')
# %% Training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('...............Starting To Run...............')
    i = 1
    j = 1
    ep = 0
    for ep in range(epoch):
        costp=0
        for j in range(len(dat[0])):
            buff0 = dat[:,j]
            for i in range(batch_size):
            	#print("channel =", j,"batch_no = ",i)
            	buff1 = buff0[i*seq_len:(i+1)*seq_len]
            	I_train= np.reshape(buff1, [1, int(buff1.shape[0]), 1])
            	Z_train = dat[i*seq_len:(i+1)*seq_len,-1].reshape(seq_len,1)
            	_,c = sess.run([train_op,cost], {tf_x: I_train, tf_exp_y: Z_train})
            cost_plot = np.append(cost_plot,c)
            #print('channel=',j, "T_Cost:%.4f" %c)
            costp += c
        print('Epoch',ep,"T_Cost:%.4f" %costp)
    plt.savefig('./cost.png')
print("\n Training Over")

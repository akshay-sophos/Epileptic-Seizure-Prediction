import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import matplotlib.pyplot as plt 
dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
dat = dat[:,:, np.newaxis]
print '.............Data Loaded...............'
# %% Load Data

#Variable Initialization
mini_batch = 300000
input_num_units = 23
output_num_units = 1
#epoch = 1000
cost_plot = []

#def my_leaky_relu(x):
#    return tf.nn.leaky_relu(x, alpha=0.01)

tf_x = tf.placeholder(tf.float64, shape=[None, input_num_units,1],name="Input")
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
cost = tf.losses.softmax_cross_entropy(tf_exp_y, output_layer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)
print '...................CNN created...................'
# %% Training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('...............Starting To Run...............')
    for i in range(mini_batch):
        I_train = dat[i*100:(i+1)*100,0:23]
        Z_train = dat[i*100:(i+1)*100,23]
        _,c = sess.run([train_op,cost], {tf_x: I_train, tf_exp_y: Z_train})
        cost_plot = np.append(cost_plot,c)
        print(i, "T_Cost:%.4f" %c)
    # To plot Cost w.r.t time
#    f, axarr = plt.subplots(2, sharex=True)
#    axarr[0].plot(cost_plot)
#    axarr[0].set_title('cost_plot')
    plt.savefig('./Cost Plot CNN/cost.png')
#plt.show()
#saver = tf.train.Saver()
#saver.save(sess, './save/model.ckpt')
print("\n Training Over")

# %% Testing

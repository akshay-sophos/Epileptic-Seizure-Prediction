import numpy as np
import tensorflow as tf
#import os
#import matplotlib

#Variable Initialization
mini_batch = 300
input_num_units = 23
output_num_units = 3
epoch = 100

#Load Data


tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_y =  tf.placeholder(tf.float32,[None,output_num_units],name="Expected_Output")
conv1 = tf.nn.conv1d(inputs = tf_x,
			filters = 4,
			kernel_size = 6,
			padding = 'valid'
			activation = tf.nn.leaky_relu(alpha = 0.01))
pool1 = tf.layers.max_pooling1d(inputs = conv1,
				pool_size = 2 ,
				strides = 2)
conv2 = tf.layers.conv1d(inputs = pool1,
			filters = 4
			kernel_size = 5,
			padding = 'valid',
			activation = tf.nn.leaky_relu(alpha = 0.01))
pool2 = tf.layers.max_pooling1d(inputs = conv2,
				pool_size = 2 ,
				strides = 2)
conv3 = tf.layers.conv1d(inputs = pool2,
			filters = 10
			kernel_size = 4,
			padding = 'valid',
			activation = tf.nn.leaky_relu(alpha = 0.01))
pool3 = tf.layers.max_pooling1d(inputs = conv3,
				pool_size = 2,
				strides = 2)
conv4 = tf.layers.conv1d(inputs = pool3,
			filters = 10
			kernel_size = 4,
			padding = 'valid',
			activation = tf.nn.leaky_relu(alpha = 0.01))
pool4 = tf.layers.max_pooling1d(inputs = conv4,
				pool_size = 2 ,
				strides = 2)
conv5 = tf.layers.conv1d(inputs = pool4,
			filters = 15
			kernel_size = 4,
			padding = 'valid',
			activation = tf.nn.leaky_relu(alpha = 0.01))
pool5 = tf.layers.max_pooling1d(inputs = conv5,
				pool_size = 2,
				strides = 2)
P = tf.contrib.layers.flatten(pool5)
fc1 = tf.contrib.layers.fully_connected(P, 50, activation_fn =  tf.nn.leaky_relu(alpha = 0.01))
fc2 = tf.contrib.layers.fully_connected(fc1, 20, activation_fn =  tf.nn.leaky_relu(alpha = 0.01))
fc3 = tf.contrib.layers.fully_connected(fc2, 3, activation_fn = tf.nn.softmax)
output_layer = fc3
cost = tf.losses.softmax_cross_entropy(tf_exp_q, output_layer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in epoch:
        #Complete the windowing
        for i in range mini_batch:
            #I_train =
            #Z_train =
            _,c = sess.run([train_op,cost], {tf_x: I_train, tf_exp_y: Z_train})
        cost_plot = np.append(cost_plot,c)#tot_cost)
        print(ep, "T_Cost:%.4f" %c)
    saver = tf.train.Saver()
    saver.save(sess, './save/model.ckpt')
    print("\n Training Over")
    # To plot Cost w.r.t time
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(cost_plot)
    axarr[0].set_title('cost_plot')
    plt.savefig('./A$K/cost.png')
    plt.show()

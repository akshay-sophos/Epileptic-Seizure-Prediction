import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import matplotlib.pyplot as plt
dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
dat = np.expand_dims(dat, axis=0)
print ('.............Data Loaded...............')

# %% Load Data
#Variable Initialization
epoch = 10
batch_size = 3
seq_len = 400
learning_rate = 0.001
n_classes = 2
n_channels = 23
window = batch_size/2
X = tf.placeholder(tf.float32, [None,seq_len,n_channels], name = 'Input') #The 1st None is batch_size
Y = tf.placeholder(tf.int32, [None,seq_len,1], name = 'Expected_Output')

# (batch, 4096, 28) --> (batch, 2048, 56)
conv1 = tf.layers.conv1d(inputs=X, filters=56, kernel_size=2, strides=1,padding='same', activation = tf.nn.relu)
max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

# (batch, 2048, 56) --> (batch, 1024, 112)
conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=112, kernel_size=2, strides=1,padding='same', activation = tf.nn.relu)
max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

# (batch, 1024, 112) --> (batch, 512, 224)
conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=224, kernel_size=2, strides=1,padding='same', activation = tf.nn.relu)
max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

# (batch, 256, 224) --> (batch, 128, 448)
conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=448, kernel_size=2, strides=1,padding='same', activation = tf.nn.relu)
max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

P = tf.contrib.layers.flatten(max_pool_4)
fc1 = tf.contrib.layers.fully_connected(P, 50*seq_len, activation_fn =tf.nn.leaky_relu)
fc2 = tf.contrib.layers.fully_connected(fc1, 20*seq_len, activation_fn = tf.nn.leaky_relu)
fc3 = tf.contrib.layers.fully_connected(fc2, 1*seq_len, activation_fn = None)
output_layer = tf.reshape(fc3,tf.shape(Y))

loss_op = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=output_layer)
#loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels=Y)
#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)#,global_step=tf.train.get_global_step())

print ('...................CNN created...................')



# %% Training

train_acc = []
train_loss = []
n = 1
I_train = dat[:,0*seq_len:(0+1)*seq_len,:-1]
Z_train = dat[:,0*seq_len:(0+1)*seq_len, -1]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for m in range(epoch):
        for i in range(n,n+batch_size):
            I_train = np.vstack([I_train,dat[:,i*seq_len:(i+1)*seq_len,:-1]])
            Z_train = np.vstack([Z_train,dat[:,i*seq_len:(i+1)*seq_len, -1]])
        n += int(window)
        feed = {X : I_train, Y : Z_train[:,:,np.newaxis]}
        loss, _  = sess.run([loss_op, train_op], feed_dict = feed)
        train_loss.append(loss)
        print("Iteration:",m," Train loss: ",loss)#,"Train acc: {:.6f}".format(acc))
        I_train = np.delete(I_train,np.s_[0:-1],axis=0)
        Z_train = np.delete(Z_train,np.s_[0:-1],axis=0)
    plt.plot(train_loss)
    n+=100
    for m in range(epoch):
        for i in range(n,n+batch_size):
            I_train = np.vstack([I_train,dat[:,i*seq_len:(i+1)*seq_len,:-1]])
            Z_train = np.vstack([Z_train,dat[:,i*seq_len:(i+1)*seq_len, -1]])
        n += int(window)
        feed = {X : I_train, Y : Z_train[:,:,np.newaxis]}
        loss = sess.run([loss_op], feed_dict = feed)
        train_loss.append(loss)
        print("Iteration:",m," Test loss: ",loss)#,"Train acc: {:.6f}".format(acc))
        I_train = np.delete(I_train,np.s_[0:-1],axis=0)
        Z_train = np.delete(Z_train,np.s_[0:-1],axis=0)
plt.savefig('./Cost Plot CNN/test_cost.png')
#saver.save(sess,"checkpoints-cnn/har.ckpt")

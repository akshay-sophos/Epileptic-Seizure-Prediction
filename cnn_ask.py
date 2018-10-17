import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import matplotlib.pyplot as plt
dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
dat = np.expand_dims(dat, axis=0)
print ('.............Data Loaded...............')

# %% Load Data
#Variable Initialization
epoch = 100
batch_size = 26
seq_len = 4096
learning_rate = 0.001
n_classes = 2
n_channels = 23
window = batch_size/2
X = tf.placeholder(tf.float32, [None,seq_len,n_channels], name = 'Input') #The 1st None is batch_size
Y = tf.placeholder(tf.float32, [None,seq_len,1], name = 'Expected_Output')

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

    # Flatten and add dropout
    #flat = tf.reshape(max_pool_4, (-1, 128*448))
    #flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Predictions
    #outputlayer = tf.layers.dense(flat, n_classes)

    # Cost function and optimizer
P = tf.contrib.layers.flatten(max_pool_4)
fc1 = tf.contrib.layers.fully_connected(P, 50, activation_fn =tf.nn.leaky_relu)
fc2 = tf.contrib.layers.fully_connected(fc1, 20, activation_fn = tf.nn.leaky_relu)
fc3 = tf.contrib.layers.fully_connected(fc2, 1 , activation_fn = None)
output_layer = fc3
#cost = tf.nn.softmax_cross_entropy_with_logits(labels = labels_, logits = output_layer)#This will be deprecated soon.SO find an alternative
#cost = tf.reduce_sum(cost)
#optimizer = tf.train.AdamOptimizer().minimize(cost)




    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
#    logits_train = conv_net(features, num_classes, dropout, reuse=False,
#                            is_training=True)
#    logits_test = conv_net(features, num_classes, dropout, reuse=True,
#                           is_training=False)
#
#    # Predictions
#    pred_classes = tf.argmax(logits_test, axis=1)
#    pred_probas = tf.nn.softmax(logits_test)
#
#    # If prediction mode, early return
#    if mode == tf.estimator.ModeKeys.PREDICT:
#        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels=Y))
#logits_train, labels=tf.cast(labels, dtype=tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)#,global_step=tf.train.get_global_step())
#    # Evaluate the accuracy of the model
#acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)



print ('...................CNN created...................')



# %% Training
# Accuracy
#correct_pred = tf.equal((output_layer, labels_))#Argmax will be deprecated
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
#BULLSHIT above

train_acc = []
train_loss = []
n = 1
I_train = dat[:,0*seq_len:(0+1)*seq_len,:-1]
Z_train = dat[0*seq_len:(0+1)*seq_len, -1].T
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for m in range(epoch):
        for i in range(n,n+batch_size):
            I_train = np.vstack([I_train,dat[:,i*seq_len:(i+1)*seq_len,:-1]])
            Z_train = np.vstack([Z_train,dat[i*seq_len:(i+1)*seq_len, -1].T])
        n += window
        feed = {X : I_train, Y : Z_train}
        loss, _  = sess.run([loss_op, train_op], feed_dict = feed)
        train_loss.append(loss)
        print("Iteration: {:d}".format(epoch),"Train loss: {:6f}".format(loss))#,"Train acc: {:.6f}".format(acc))
        I_train = np.delete(I_train,np.s_[0:-1],axis=0)
        Z_train = np.delete(Z_train,np.s_[0:-1],axis=0)
    plt.plot(cost_plot)
plt.savefig('./Cost Plot CNN/cost.png')
#saver.save(sess,"checkpoints-cnn/har.ckpt")

import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import matplotlib.pyplot as plt 
dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
dat = np.expand_dims(dat, axis=0) #MAke sure that the dimension is added in the beginning___> yes it is added in the begining
print ('.............Data Loaded...............')

# %% Load Data
#Variable Initialization

batch_size = 225       
seq_len = 4096         
learning_rate = 0.001
n_classes = 2
n_channels = 23

#Initialize all the variables here


inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs') #As per RAvikanths Explanation for conv1D should that NOne be converted to 1
#always??
labels_ = tf.placeholder(tf.float32, [None, 1], name = 'labels')
keep_prob_ = tf.placeholder(tf.float32, name = 'keep') #Why to keep this variable as tf variable_
#------> no need variable is also enough if we are fixed with one prob...for debugging if we want to change the placeholder serves more purpose
learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate') #SAme question as above--->same as above
	

# (batch, 4096, 28) --> (batch, 2048, 56)
#What about the 3rd dimension for conv 1 Will it automatically figure it out?-------------yes it will figure it out u can visualise it in pic that i had send to u
conv1 = tf.layers.conv1d(inputs=inputs_, filters=56, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
# (batch, 2048, 56) --> (batch, 1024, 112)
conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=112, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
# (batch, 1024, 112) --> (batch, 512, 224)
conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=224, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
# (batch, 256, 224) --> (batch, 128, 448)
conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=448, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
#I am assuming everything till here is functional correct.Ravikanth cross check them manually
	
    # Flatten and add dropout
    #flat = tf.reshape(max_pool_4, (-1, 128*448))
    #flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    #outputlayer = tf.layers.dense(flat, n_classes)
    
    # Cost function and optimizer
P = tf.contrib.layers.flatten(max_pool_4)
fc1 = tf.contrib.layers.fully_connected(P, 50, activation_fn =tf.nn.leaky_relu)
fc2 = tf.contrib.layers.fully_connected(fc1, 20, activation_fn = tf.nn.leaky_relu)
fc3 = tf.contrib.layers.fully_connected(fc2, 2, activation_fn = None)
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
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer,labels=labels_))
#logits_train, labels=tf.cast(labels, dtype=tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())

#    # Evaluate the accuracy of the model
#acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)



print '...................CNN created...................'



# %% Training
# Accuracy
#correct_pred = tf.equal((output_layer, labels_))#Argmax will be deprecated
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
#BULLSHIT above
   
train_acc = []
train_loss = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for i in range(batch_size):
        I_train = dat[:,i*seq_len:(i+1)*seq_len,:-1]  #MEaning of [:,...,...] SHouldnt it be [1,...,..]---> any thing is no problem
        X = dat[i*seq_len:(i+1)*seq_len, -1] #SHouldnt we reduce the dimension from 3D to 2d? ----> no
        Z_train = X.T #IS it reallyt transpose? and why are we taking the transpose? -> yes it is transpose
        feed = {inputs_ : I_train, labels_ : Z_train, keep_prob_ : 0.5, learning_rate_ : learning_rate} #WE arent using drop
        loss, _  = sess.run([cost, optimizer], feed_dict = feed)
#        train_acc.append(acc)
        train_loss.append(loss)
        if (iteration % 5 == 0):
            print("Iteration: {:d}".format(iteration),"Train loss: {:6f}".format(loss))#,"Train acc: {:.6f}".format(acc))
        #Plot also for every 5 iteraton
        iteration += 1  
    plt.plot(cost_plot)  
plt.savefig('./Cost Plot CNN/cost.png')
#saver.save(sess,"checkpoints-cnn/har.ckpt")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import genfromtxt

dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
print ('.............Data Loaded...............')

TF_LEARN_RATE = 0.005 #Learning Rate for Gradient Descent
ep = 375000
NUM_EPISODES = 10000+ep
SEQ_LEN = 100
PLOT_APPEND = 10
cost_plot = []
# number of neurons in each layer
input_num_units = 23
hidden_num_units1 = 60
hidden_num_units2 = 60
hidden_num_units3 = 60
output_num_units = 1
seed = 10

# define placeholders
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_y =  tf.placeholder(tf.int32,[None,1],name="Actual_Y")
hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
hidden_layer3 = tf.layers.dense(hidden_layer2, hidden_num_units3, tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer3, output_num_units)
cost = tf.losses.sparse_softmax_cross_entropy(tf_exp_y,output_layer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)
print ('...................NN created...................')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while ep<=NUM_EPISODES:
        tot_cost = 0
        I = dat[(ep-1)*SEQ_LEN:(ep)*SEQ_LEN,:-1]
        Z = dat[(ep-1)*SEQ_LEN:(ep)*SEQ_LEN, -1].reshape(SEQ_LEN,1)
        #If this doesn't work use the for loop used in real_uruhl
        _,c = sess.run([train_op,cost], {tf_x: I, tf_exp_y: Z})
        if(ep%PLOT_APPEND == 0):
            cost_plot = np.append(cost_plot,c)
            print(ep, "T_Cost:%.4f" %c)
        ep = ep+1
    print("\n Training Over")
    saver = tf.train.Saver()
    saver.save(sess, './save/model.ckpt')
    # To plot Cost w.r.t time
    plt.plot(cost_plot)
    plt.title('cost_plot')
    plt.savefig('./cost_angle.png')
    #plt.show()

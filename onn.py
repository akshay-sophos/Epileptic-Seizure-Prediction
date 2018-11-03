import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import genfromtxt


dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
print ('.............Data Loaded...............')

learning_rate = 0.003
THRESHOLD = 0.5
NUM_EPISODES = 40
accuracy_plot_train=[]
cost_plot_train=[]

# number of neurons in each layer
input_num_units = 23
hidden_num_units1 = 50
hidden_num_units2 = 50
hidden_num_units3 = 50
output_num_units = 1
seed = 10

# define placeholders
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_y =  tf.placeholder(tf.float32,[None,1],name="Actual_Y")
hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
hidden_layer3 = tf.layers.dense(hidden_layer2, hidden_num_units3, tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer3, output_num_units)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_exp_y, logits=output_layer)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
predicted = tf.nn.sigmoid(output_layer)
correct_pred = tf.equal(tf.round(predicted), tf_exp_y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#cm = tf.confusion_matrix(tf_exp_y,predicted>THRESHOLD)
print ('...................NN created...................')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(dat[:,:-1], dat[:,-1], test_size = 0.3, random_state = 0)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	y_train = y_train.reshape(y_train.size,1)
	epoch = 0
	for epoch in range(NUM_EPISODES):
		_,a,c= sess.run([optimizer,accuracy,cost], {tf_x: X_train, tf_exp_y:y_train})
		cost_plot_train = np.append(cost_plot_train,c)
		accuracy_plot_train = np.append(accuracy_plot_train,a)
		print(epoch, "T_Cost:%.4f" %c,"Accuracy:%.4f" %a)
	print("\n Training Over")
	y_pred= sess.run(predicted, {tf_x: X_train, tf_exp_y:y_train})
	y__pred = (y_pred>THRESHOLD).astype(float)
	print(confusion_matrix(y_train,y__pred))
	print("Precision",precision_score(y_train, y__pred))
	print("Recall",recall_score(y_train, y__pred))
	print("F1",f1_score(y_train, y__pred))
	#TEST...........................
	y_test = y_test.reshape(y_test.size,1)
	c,a,y_pred= sess.run([cost,accuracy,predicted], {tf_x: X_test, tf_exp_y:y_test})
	print("T_Cost:%.4f" %c,"Accuracy:%.4f" %a)
	y__pred = (y_pred>THRESHOLD).astype(float)
	#print(np.sum(np.absolute(y__pred-y_test)))
	print("\n Testing Over")
	print(confusion_matrix(y_test,y__pred))
	print("Precision",precision_score(y_test, y__pred))
	print("Recall",recall_score(y_test, y__pred))
	print("F1",f1_score(y_test, y__pred))

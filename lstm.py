import numpy as np
from numpy import genfromtxt
dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
#dat = np.random.rand(30,24)
print('............Data Loaded.................')

#Variable Declaration
THRESHOLD = 0.5
p_dropout = 0.2
batch_size= 10
epoch     = 100
hip       = 23
h1        = 60
h2        = 60
h3        = 60
hop       = 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dat[:,:-1], dat[:,-1], test_size = 0.3, random_state = 0)
print('............Data Split..................')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = np.expand_dims(X_train, axis=0)
y_train = np.expand_dims(y_train, axis=0).transpose()
y_train = np.expand_dims(y_train, axis=0)
X_test = np.expand_dims(X_test, axis=0)
y_test = np.expand_dims(y_test, axis=0).transpose()
y_test = np.expand_dims(y_test, axis=0)
print(np.shape(X_train))
print(np.shape(y_train))
print(np.shape(X_test))
print(np.shape(y_test))
print('............Data Normalized.............')

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz


classifier = Sequential()
#Adding input layer and first hidden layer
classifier.add(LSTM(h1, init = 'uniform',return_sequences=True, activation = 'relu',use_bias=True, input_shape = (None,hip)))
classifier.add(Dropout(p_dropout))
#Adding second hidden layer
classifier.add(LSTM(h2, return_sequences=True,use_bias=True,activation = 'relu'))
classifier.add(Dropout(p_dropout))
#Adding third hidden layer
classifier.add(LSTM(h3, return_sequences=True,use_bias=True,activation = 'relu'))
classifier.add(Dropout(p_dropout))
#Adding the output layer
classifier.add(Dense(units =hop, use_bias=True,init = 'uniform', activation = 'sigmoid'))
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the ANN to the training set
history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size = batch_size, epochs = epoch, verbose=0)
print('............Data Trained................')

# Predicting the Test set results
y_pred = classifier.predict(X_train)
y_pred = (y_pred > THRESHOLD).astype(float)
print(y_train.shape,y_pred.shape)
y_pred = y_pred.reshape(np.size(y_pred))
y_train = y_train.reshape(np.size(y_train))
print(y_train.shape,y_pred.shape)
print(y_train,y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report
print(confusion_matrix(y_train,y_pred))
print("Precision",precision_score(y_train, y_pred))
print("Recall",recall_score(y_train, y_pred))
print("F1",f1_score(y_train, y_pred))
print('............Test Over...................')


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > THRESHOLD)
y_pred = y_pred.reshape(np.size(y_pred))
y_test = y_test.reshape(np.size(y_test))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report
print(confusion_matrix(y_test,y_pred))
print("Precision",precision_score(y_test, y_pred))
print("Recall",recall_score(y_test, y_pred))
print("F1",f1_score(y_test, y_pred))
print('............Test Over...................')

#%%
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
ann_viz(classifier, view=True, filename='net.gv', title='Neural Network')
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

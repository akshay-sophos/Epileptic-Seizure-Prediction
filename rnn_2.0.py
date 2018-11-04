import numpy as np
from numpy import genfromtxt
from features import get_time_domain_features

X = np.array(0,np.float64)
sec = 3

dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
numrows = len(dat)
numcols = len(dat[0])
j = 0
while j<(numcols-1):
 i = 0
 while i<numrows:
  if ((i == 0)&(j == 0)):
   #X = dat[i:i+256*sec,j].transpose()
   X_f = get_time_domain_features(dat[i:i+256*sec,j].transpose())
   y = dat[i,23]
  else :
   #X = np.vstack((X,dat[i:i+256*sec,j].transpose()))
   X_f = np.vstack((X_f,get_time_domain_features(dat[i:i+256*sec,j].transpose())))
   y = np.vstack((y,dat[i,23]))
  i = i+256*sec
 j = j+1
 print(j)
#%% 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz


def f1_score(cm):
	tp = cm[1,1] 
	fn = cm[1,0] 
	fp = cm[0,1] 
	tn = cm[0,0]

	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	F1_score = 2*((precision*recall)/(precision+recall))
	return F1_score

classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 8))

#Adding second hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size = 10, epochs = 100, verbose=0)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('F1_score =    {:.3f}'.format(f1_score(cm)))

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

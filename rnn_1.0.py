import numpy as np
from numpy import genfromtxt

X = np.array(0,np.float64)

dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
numrows = len(dat)
numcols = len(dat[0])
j = 0
while j<(numcols-1):
 i = 0
 while i<numrows:
  if ((i == 0)&(j == 0)):
   X = dat[i:i+256,j].transpose()
   y = dat[i,23]
  else :
   X = np.vstack((X,dat[i:i+256,j].transpose()))
   y = np.vstack((y,dat[i,23]))
  i = i+256
 j = j+1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 116, init = 'uniform', activation = 'relu', input_dim = 256))

#Adding second hidden layer
classifier.add(Dense(output_dim = 116, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
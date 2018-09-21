import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt 
import pandas as pd

dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')



X = dat[:, 1:22]
y = dat[:, :23]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Making the ANN

#Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(output_dim = , init = 'uniform', activation = 'relu', input_dim = ))

#Adding second hidden layer
classifier.add(Dense(output_dim = , init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = , init = 'uniform', activation = 'sigmoid'))

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
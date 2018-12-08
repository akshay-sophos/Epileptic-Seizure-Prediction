import numpy as np
import time
from numpy import genfromtxt
from keras import regularizers,optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import glob
#Variable Declaration
THRESHOLD = 0.5
lambda_reg = 0.1
l_rate = 0.01
l_rate_decay = 0.1
p_dropout = 0.2
batch_size= 500
valid_split = 0
epoch     = 8
hip       = 23
h1        = 20
h2        = 10
h3        = 20
h4        = 30
hop       = 1
#%%

#from ann_visualizer.visualize import ann_viz

#Should we use BatchNormalization before or after activation function??????

classifier = Sequential()
#Adding input layer and first hidden layer
classifier.add(LSTM(h1,return_sequences=True, activation = 'relu',use_bias=True,kernel_regularizer=regularizers.l2(lambda_reg), input_shape = (None,hip)))
classifier.add(BatchNormalization())
classifier.add(Dropout(p_dropout))
#Adding second hidden layer
classifier.add(LSTM(h2, return_sequences=True,use_bias=True,kernel_regularizer=regularizers.l2(lambda_reg),activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(p_dropout))
#Adding third hidden layer
classifier.add(LSTM(h3, return_sequences=True,use_bias=True,kernel_regularizer=regularizers.l2(lambda_reg),activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(p_dropout))
#4th layer
classifier.add(LSTM(h4, return_sequences=True,use_bias=True,activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(p_dropout))
#Adding the output layer
classifier.add(Dense(units =hop, use_bias=True, activation = 'sigmoid'))
adam = optimizers.Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=l_rate_decay, amsgrad=False)
#Compiling the ANN
#%%
path = "D:/Major/Data/chb_01/"
file = glob.glob(path+'\chb_01_'+'[0-9][0-9]'+'.csv')
#file = [f for f in os.listdir(path) if re.match(r'chb_01_[0-9]*[0-9].csv', f)]
for i in sorted(file):
    t1 = time.time();
    dat = genfromtxt(i,dtype =(float), delimiter=',')
    t2 = time.time()
    print('............Data Loaded.................')
    print("Time for "+str(i[-6:-4])+" is "+str(t2-t1)+"sec")
    
    X_train, X_test, y_train, y_test = train_test_split(dat[:,:-1], dat[:,-1], test_size = 0.3, random_state = 0)
    print('............Data Split..................')

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = np.expand_dims(X_train, axis=1) 
    X_test = np.expand_dims(X_test, axis=1) 
    y_train = np.expand_dims(y_train, axis=1) 
    y_train = np.expand_dims(y_train, axis=1) 
    y_test = np.expand_dims(y_test, axis=1) 
    y_test = np.expand_dims(y_test, axis=1) 
    print('............Data Normalized.............')
    print(X_train.shape)
    print(y_train.shape)
    #
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #Fitting the ANN to the training set
    history = classifier.fit(X_train, y_train, validation_split=valid_split, batch_size = batch_size, epochs = epoch, verbose=1)
    print('............Data Trained................')
    # Predicting the Test set results
    y_pred = classifier.predict(X_train,verbose=1)
    y_pred = (y_pred > THRESHOLD).astype(float)
    y_pred = np.squeeze(y_pred)
    y_train = np.squeeze(y_train)
    print(y_pred.shape,y_pred[1:10],"    ",y_train.shape,y_train[1:10])
    #
    #y_pred = y_pred.reshape(np.size(y_pred))
    #y_train = y_train.reshape(np.size(y_train))
    # Making the Confusion Matrix
    print(confusion_matrix(y_train,y_pred,labels=[0,1]))
    print("Precision",precision_score(y_train, y_pred,labels=[0,1]))
    print("Recall",recall_score(y_train, y_pred,labels=[0,1]))
    print("F1",f1_score(y_train, y_pred,labels=[0,1]))
    print('............Test Over...................')
    
    
    #%Predicting the Test set results
    y_pred = classifier.predict(X_test,verbose=1)
    y_pred = (y_pred > THRESHOLD).astype(float)
    y_pred = np.squeeze(y_pred)
    y_test = np.squeeze(y_test)
    # Making the Confusion Matrix
    print(confusion_matrix(y_test,y_pred,labels=[0,1]))
    print("Precision",precision_score(y_test, y_pred,labels=[0,1]))
    print("Recall",recall_score(y_test, y_pred,labels=[0,1]))
    print("F1",f1_score(y_test, y_pred,labels=[0,1]))
    print('............Test Over...................')

#%%
plot_model(classifier, to_file='D:\Major\model_plot.png', show_shapes=True, show_layer_names=True)
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#ann_viz(classifier, view=True, filename='net.gv', title='Neural Network')

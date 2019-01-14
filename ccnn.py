#importing libraries
import keras
from keras.models import Sequential
#from keras.layers import GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras import regularizers
from sklearn.metrics import confusion_matrix

#%%
import glob,time
path = "D:\seizure data\seizureddata(0,1,2)/"
file = glob.glob(path+'\chb_'+'[0-9][0-9]'+'_'+'[0-9][0-9]'+'.csv')
#%%
#file = [f for f in os.listdir(path) if re.match(r'chb_01_[0-9]*[0-9].csv', f)]
df = np.empty([0,25])
for i in sorted(file):
    t1 = time.time();
    n = pd.read_csv(i,header=None)# We can definitely remove the variable by replacing n in the next line
    print(n.shape,df.shape)
    df = np.vstack((df,n))
    t2 = time.time()
    print(df.shape)
    print("Time for "+str(i[-6:-4])+" is "+str(t2-t1)+"sec")
print('............Data Loaded.................')
#%%
print(df.shape)
data = pd.DataFrame(data=df)
#%%
print (len(data))
df_x = data.loc[:,0:22].values.reshape(71684,256,23).astype('float32')

yy = data.iloc[:,-2]

yy
#%%
y = yy[yy.index % 256 == 0] 


#%%
df_y = keras.utils.to_categorical(y,num_classes=2)
df_x = np.array(df_x)
df_y = np.array(df_y)
df_y
    
#%%
#test train split

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#%%

adda = keras.optimizers.SGD(lr=0.01, decay=0.001)
#%%

#CNN model
model = Sequential()
model.add(Convolution1D(4, 6, input_shape=(256, 23)))
model.add(LeakyReLU())
model.add(MaxPooling1D(2, 2))
model.add(Convolution1D(4, 5))
model.add(LeakyReLU())
model.add(MaxPooling1D(2, 2))
model.add(Convolution1D(10, 2))
model.add(LeakyReLU())
model.add(MaxPooling1D(1, 1))
model.add(Convolution1D(10, 1))
model.add(LeakyReLU())
model.add(MaxPooling1D(1, 1))
model.add(Convolution1D(10, 1))
model.add(LeakyReLU())
model.add(MaxPooling1D(1, 1))
model.add(Flatten())
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dense(20))
model.add(LeakyReLU())
#model.add(Dropout(0.5))
model.add(Dense(2))#,kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = adda , metrics = ['accuracy'])
model.summary()
#%%
model.fit(x_train,y_train,epochs = 3,batch_size=50,validation_data=(x_test,y_test),verbose=1)
#%%
model.evaluate(x_test,y_test)
#%%
def f1_score(cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1_score = 2*((precision*recall)/(precision+recall))
    return F1_score
#%%

# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.1)

# Making the Confusion Matrx
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


print(cm)
print('F1_score =    {:.3f}'.format(f1_score(cm)))
#%%
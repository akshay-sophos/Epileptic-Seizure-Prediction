import numpy as np
from numpy import genfromtxt


title = '21_10'

X = np.array(0,np.float64)
dat = genfromtxt('chb'+title+'.csv',dtype =(float), delimiter=',')
numrows = len(dat)
numcols = len(dat[0])
j = 0
while j<(numcols-1):
 i = 0
 while i<numrows:
  if ((i == 0)&(j == 0)):
   X = dat[i:i+256,j].transpose()
   yd = dat[i,-2]
   yp = dat[i,-1]
  else :
   X = np.vstack((X,dat[i:i+256,j].transpose()))
   yd = np.vstack((y,dat[i,-2]))
   yp = np.vstack((y,dat[i,-1]))
  i = i+256
 j = j+1
 print(j)
print("Saving")

np.save('./X'+title+'.npy', X)
np.save('./Yp'+title+'.npy', yp)
np.save('./Yd'+title+'.npy', yd)

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
   y = dat[i,23]
  else :
   X = np.vstack((X,dat[i:i+256,j].transpose()))
   y = np.vstack((y,dat[i,23]))
  i = i+256
 j = j+1
 print(j)
print("Saving")

np.save('./X'+title+'.npy', X)
np.save('./Y'+title+'.npy', y)

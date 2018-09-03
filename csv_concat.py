import csv
import os
import numpy as np
a = np.empty((1,23))
patient='01_'
hours = 42
for i in range(hours):
    fname = 'chb'+patient'+str(i)+'.csv'
    with open(fname) as myFile:
        reader = csv.reader(myFile)
        for row in reader:
            a = np.vstack((a,row))
a = np.delete(a, 0,0)
print a

myFile = open('final.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(a)

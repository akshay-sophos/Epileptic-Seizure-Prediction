import csv
import numpy as np
a = np.empty((1,23))
patient='01'
hours = 42
for i in range(hours+1):
    fname = './chb_'+patient+'/chb_'+patient+'_'+str(i)+'.csv'
    with open(fname) as myFile:
        reader = csv.reader(myFile)
        for row in reader:
            a = np.vstack((a,row))
a = np.delete(a,0,0)
myFile = open('final.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(a)

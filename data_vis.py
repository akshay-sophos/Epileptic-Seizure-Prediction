#import matplotlib.pyplot as plt 
from numpy import genfromtxt
import numpy as np
dat = genfromtxt('chb_01_1.csv',dtype =(float), delimiter=',')
# %% Cell 1
#print(dat[:,0])
#plt.plot(dat[:,0])
#plt.show()
from bokeh.plotting import figure, show
p = figure(title="line", plot_width=2850, plot_height=900)
p.line(x=np.arange(10000),y=dat[:10000,0])
show(p)

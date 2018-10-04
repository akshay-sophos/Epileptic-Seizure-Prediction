import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import kurtosis
from scipy.stats import skew
import math

def get_hilbert_transform(array):
    hilbert_transform = np.imag(hilbert(array))
    return(hilbert_transform)

def get_timedomain_stat_features(array1):
    features = []
    array = np.absolute(array1)
    features.append(np.mean(array))
    features.append(np.var(array))
    features.append(skew(array))
    features.append(kurtosis(array))
    features.append((math.sqrt(features[1])// features[0]))
    features.append(np.mean(np.absolute(array1 - np.mean(array1))))
    features.append(np.sqrt(np.mean(array**2)))
    features.append(array1[(3*(len(array1) + 1) // 4)] - array1[(len(array1) + 1) // 4])
    return(features)

x = [1,1,1,1,13,3,3,3,3,5,5,5,5,7,7,77,7]
y = get_hilbert_transform(x)
z = get_timedomain_stat_features(y)
print(z)


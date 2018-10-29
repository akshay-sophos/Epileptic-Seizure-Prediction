import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.fftpack import fft
import math

# All time_domain_features and frequency_domain_features requires hilbert_transform(ht) and analytic_signal(as) of the given eeg signal(array) respectively.

def get_ht_or_as(array,i):
    
# hilbert(array) gives the analytic signal in which real part is the actual signal and imaginary part is the hilbert transform of input

    if i==1 :
        return(hilbert(array))

    if i==2 :
        return(np.imag(hilbert(array)))

def get_entropyOfGivenArray(array):
    z = np.absolute(array)
    entropy = 0
    for i in range(len(array)):
        entropy = entropy + (z[i]*(math.log(z[i],2)))
    return(-1*entropy)
        
    
def get_time_domain_features(array1):
    
    features = []
    array = np.absolute(array1)
    
    # Features Based on stastical moments
    # Mean, Variance, Skewness, Kurtosis, Coefficient of variation of EEG Signal

    features.append(np.mean(array))
    features.append(np.var(array))
    features.append(skew(array))
    features.append(kurtosis(array))
    features.append((math.sqrt(features[1])// features[0]))

    # Features Based on Amplitude
    # Median absolute deviation of EEG Amplitude, Root Mean Square Amplitude, Inter-Quartile Range

    features.append(np.mean(np.absolute(array1 - np.mean(array1))))
    features.append(np.sqrt(np.mean(array**2)))
    features.append(array1[(3*(len(array1) + 1) // 4)] - array1[(len(array1) + 1) // 4])

    # Features Based on Entropy
    # Shanon Entropy
    features.append(get_entropyOfGivenArray(array))
    
    return(features)

# input array is given in time domain later in the function we will convert it into frequency domain

def get_freq_domain_features(array2):

    # Applying Fourier transform
    
    signal = fft(array2)
    signal_abs = np.absolute(signal)
    signal_abs_sum = (len(signal_abs)*(np.mean(signal_abs)))
    features1 = []
    
    # Features Based on Power Spectrum
    # Need some discussion here

    # Features Based on Special Information
    # Spectral Centroid, Spectral Roll-ff(k need to mention), Spectral flatness
    
    c = 0
    for i in range(len(signal_abs)):
        c = c + ((i+1)*(signal_abs[i]))

    features1.append(c // signal_abs_sum)
    features1.append(k*c)

    d = 1
    for i in range(len(signal)):
        d = ((d)*(signal[i]))

    e = d**(1//(len(signal)))
    
    features1.append(e*(1//(len(signal) * np.mean(signal))))

    # Features Based on Entropy
    # Special Entropy
    en = 0
    for i in range(len(signal_abs)):
        en = en + ((signal_abs[i] ** 2)*(np.log(signal_abs[i] ** 2)))

    features1.append((en // (math.log(len(signal_abs)))))

    return (features1)
    

x = [1,1,1,1,13,3,3,3,3,5,5,5,5,7,7,77,7]
y = get_ht_or_as(x,2)
k = get_ht_or_as(x,1)
l = get_time_domain_features(k)
z = get_freq_domain_features(y)
print(l)
print(z)



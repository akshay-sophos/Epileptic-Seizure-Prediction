import numpy as np # linear algebra
import math
import pandas as pd
from numpy import genfromtxt
from features import get_time_domain_features
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

def estimateGaussian(X):
    mu = X.mean()

    m, n = X.shape # number of training examples, number of features

    sigma = np.zeros((n,n))

    for i in range(0,m):
        sigma = sigma + (X.iloc[i] - mu).values.reshape(n,1).dot((X.iloc[i] - mu).values.reshape(1, n))

    sigma = sigma * (1.0/m) # Use 1.0 instead of 1 to force float conversion

    return mu, sigma

def multivariateGaussian(X, mu, sigma):
    
    m, n = X.shape # number of training examples, number of features

    X = X.values - mu.values.reshape(1,n) # (X - mu)

    # vectorized implementation of calculating p(x) for each m examples: p is m length array
    p = (1.0 / float((math.pow((2 * math.pi), n / 2.0) * math.pow(np.linalg.det(sigma),0.5))) * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(sigma)) * X, axis=1)))

    return p

def selectThreshold(yval, pval):
    yval = np.squeeze(yval.values).astype(int)

    bestEpsilon = 0.0
    bestF1 = 0.0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        predictions = (pval < epsilon).astype(int)

        tp = np.sum((predictions == 1).astype(int) & (yval == 1).astype(int))
        fp = np.sum((predictions == 1).astype(int) & (yval == 0).astype(int))
        fn = np.sum((predictions == 0).astype(int) & (yval == 1).astype(int))

        # calculate precision & recall
        prec = (tp * 1.0) / (tp + fp)
        rec = (tp * 1.0) / (tp + fn)

        F1 = (2 * prec * rec) * 1.0 / (prec + rec) # calculate F1 score using current epsilon

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

def main():

    sec = 3

    dat = genfromtxt('chb_01_4.csv',dtype =(float), delimiter=',')
    numrows = len(dat)
    numcols = len(dat[0])
    j = 0
    while j<(numcols-1):
     i = 0
     while i<numrows:
      if ((i == 0)&(j == 0)):
       #X = dat[i:i+256*sec,j].transpose()
       X_f = get_time_domain_features(dat[i:i+256*sec,j].transpose())
       y = dat[i,23]
      else :
       #X = np.vstack((X,dat[i:i+256*sec,j].transpose()))
       X_f = np.vstack((X_f,get_time_domain_features(dat[i:i+256*sec,j].transpose())))
       y = np.vstack((y,dat[i,23]))
      i = i+256*sec
     j = j+1
     print(j)
     
    datas = np.hstack((X_f,y))
    data = pd.DataFrame(data=datas[0:,0:], columns=['f','f','f','f','f','f','f','val'])  
    
    # Group positive and negative examples
    negData = data.groupby('val').get_group(0)
    posData = data.groupby('val').get_group(1)

    # Give 60:20:20 split of negative examples for train, validate, test
    train, negCV, negTest = np.split(negData.sample(frac=1), [int(.6*len(negData)), int(.8*len(negData))])

    # Give 50:50 split of positive exampels for validate, test
    #posCV, posTest = np.split(posData, 2)
    posCV, posTest = train_test_split(posData, test_size = 0.5, random_state = 0)

    # Concatenate to form final cv and test set
    cv = negCV.append(posCV)
    test = negTest.append(posTest)
        
    rtrain, ctrain = train.shape
    Xtrain = train.iloc[0:rtrain-1,0:ctrain-2]
    ytrain = train.iloc[0:rtrain-1,ctrain-1]
    rcv, ccv = cv.shape
    XCV = cv.iloc[0:rcv-1,0:ccv-2]
    yCV = cv.iloc[0:rcv-1,ccv-1]
    rtest, ctest = test.shape
    Xtest = test.iloc[0:rtest-1,0:ctest-2]
    ytest = test.iloc[0:rtest-1,ctest-1]

    print ("Finished splitting data...")

    # Get parameters of gaussian distribution for every feature in Xtrain
    # mu is mean of dataset, and sigma is covariance matrix
    mu, sigma = estimateGaussian(Xtrain)

    print ("Learned mu and sigma...")

    # ptrain = multivariateGaussian(Xtrain, mu, sigma)

    pCV = multivariateGaussian(XCV, mu, sigma)

    print ("Calculated p(x)...")

    epsilon, F1 = selectThreshold(yCV, pCV)

    print ("Found best epsilon = " + str(epsilon) + ", best F1 = " + str(F1))

    ptest = multivariateGaussian(Xtest, mu, sigma) # Fit final model on test set

    predictions = (ptest < epsilon).astype(int)
    ytest = np.squeeze(ytest.values).astype(int)

    tp = np.sum((predictions == 1).astype(int) & (ytest == 1).astype(int))
    fp = np.sum((predictions == 1).astype(int) & (ytest == 0).astype(int))
    fn = np.sum((predictions == 0).astype(int) & (ytest == 1).astype(int))
    tn = np.sum((predictions == 0).astype(int) & (ytest == 0).astype(int))

    prec = (tp * 1.0) / (tp + fp)
    rec = (tp * 1.0) / (tp + fn)
    f1_score = 2*prec*rec/(prec+rec)

    print ("Precision = " + str(prec) + ", Recall = " + str(rec))
    print("F1 score = " + str(f1_score))


if __name__ == "__main__":
    main()

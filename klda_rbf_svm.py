#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np

from warnings import filterwarnings
from sklearn import svm
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score

# Disable warnings from being printed
filterwarnings('ignore')


# In[38]:


# Get the train and validation data
train = pd.read_csv("../Arcene/arcene_train.data.txt", header=None, sep=" ", usecols=range(10000))
train_labels = pd.read_csv("../Arcene/arcene_train.labels.txt", header=None)
valid = pd.read_csv("../Arcene/arcene_valid.data.txt", header=None, sep=" ", usecols=range(10000))
valid_labels = pd.read_csv("../Arcene/arcene_valid.labels.txt", header=None)

#train = pd.read_csv("../Madelon/madelon_train.data.txt", header=None, sep=" ", usecols=range(500))
#train_labels = pd.read_csv("../Madelon/madelon_train.labels.txt", header=None)
#valid = pd.read_csv("../Madelon/madelon_valid.data.txt", header=None, sep=" ", usecols=range(500))
#valid_labels = pd.read_csv("../Madelon/madelon_valid.labels.txt", header=None)


# In[39]:


def KLDA(X, X_labels, gamma, lmb):
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM RBF kernel matrix.
    
    # For RBF kernel
    K = exp(-gamma * mat_sq_dists)
 
    Karr = np.array(K, dtype=np.float)
    yarr = np.array(X_labels, dtype=np.int)

    labels = np.unique(yarr)
    n = yarr.shape[0]

    idx1 = np.where(yarr==labels[0])[0]
    idx2 = np.where(yarr==labels[1])[0]
    n1 = idx1.shape[0]
    n2 = idx2.shape[0]
    
    K1, K2 = Karr[:, idx1], Karr[:, idx2]
    
    N1 = np.dot(np.dot(K1, np.eye(n1) - (1 / float(n1))), K1.T)
    N2 = np.dot(np.dot(K2, np.eye(n2) - (1 / float(n2))), K2.T)
    N = N1 + N2 + np.diag(np.repeat(lmb, n))

    M1 = np.sum(K1, axis=1) / float(n1)
    M2 = np.sum(K2, axis=1) / float(n2)
    M = M1 - M2
    
    coeff = np.linalg.solve(N, M).reshape(-1, 1)
            
    return coeff


# In[40]:


def project(data, X, coeff, gamma):
    projected_data = np.zeros((data.shape[0], 1))
    X_arr = np.array(X)
    data_arr = np.array(data)
    for i in range(data_arr.shape[0]):
        cur_dist = np.array([np.sum((data_arr[i]-x)**2) for x in X_arr])
        cur_k = np.exp(-gamma * cur_dist)
        projected_data[i, :] = cur_k.dot(coeff)
    return projected_data    


# In[41]:


lmb = 1e-3
gamma = 1e-10
coeff = KLDA(train, train_labels, gamma, lmb)
projected_valid = project(valid, train, coeff, gamma)
projected_train = project(train, train, coeff, gamma)
clf = svm.SVC(kernel="rbf", max_iter=1000000)
clf.fit(projected_train, train_labels)
results = clf.predict(projected_valid)
print(accuracy_score(valid_labels, results))


# In[ ]:





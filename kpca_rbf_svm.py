#!/usr/bin/env python
# coding: utf-8

# In[23]:


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


# In[24]:


# Get the train and validation data for Arcene dataset
# train = pd.read_csv("../Arcene/arcene_train.data.txt", header=None, sep=" ", usecols=range(10000))
# train_labels = pd.read_csv("../Arcene/arcene_train.labels.txt", header=None)
# valid = pd.read_csv("../Arcene/arcene_valid.data.txt", header=None, sep=" ", usecols=range(10000))
# valid_labels = pd.read_csv("../Arcene/arcene_valid.labels.txt", header=None)

# Get the train and validation data for Madelon dataset
train = pd.read_csv("../Madelon/madelon_train.data.txt", header=None, sep=" ", usecols=range(500))
train_labels = pd.read_csv("../Madelon/madelon_train.labels.txt", header=None)
valid = pd.read_csv("../Madelon/madelon_valid.data.txt", header=None, sep=" ", usecols=range(500))
valid_labels = pd.read_csv("../Madelon/madelon_valid.labels.txt", header=None)


# In[25]:


def KPCA(X, k, gamma):
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM RBF kernel matrix.

    K = exp(-gamma * mat_sq_dists)

    # Normalizing the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in ascending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K_norm)

    # Obtaining the i eigenvectors (alphas) that corresponds to the i highest eigenvalues (lambdas).
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,k+1)))
    lambdas = [eigvals[-i] for i in range(1,k+1)]

    return alphas, lambdas


# In[26]:


def project(valid, X, k, gamma, alphas, lambdas):
    projected_data = np.zeros((valid.shape[0], k))
    X_arr = np.array(train)
    valid_arr = np.array(valid)
    for i in range(valid_arr.shape[0]):
        cur_dist = np.array([np.sum( (valid_arr[i]-x) ** 2) for x in X_arr])
        cur_k = np.exp(-gamma * cur_dist)
        projected_data[i, :] = cur_k.dot(alphas / lambdas)
    return projected_data   


# In[27]:


gamma = 1e-10
ks = [10, 100]

for k in ks:
    alphas, lambdas = KPCA(train, k, gamma)
    projected_valid = project(valid, train, k, gamma, alphas, lambdas)
    projected_train = project(train, train, k, gamma, alphas, lambdas)
    clf = svm.SVC(kernel="rbf", max_iter=1000000)
    clf.fit(projected_train, train_labels)
    results = clf.predict(projected_valid)
    print("For k=", k, ",", "Accuracy=", accuracy_score(valid_labels, results))


# In[ ]:





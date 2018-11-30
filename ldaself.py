#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as lg 
get_ipython().magic(u'matplotlib inline')


# In[2]:


df = pd.read_csv('SCLC_study_output_filtered_2.csv',header=None)


# In[13]:


df1 = df.drop(df.index[0])

df2 = df1.drop(df.columns[0], axis=1)

df3 = df2

df3_1 = df2.values[0:20,:]
df3_2 = df2.values[20:, : ]

m_1 = df3_1.mean(axis = 0)
m_2 = df3_2.mean(axis = 0)
mean_all = df2.mean(axis = 0)

mean_1 = m_1.reshape(1,19)
mean_1 = np.repeat(mean_1,20,axis = 0)

mean_2 = m_2.reshape(1,19)
mean_2 = np.repeat(mean_2,20,axis = 0)

within_class_scatter = np.zeros((19,19))
wcs_1 = np.zeros((19,19))
wcs_1 = np.matmul((np.transpose(df3_1 - mean_1 )), (df3_1 - mean_1))

wcs_2 = np.zeros((19,19))
wcs_2 = np.matmul((np.transpose(df3_2 - mean_2 )), (df3_2 - mean_2))

within_class_scatter = np.add(wcs_1,wcs_2)

bcs_1 = np.multiply(len(df3_1),np.outer((m_1 - mean_all),(m_1 - mean_all)))
bcs_2 = np.multiply(len(df3_2),np.outer((m_2 - mean_all),(m_2 - mean_all)))

between_class_scatter = np.add(bcs_1,bcs_2)

e_val, e_vector = np.linalg.eig(np.dot(lg.inv(within_class_scatter),between_class_scatter))
for e in range (len(e_val)):
    e_scatter = e_vector[:,e].reshape(19,1)

    print(e_val[e].real)

print(between_class_scatter)


eig_pairs = [(np.abs(e_val[i]).real, e_vector[:,i].real) for i in range(len(e_val))]


eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)


# sorting eigenvalues 
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

W= eig_pairs[0][1].reshape(19,1)

W

lda_project = np.dot(df2,W)

lda_project


# In[177]:

#plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('LDA')
ax.plot(lda_project[0:20], np.zeros(20), linestyle='None', marker='o', color='blue', label='NSCLC')
ax.plot(lda_project[20:40], np.zeros(20), linestyle='None', marker='o', color='red', label='SCLC')
fig.show()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


y1_ = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 

# LDA

sklearn_lda = LDA(n_components=1)
X_lda_sklearn = sklearn_lda.fit_transform(df2, y1_)

X_lda_sklearn= -X_lda_sklearn

print(X_lda_sklearn)

#plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('LDA')
ax.plot(X_lda_sklearn[0:20], np.zeros(20), linestyle='None', marker='o', color='blue', label='NSCLC')
ax.plot(X_lda_sklearn[20:40], np.zeros(20), linestyle='None', marker='o', color='red', label='SCLC')
fig.show()


# In[12]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(df2, y1_)
#print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(lda_project[0:20], lda_project[20:40])))
#print('Accuracy of SVM classifier on test set: {:.2f}'
print(svm.score(df2, y1_))


# In[7]:


# In[185]:



# In[ ]:





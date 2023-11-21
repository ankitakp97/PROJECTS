#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris_datasets=load_iris()


# In[4]:


iris_datasets


# In[5]:


iris_data=pd.DataFrame(iris_datasets.data)


# In[6]:


iris_data


# In[7]:


iris_data.head()


# In[8]:


iris_data.shape


# In[10]:


print(iris_datasets.DESCR)


# In[14]:


print(iris_datasets.feature_names)


# In[16]:


print(iris_datasets.target)


# In[17]:


X_feature=iris_datasets.data


# In[18]:


Y_target=iris_datasets.target


# In[19]:


X_feature.shape


# In[20]:


Y_target.shape


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[29]:


print(knn)


# In[30]:


knn.fit(X_feature,Y_target)


# In[31]:


X_new=[[3,5,4,1],[5,3,4,2]]


# In[33]:


knn.predict(X_new)


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


logReg=LogisticRegression()


# In[36]:


logReg.fit(X_feature,Y_target)


# In[37]:


logReg.predict(X_new)


# In[38]:


# unsupervised learning_K Means
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[39]:


n_samples=300
random_state=30
X,y=make_blobs(n_samples=n_samples,n_features=5,random_state=None)


# In[40]:


predict_y=KMeans(n_clusters=3,random_state=random_state).fit_predict(X)
predict_y


# In[41]:


# PCA
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


# In[43]:


n_sample=20
random_state=20


# In[45]:


blobs=make_blobs()


# In[46]:


blobs


# In[53]:


X,y=make_blobs(n_samples=n_sample,n_features=10,random_state=None)


# In[54]:


X.shape


# In[68]:


pca=PCA(n_components=5)


# In[69]:


pca.fit(X)
print(pca.explained_variance_ratio_)


# In[71]:


first_pca=pca.components_[4]
print(first_pca)


# In[72]:


pca_reduced=pca.transform(X)


# In[73]:


pca_reduced.shape


# In[ ]:





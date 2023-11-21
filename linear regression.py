#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.datasets import load_boston
boston_dataset=load_boston()


# In[3]:


df_boston=pd.DataFrame(boston_dataset.data)
df_boston.columns=boston_dataset.feature_names


# In[4]:


df_boston['Price']=boston_dataset.target


# In[5]:


df_boston.head()


# In[6]:


x_features=boston_dataset.data


# In[7]:


y_target=boston_dataset.target


# In[8]:


from sklearn.linear_model import LinearRegression
linReg=LinearRegression()


# In[9]:


linReg.fit(x_features,y_target)


# In[10]:


print('The estimated intercept is: %2f' %linReg.intercept_)


# In[11]:


print('The coefficient is: %d' %len(linReg.coef_))


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x_features,y_target)


# In[14]:


print(boston_dataset.data.shape)


# In[15]:


print(x_test.shape,x_train.shape,y_train.shape,y_test.shape)


# In[16]:


linReg.fit(x_train,y_train)


# In[17]:


print('The mean squared error(MSE):% 2f' % np.mean((linReg.predict(x_test)-y_test)**2))


# In[18]:


print('Variance score:% 2f' %linReg.score(x_test,y_test))


# In[19]:


x_features.shape


# In[ ]:





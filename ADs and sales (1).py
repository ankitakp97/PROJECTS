#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The given dataset contains ad budgets for different media channels and the corresponding ad sales of XYZ firm. Evaluate the dataset to:

# Find the features or media channels used by the firm
# Find the sales figures for each channel
# Create a model  to predict the sales outcome
# Split as training and testing datasets for the model
# Calculate the Mean Square Error (MSE)


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


adbudget=pd.read_csv(r"C:\Users\akpat\OneDrive\Desktop\Advertising Budget and Sales.csv",index_col=0)


# In[4]:


adbudget


# In[5]:


X_features=adbudget[['TV Ad Budget ($)','Radio Ad Budget ($)','Newspaper Ad Budget ($)']]


# In[6]:


X_features.shape


# In[7]:


Y_target=adbudget[['Sales ($)']]


# In[8]:


Y_target.shape


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


LinReg=LinearRegression()


# In[11]:


LinReg.fit(X_features,Y_target)


# In[12]:


print('The estimated intercept is: %2f' %LinReg.intercept_)


# In[13]:


print('The coefficient is: %d' %len(LinReg.coef_))


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_features,Y_target,random_state=1)


# In[15]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[16]:


LinReg.fit(x_train,y_train)


# In[17]:


print("The mean squared error is:%2f" % np.mean((LinReg.predict(x_test)-y_test)**2))


# In[18]:


print('Variance score:% 2f' %LinReg.score(x_test,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





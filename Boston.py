#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style


# In[2]:


from sklearn.datasets import load_boston
boston_dataset=load_boston()


# In[3]:


boston_data=pd.DataFrame(boston_dataset.data)


# In[4]:


boston_data


# In[5]:


boston_data.columns=boston_dataset.feature_names


# In[6]:


boston_data['Price']=boston_dataset.target


# In[7]:


boston_data.head()


# In[8]:


X_features=boston_dataset.data
Y_target=boston_dataset.target


# In[9]:


from sklearn.linear_model import LinearRegression
LinReg=LinearRegression()


# In[10]:


LinReg.fit(X_features,Y_target)


# In[11]:


print('The estimated inetrcept is: %2f'% LinReg.intercept_)


# In[12]:


print('The Coefficient is: %d' %len(LinReg.coef_))


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_features,Y_target)


# In[14]:


print(boston_dataset.data.shape)


# In[15]:


print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[16]:


LinReg.fit(X_train,Y_train)


# In[17]:


print('The mean squared error: %2f' % np.mean((LinReg.predict(X_test)-Y_test)**2))


# In[18]:


print('The variance score: %2f'% LinReg.score(X_test,Y_test))


# In[19]:


x_axis=X_features
y_axis=Y_target


# In[20]:


style.use('ggplot')


# In[21]:


plt.figure(figsize=(7,7))


# In[22]:


plt.hist(y_axis,bins=50)
plt.xlabel('price in 1000s USD')
plt.ylabel('Number of houses')
plt.show()


# In[23]:


plt.scatter(X_features[:,5],Y_target)
plt.xlabel('price in 1000s USD')
plt.ylabel('Number of houses')
plt.show()


# In[32]:


LinReg.fit(X_test,Y_test)
prediction=LinReg.predict(X_test)


# In[33]:


prediction


# In[34]:


from sklearn.metrics import mean_squared_error
from math import sqrt
print(sqrt(mean_squared_error(Y_test,prediction)))


# In[ ]:


# if the mean squared error (MSE) is 4, it indicates the average squared difference between the predicted values and the actual values in a regression or prediction model.

# The MSE is a measure of the model's accuracy, where a lower value indicates a better fit to the data. In this case, an MSE of 4 suggests that, on average, the predicted values differ from the actual values by approximately 4 units squared.

# It's important to note that the interpretation of the MSE depends on the specific context and scale of the data. For example, if the predicted values represent housing prices in thousands of dollars, an MSE of 4 would imply an average squared difference of $4,000 between the predicted and actual prices.

# In summary, an MSE of 4 indicates the average squared difference between predicted and actual values in a model, reflecting the model's accuracy in capturing the underlying patterns in the data.


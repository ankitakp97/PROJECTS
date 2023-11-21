#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv(r"C:\Users\akpat\OneDrive\Desktop\Salaries.csv")


# In[3]:


data


# In[4]:


data.isnull().any()


# In[5]:


data.head()


# In[6]:


data.describe()


# In[31]:


print(data.groupby('Year').apply(lambda x: x[x['TotalPay'] == x['TotalPay'].max()]))


# In[12]:


# Top Earners 
# NATHANIEL FORD =2011
# Gary Altenberg   :2012
# Samson  Lai   :2013
# David Shinn :2014


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


sns.boxplot(x=data['BasePay'])


# In[41]:


filter=data['BasePay'].values<200000
data_outlier_rem=data[filter]


# In[42]:


data_outlier_rem


# In[43]:


data.isnull().any()


# In[46]:


data_outlier_rem.describe()


# In[ ]:





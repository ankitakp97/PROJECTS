#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Building a model to predict Diabetes
# DESCRIPTION

# Problem:

# The given dataset lists the glucose level readings of several pregnant women taken either during a survey examination or routine medical care. It specifies if the 2-hour post-load plasma glucose was at least 200 mg/dl. Analyze the dataset to:

# Find the features of the dataset,
# Find the response label of the dataset,
# Create a model  to predict the diabetes outcome,
# Use training and testing datasets to train the model, and
# Check the accuracy of the model.


# In[1]:


import numpy as np
import pandas as pd


# In[2]:


column_names=["Number of times pregnant",
    "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
   "Diastolic blood pressure (mm Hg)",
    "Triceps skin fold thickness (mm)",
   "2-Hour serum insulin (mu U/ml)",
    "Body mass index (weight in kg/(height in m)^2)",
   "Diabetes pedigree function",
    "Age (years)",
   "Class variable (0 or 1)"]


# In[3]:


medical=pd.read_csv(r"C:\Users\akpat\OneDrive\Desktop\pima-indians-diabetes.data",names=column_names)


# In[4]:


medical


# In[5]:


medical.head()


# In[6]:


m_columns=medical.columns


# In[7]:


m_columns


# In[8]:


feature_colms=['Number of times pregnant','2-Hour serum insulin (mu U/ml)',
       'Body mass index (weight in kg/(height in m)^2)','Age (years)']


# In[9]:


X_features=medical[feature_colms]
Y_target=medical['Class variable (0 or 1)']
print(X_features.shape)
Y_target.shape


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_features,Y_target)


# In[14]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[15]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)


# In[16]:


y_pred=logreg.predict(x_test)


# In[18]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:





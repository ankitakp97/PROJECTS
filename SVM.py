#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data=pd.read_csv(r"C:\Users\akpat\Downloads\voice-classification.csv")


# In[4]:


data


# In[6]:


data.head()


# In[8]:


data.shape


# In[23]:


X=data.iloc[:, :-1]
X.shape


# In[32]:



from sklearn.preprocessing import LabelEncoder
y=data.iloc[:,-1]
gender_encoder=LabelEncoder()
y=gender_encoder.fit_transform(y)
y


# In[37]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X)
X=sc.transform(X)


# In[55]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=20)


# In[56]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report


# In[57]:


svc_model=SVC()


# In[58]:


svc_model.fit(x_train,y_train)
y_pred=svc_model.predict(x_test)


# In[59]:


y_pred


# In[60]:


print(metrics.accuracy_score(y_pred,y_test))


# In[61]:


cm=confusion_matrix(y_test,y_pred)


# In[62]:


cm


# In[65]:


# 466+465=Correct predictions


# In[64]:


print(classification_report(y_test,y_pred))


# In[ ]:





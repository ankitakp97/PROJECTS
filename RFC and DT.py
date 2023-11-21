#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv(r"C:\Users\akpat\OneDrive\Desktop\loan_borowwer_data.csv")


# In[4]:


data


# In[7]:


data.info()


# In[6]:


data.head()


# In[8]:


data['purpose'].unique()


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,6))
data[data['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='credit.policy=1')
data[data['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='credit.policy=0')
plt.legend()
plt.xlabel('FICO')


# In[11]:


plt.figure(figsize=(10,6))
data[data['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
data[data['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[31]:


df=data.copy()
df


# In[32]:


labels = 'FULLY PAID','NOT FULLY PAID'
sizes = [df.not_fully_paid[df['not_fully_paid']==0].count(), df.not_fully_paid[df['not_fully_paid']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer with fully paid and fully not paid", size = 20)
plt.show()


# In[33]:


labels = df['purpose'].astype('category').cat.categories.tolist()
counts = df['purpose'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
explode = (0.1, 0, 0, 0, 0,0,0)
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[34]:


cat_feats=['purpose']


# In[35]:


final_data=pd.get_dummies(data,columns=cat_feats,drop_first=True)
final_data.info()


# In[37]:


X=final_data.drop('not_fully_paid',axis=1)
X.describe()


# In[38]:


y = final_data['not_fully_paid']


# In[39]:


y


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=101)


# In[ ]:


# Training Decision Tree Model


# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[46]:


dtree=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree.fit(X_train,y_train)


# In[47]:


predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[48]:


print(confusion_matrix(y_test,predictions))


# In[49]:


#Training Random Forest Model:
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=600,criterion='entropy')
rfc.fit(X_train,y_train)


# In[51]:


pred=dtree.predict(X_test)


# In[52]:


print(classification_report(y_test,pred))


# In[53]:


print(confusion_matrix(y_test,pred))


# In[54]:


print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))


# In[55]:


2010+93


# In[56]:


2003+100


# In[ ]:





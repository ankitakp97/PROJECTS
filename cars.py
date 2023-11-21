#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np  


# In[2]:


df=pd.read_csv(r"C:\Users\akpat\OneDrive\Desktop\mtcars.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df.head()


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


df.hist(bins=30,figsize=(5,10))


# In[10]:


df.info()


# In[11]:


df['hp'].value_counts()


# ## train test split
# 

# In[12]:


def split_train_test(data,test_ratio):
    np.random.seed(13)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[ :test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[13]:


train_set,test_set=split_train_test(df,0.22)


# In[14]:


print(f"Rows in train set:{len(train_set)}\n Rows in test set:{len(test_set)}\n")


# In[15]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\n Rows in test set:{len(test_set)}\n")


# In[16]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(df,df['vs'],df['am']):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]


# In[17]:


strat_test_set['vs'].value_counts()


# In[18]:


strat_test_set['am'].value_counts()


# In[19]:


strat_train_set['vs'].value_counts()


# In[20]:


4/3


# In[21]:


14/11


# In[22]:


strat_train_set['am'].value_counts()


# In[23]:


5/2


# In[24]:


14/11


# In[25]:


corr_matrix=df.corr()


# In[26]:


corr_matrix['hp'].sort_values(ascending=False)


# In[27]:


from pandas.plotting import scatter_matrix
attributes=['hp','cyl','disp','wt','mpg','gear']
scatter_matrix(df[attributes],figsize=(12,8))


# In[28]:


df.plot(kind='scatter',x='hp',y='cyl',alpha=0.8)


# In[29]:


import random
for i in range(1):
    random.seed(42)
    print(random.randint(1, 1000))
    


# In[30]:


import seaborn as sns
correlations=df.corr()
sns.heatmap(data=correlations,square=True,cmap="bwr")
plt.yticks(rotation=0)
plt.xticks(rotation=270)


# In[31]:


len(df)


# In[32]:


df.isnull().any()


# In[34]:


import seaborn as sns

sns.boxplot(x=df['hp'])
plt.show()


# In[ ]:





# In[ ]:





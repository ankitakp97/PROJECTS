#!/usr/bin/env python
# coding: utf-8

# In[12]:


from datetime import datetime
current_time=datetime.now()
print(current_time)
current_hour = int(datetime.now().strftime('%H'))
if current_hour<12:
    print('Good morning')
elif 12<=current_hour<18:
    print('Good afternoon')
else:
    print('Good Evening')


# In[13]:


n=input("Enter a number, ")


# In[28]:


for k in range(1,12, ):
   print(k)


# In[31]:


i=0
while(i<=3):
    print(i)
    i=i+1


# In[34]:


count=5
while (count >0):
    print(count)
    count=count-1


# In[38]:


for i in range(12):
    print("5 X",i+1,"=",5* (i+1))
    if(i==10):
      continue


# In[43]:


def isGreater(a,b):
    if(a>b):
        print("first is greater")
        
    else:
        print("second is greater or equal")
            


# In[45]:


a=3
b=7
isGreater(a,b)


# In[ ]:





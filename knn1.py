#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[2]:


dataset=pd.read_csv("Social_Network_Ads.csv")


# In[3]:


dataset


# In[4]:


dataset.info()


# In[5]:


dataset.head()


# In[6]:


dataset.columns


# In[7]:


X=dataset[['Age','EstimatedSalary']]


# In[8]:


y=dataset["Purchased"]


# In[9]:


import seaborn as sns


# In[10]:


sns.set()


# In[11]:


sns.scatterplot(x="Age",y="EstimatedSalary",data=dataset,hue="Purchased")


# In[12]:


type(X)


# In[13]:


X=X.values


# In[14]:


type(X)


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[17]:


X_train


# In[18]:


X_test


# In[ ]:





# In[ ]:





# In[19]:


from sklearn.neighbors import KNeighborsClassifier


# In[67]:


model=KNeighborsClassifier(n_neighbors=3)


# In[68]:


model.fit(X_train,y_train)


# In[69]:


model.predict([[61,200000]])


# In[70]:


y_pred=model.predict(X_test)


# In[71]:


y_pred


# In[72]:


y_test


# In[73]:


from sklearn.metrics import confusion_matrix


# In[74]:


confusion_matrix(y_test,y_pred)


# In[75]:


from sklearn.metrics import accuracy_score


# In[76]:


accuracy_score(y_test,y_pred)


# In[ ]:





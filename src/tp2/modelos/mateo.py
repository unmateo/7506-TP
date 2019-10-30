#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)
from modelo import Modelo
from datos import levantar_datos


# In[2]:


train, test = levantar_datos()


# In[3]:


test.columns


# In[8]:


import matplotlib as plt
test.fecha.sort_values().hist(bins=60)


# In[9]:


from sklearn.metrics import mean_absolute_error
from sklearn.


# In[10]:


train.fecha.sort_values().hist(bins=60)


# In[11]:





# In[20]:


train, test = train_test_split(train)


# In[22]:


train.shape


# In[23]:


test.shape


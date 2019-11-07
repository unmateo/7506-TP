#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)

from modelo import Modelo
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[2]:





# In[3]:


modelo = Modelo()


# In[4]:


modelo.cargar_datos()


# In[6]:


modelo.submit_data.count().sort_values()


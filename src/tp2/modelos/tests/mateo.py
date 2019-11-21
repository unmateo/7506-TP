#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)
xgboost_path = path + "/xgboost_regressor"
promedios_path = path + "/promedio_zona"

import pandas as pd
import matplotlib
from datos import FEATURES_DISPONIBLES
from modelo import Modelo

pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[2]:


modelo = Modelo()


# In[3]:


modelo.cargar_datos()


# In[5]:


modelo.train_data.drop()


# In[ ]:


modelo.train_data['antiguedad'].describe()


# In[ ]:


modelo.train_data['antiguedad'].value_counts()


# In[ ]:


modelo.submit_data['metrostotales'].describe()


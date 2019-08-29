#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
module_path = os.path.abspath(os.path.join('..','..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.dataset_parsing import levantar_datos, DATASET_RELATIVE_PATH


# In[2]:


df = levantar_datos("../../"+DATASET_RELATIVE_PATH)


# In[ ]:





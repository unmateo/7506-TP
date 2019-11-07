#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)

from datos import FEATURES_DISPONIBLES
from modelo import Modelo
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[10]:





# In[5]:


import xgboost as xgb


# In[17]:


class XGBoost(Modelo):
    """
    """
    
    def cargar_datos(self):
        """
        """
        excluir = {
            "tipodepropiedad", "ciudad", "provincia", "idzona", "fecha"
        }
        features = FEATURES_DISPONIBLES - excluir
        super().cargar_datos(features)
        self.train_data = self.train_data.drop(columns=["fecha"])
        self.test_data = self.test_data.drop(columns=["fecha"])
        self.submit_data = self.submit_data.drop(columns=["fecha"])
        
    def entrenar(self):
        """
        """
        pass
        


# In[18]:


modelo = XGBoost()
modelo.cargar_datos()


# In[16]:


modelo.train_data.head()


# In[27]:


label = 'gimnasio'
dtrain = xgb.DMatrix(modelo.train_data, label=modelo.train_data[label].values)
dtest = xgb.DMatrix(modelo.test_data, label=modelo.test_data[label].values)


# In[28]:


params = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': 'auc'}
booster = xgb.train(params, dtrain)


# In[46]:


modelo.test_data['pred'] = booster.predict(dtest)


# In[47]:


pred = modelo.test_data[[label, 'pred']]


# In[48]:


pred["pred"].hist()


# In[49]:


pred["label"] = pred["pred"].map(lambda x: x > 0.5)


# In[57]:


pred["dif"] = ~(pred[label] == pred["label"])


# In[58]:


pred["dif"].value_counts()


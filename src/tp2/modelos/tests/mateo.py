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


# In[4]:


df = modelo.test_data


# # Campos Faltantes
# 
# Análisis de qué campos tienen valores para cada df (train test submit)

# In[13]:


todo = modelo.test_data.append(modelo.train_data, sort=False).append(modelo.submit_data, sort=False)


# In[24]:


todo.isna().sum().sort_values(ascending=False)


# In[39]:


len(todo.banos.unique()) - 1 


# In[ ]:


{'antiguedad', 'garages', 'banos', 'habitaciones'}


# # Evaluación de modelo
# 
# Análisis de en qué propiedades hay mayor diferencia entre real/esperado.
# En xgboost da que hay mucha diferencia en los NA

# In[ ]:


predicciones = modelo.predecir(modelo.test_data)
predicciones['dif'] = abs(predicciones['precio'] - predicciones['target']) / predicciones['precio']
peores_predicciones = modelo.test_data.loc[predicciones.sort_values(by='dif').tail(100).index]
mejores_predicciones = modelo.test_data.loc[predicciones.sort_values(by='dif').head(100).index]
def plot_dif(feature):
    plot = peores_predicciones[feature].hist(color='red')
    plot = mejores_predicciones[feature].hist(alpha=0.5, color='green')
    print('peores')
    print(peores_predicciones[feature].isna().value_counts(normalize=True))
    print('mejores')
    print(mejores_predicciones[feature].isna().value_counts(normalize=True))

plot_dif('banos')
plot_dif('garages')
plot_dif('metrostotales')
plot_dif('metroscubiertos')


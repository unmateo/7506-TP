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


from xgboost_regressor.xgboost_predictor import XGBoostRegressor
from promedio_zona.promedio_zona import PromedioZona


# In[7]:


class EnsamblePromedioXGBoostConcatenados(XGBoostRegressor):
    """
        Usa el resultado de las predicciones del modelo PromedioZona
        para entrenar y predecir con un XGBoostRegressor.
    """
    
    
    @Modelo.cronometrar()
    def __init__(self):
        self.modelo_promedios = PromedioZona()
        super().__init__()        
        
    @Modelo.cronometrar()
    def cargar_datos(self):
        self.modelo_promedios.cargar_datos()
        super().cargar_datos()
    
    @Modelo.cronometrar()
    def entrenar(self):
        self.modelo_promedios.entrenar()
        predicciones_train =  self.modelo_promedios.predecir(self.modelo_promedios.train_data)
        predicciones_test = self.modelo_promedios.predecir(self.modelo_promedios.test_data)
        predicciones_submit = self.modelo_promedios.predecir(self.modelo_promedios.submit_data)
        self.train_data['prediccion_promedios'] = predicciones_train['target']
        self.test_data['prediccion_promedios'] = predicciones_test['target']
        self.submit_data['prediccion_promedios'] = predicciones_submit['target']
        super().entrenar()


# In[8]:


ensamble = EnsamblePromedioXGBoostV2()


# In[9]:


ensamble.cargar_datos()


# In[10]:


ensamble.entrenar()


# In[11]:


ensamble.validar()


# In[12]:


predicciones = ensamble.predecir(ensamble.submit_data)


# In[13]:


comentario = "ensamble promedios + xgboost (concatenados)"
ensamble.presentar(predicciones, comentario)


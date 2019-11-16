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


# In[3]:


class EnsamblePromedioXGBoost(Modelo):
    """
        En principio, voy a poner un promedio de ambas predicciones.
    """
    
    @Modelo.cronometrar()
    def __init__(self):
        self.modelo_promedios = PromedioZona()
        self.modelo_xgboost = XGBoostRegressor()
        super().__init__()

    @Modelo.cronometrar()
    def cargar_datos(self):
        self.modelo_promedios.cargar_datos()
        self.modelo_xgboost.cargar_datos()
        self.cargado = True
    
    @Modelo.cronometrar()
    def entrenar(self):
        self.modelo_promedios.entrenar()
        self.modelo_xgboost.entrenar()
        super().entrenar()
    
    @Modelo.cronometrar()
    def validar(self):
        validacion_promedios = self.modelo_promedios.validar()
        validacion_xgboost = self.modelo_xgboost.validar()
        score = (validacion_promedios + validacion_xgboost) / 2
        self.resultado_validacion = score
        self.validado = True
        
    @Modelo.cronometrar()
    def predecir_submit(self):
        prediccion_promedios = self.modelo_promedios.predecir(self.modelo_promedios.submit_data)
        prediccion_xgboost = self.modelo_xgboost.predecir(self.modelo_xgboost.submit_data)
        predicciones = prediccion_promedios[['target']].join(prediccion_xgboost[['target']], lsuffix='_promedio', rsuffix='_xgboost')
        predicciones['target'] = predicciones.mean(axis='columns')
        return predicciones


# In[4]:


ensamble = EnsamblePromedioXGBoost()


# In[5]:


ensamble.cargar_datos()


# In[6]:


ensamble.entrenar()


# In[ ]:


ensamble.validar()


# In[ ]:


predicciones = ensamble.predecir_submit()


# In[22]:


comentario = "test ensamble promedios + xgboost"
ensamble.presentar(predicciones, comentario)


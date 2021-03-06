#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)


import pandas as pd
import matplotlib
from datos import FEATURES_DISPONIBLES
from modelo import Modelo

pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[2]:


from xgboost_classifier.xgboost_predictor import XGBoost
from xgboost_regressor.xgboost_predictor import XGBoostRegressor
from promedio_zona.promedio_zona import PromedioZona
from regresion_lineal.regresion_lineal import RegresionLineal


# In[3]:


class EnsambleConcatenadosPrediciendoNaNs(XGBoostRegressor):
    """
        Usa el resultado de las predicciones del modelo PromedioZona
        para entrenar y predecir con un XGBoostRegressor.
    """
    
    
    @Modelo.cronometrar()
    def __init__(self):
        self.modelo_promedios = PromedioZona()
        self.modelo_lineal = RegresionLineal()
        super().__init__()        
        
    @Modelo.cronometrar()
    def cargar_datos(self):
        self.modelo_promedios.cargar_datos()
        self.modelo_lineal.cargar_datos()
        super().cargar_datos()
    
    @Modelo.cronometrar()
    def entrenar(self):
        self.completar_nans()
        self.agregar_predicciones_modelo(self.modelo_lineal)
        self.agregar_predicciones_modelo(self.modelo_promedios)
        super().entrenar()
    
    def agregar_predicciones_modelo(self, modelo):
        columna = 'prediccion_' + modelo.modelo
        modelo.entrenar()
        predicciones_train =  modelo.predecir(modelo.train_data)
        predicciones_test = modelo.predecir(modelo.test_data)
        predicciones_submit = modelo.predecir(modelo.submit_data)
        self.train_data[columna] = predicciones_train['target']
        self.test_data[columna] = predicciones_test['target']
        self.submit_data[columna] = predicciones_submit['target']
        
    def completar_nans(self):
        """
            Completa NaNs en todos los df.
        """
        features_a_completar = ['banos', 'garages', 'habitaciones']
        dfs_a_completar = (self.train_data, self.test_data, self.submit_data)
        for feature in features_a_completar:
            modelo = XGBoost(feature)
            modelo.cargar_datos()
            modelo.entrenar()
            predicciones = modelo.predecir(modelo.submit_data)['target']
            for df in dfs_a_completar:
                df[feature].fillna(predicciones, inplace=True)


# In[4]:


ensamble = EnsambleConcatenadosPrediciendoNaNs()


# In[5]:


ensamble.cargar_datos()


# In[6]:


ensamble.entrenar()


# In[ ]:





# In[7]:


ensamble.validar()


# In[8]:


predicciones = ensamble.predecir(ensamble.submit_data)


# In[10]:


comentario = "ensamble prediciendo nans - local 622833.06"
ensamble.presentar(predicciones, comentario)


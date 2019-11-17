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
        Predice con un promedio ponderado entre los dos modelos.
    """
    
    
    @Modelo.cronometrar()
    def __init__(self):
        self.modelo_promedios = PromedioZona()
        self.modelo_xgboost = XGBoostRegressor()
        self.peso_xgboost = 0.7
        super().__init__()
        
        
    @property
    def peso_xgboost(self):
        return self.__peso_xgboost

    @peso_xgboost.setter
    def peso_xgboost(self, peso):
        """ 
            Indica qué peso asignarle al modelo xgboost al predecir.
            Podría llevar a overfitting del set de validación.
        """
        if not (0 <= peso <= 1):
            raise ValueError("peso_xgboost debe estar en [0,1]")
        self.__peso_xgboost = peso

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
        """
        """
        predicciones = self.predecir('test')
        score = self.puntuar(predicciones[self.feature], predicciones["target"])
        self.resultado_validacion = score
        self.validado = True
        return score

    @Modelo.cronometrar()
    def predecir(self, cual):
        """
            cual: {'test', 'submit'}
        """
        sets_disponibles  = {
            "test": {
                "promedios": self.modelo_promedios.test_data,
                "xgboost": self.modelo_xgboost.test_data,
            },
            "submit": {
                "promedios": self.modelo_promedios.submit_data,
                "xgboost": self.modelo_xgboost.submit_data
            }
        }
        if cual not in sets_disponibles: raise Exception('No puedo predecir eso')
        
        columnas = [self.feature, 'target'] if cual == 'test' else ['target']
        prediccion_promedios = self.modelo_promedios.predecir(sets_disponibles.get(cual).get('promedios'))[columnas]
        prediccion_xgboost = self.modelo_xgboost.predecir(sets_disponibles.get(cual).get('xgboost'))[columnas]
        predicciones = prediccion_promedios.join(prediccion_xgboost, lsuffix='_promedio', rsuffix='_xgboost')
        predicciones['target'] = predicciones[['target_promedio', 'target_xgboost']].mean(axis='columns')
        predicciones['target'] = predicciones['target_xgboost'] * self.peso_xgboost + predicciones['target_promedio'] * (1-self.peso_xgboost)
        if cual == 'test':
            predicciones[self.feature] = predicciones[[self.feature+'_promedio', self.feature+'_xgboost']].mean(axis='columns')
        return predicciones    


# In[4]:


ensamble = EnsamblePromedioXGBoost()


# In[5]:


ensamble.cargar_datos()


# In[6]:


ensamble.entrenar()


# In[7]:


ensamble.validar()


# In[8]:


predicciones = ensamble.predecir('submit')


# In[9]:


ensamble.peso_xgboost


# In[10]:


comentario = "ensamble promedios + xgboost con peso 0.7"
ensamble.presentar(predicciones, comentario)


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


# In[50]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[53]:


class XGBoost(Modelo):
    """
        Este modelo lo vamos a usar para predecir algunos
        valores faltantes en los tres sets de datos.
    """

     
    def cargar_datos(self):
        """
        """
        excluir = {
            "tipodepropiedad", "ciudad", "provincia", "idzona", "fecha",
            "precio_metro_total", "precio", "precio_metro_cubierto"
        }
        features = FEATURES_DISPONIBLES - excluir
        super().cargar_datos(features)
        self.train_data = self.train_data.drop(columns=["fecha"])
        self.test_data = self.test_data.drop(columns=["fecha"])        
        self.submit_data = self.submit_data.drop(columns=["fecha"])
        return True

    def _split_data_label(self, df, label):
        data = df.loc[:, df.columns != label]
        label = df[label].values
        return data, label
    
    @Modelo.cronometrar()
    def entrenar(self, params=None):
        """
        """
        if not params:
            params = {
                'max_depth': 2,
                'eta': 1,
                'objective': 'binary:logistic',
                'nthread': 4,
                'eval_metric': 'auc'
            }
        train_data, train_label = self._split_data_label(modelo.train_data, self.feature)
        dtrain = xgb.DMatrix(train_data, label=train_label)
        self.predictor = xgb.train(params, dtrain)
        super().entrenar()
        return True
    
    @Modelo.cronometrar()
    def predecir(self, data, to_bool=True):
        """
            to_bool: bool
                Transformar la columna target (0,1] a bool.
                Lo hace poniendo un límite en 0.5
        """
        predict_data, predict_label = self._split_data_label(data, self.feature)
        dpredict = xgb.DMatrix(predict_data)
        predictions = self.predictor.predict(dpredict)
        if to_bool:
            predictions = [ x>0.5 for x in predictions ]
        data["target"] = predictions
        return data

    
    def puntuar(self, real, prediccion):
        """
            
        """
        try:
            puntaje = super().puntuar(real, prediccion)
        except Exception:
            puntaje = accuracy_score(real, prediccion)
        return puntaje


# In[73]:


modelo = XGBoost(feature='gimnasio')
modelo.cargar_datos()
modelo.entrenar()
modelo.validar()
#predicciones = modelo.predecir(modelo.submit_data)


# In[71]:


modelo.submit_data.info()


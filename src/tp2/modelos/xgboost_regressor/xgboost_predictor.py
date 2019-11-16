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


# In[10]:


import xgboost as xgb
from sklearn.metrics import accuracy_score
from operator import concat
from functools import reduce


# In[149]:


class XGBoostRegressor(Modelo):
    """
        Este modelo lo vamos a usar para predecir algunos
        valores faltantes en los tres sets de datos.
    """

     
    def cargar_datos(self):
        """
        """
        excluir = {
            "idzona", "fecha",
            "precio_metro_cubierto", "precio_metro_total"
        }
        features = FEATURES_DISPONIBLES - excluir
        super().cargar_datos(features)
        self.train_data = self.preparar_datos(self.train_data)
        self.test_data = self.preparar_datos(self.test_data)      
        self.submit_data = self.preparar_datos(self.submit_data)
        self.agregar_columnas_faltantes()
        return True
    
    def preparar_datos(self, df):
        """
        """
        df = df.drop(columns="fecha")   
        categoricas = {"tipodepropiedad", "provincia", "ciudad"}
        return self.one_hot_encode(df, categoricas)
    
    def agregar_columnas_faltantes(self):
        """
            Al hacer one hot encoding individualemente sobre los dfs,
            puede pasar que queden con columnas dispares. Por eso,
            en esta función las agrego a cada uno.
        """
        dfs = (self.train_data, self.test_data, self.submit_data)
        columnas_todas = set(reduce(concat, [list(df.columns.values) for df in dfs], []))
        def agregar_faltantes(df):
            faltantes = list(columnas_todas - {'precio'} - set(df.columns.values))
            for faltante in faltantes:
                df[faltante] = False
            print(faltantes)
            return df.reindex(columnas_todas, axis='columns')
        self.train_data = agregar_faltantes(self.train_data)
        self.test_data = agregar_faltantes(self.test_data)
        self.submit_data = agregar_faltantes(self.submit_data)
        return True

    def _split_data_label(self, df, label):
        data = df.loc[:, df.columns != label]
        label = df[label].values if label in df.columns else None
        return data, label
    
    @Modelo.cronometrar()
    def entrenar(self, params=None):
        """
        """
        train_data, train_label = self._split_data_label(modelo.train_data, self.feature)
        self.model = xgb.XGBRegressor()
        self.model.fit(train_data, train_label)
        super().entrenar()
        return True
    
    @Modelo.cronometrar()
    def predecir(self, data):
        """
        """
        predict_data, predict_label = self._split_data_label(data, self.feature)
        predictions = self.model.predict(predict_data)
        data["target"] = predictions
        return data


# In[150]:


modelo = XGBoostRegressor()
modelo.cargar_datos()


# In[151]:


modelo.entrenar()


# In[152]:


modelo.validar()


# In[153]:


predicciones = modelo.predecir(modelo.submit_data)


# In[154]:


comentario = "xgboost regressor con one hot encoding para tipodepropiedad,provincia y ciudad - puntaje local 738739.3"
modelo.presentar(predicciones, comentario)


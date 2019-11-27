#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import xgboost as xgb
from sklearn.metrics import accuracy_score
from operator import concat
from functools import reduce
from random import choice


# In[ ]:


class XGBoostRegressor(Modelo):
    """
        Este modelo lo vamos a usar para predecir algunos
        valores faltantes en los tres sets de datos.
    """

     
    def cargar_datos(self):
        """
        """
        excluir = {
            "idzona",
            "precio_metro_cubierto",
            "precio_metro_total",
            "gps", "lat", "lng"
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
        df = df.drop(columns=["fecha", "titulo", "descripcion"]) 
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
        hiperparametros = {
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 10,
            'number_estimators': 500,
            'gamma': 0.5,
            'min_child_weight': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 1,
            'base_score': 500000
        }
        if params:
            hiperparametros.update(params)
        train_data, train_label = self._split_data_label(self.train_data, self.feature)
        self.model = xgb.XGBRegressor(**hiperparametros)
        self.model.fit(train_data, train_label)
        super().entrenar()
        return True
    
    @Modelo.cronometrar()
    def predecir(self, df):
        """
        """
        data = df.copy()
        predict_data, predict_label = self._split_data_label(data, self.feature)
        predictions = self.model.predict(predict_data)
        data["target"] = predictions
        return data


# In[ ]:


def params_to_tuple(params):
    return tuple(params.items())


# In[ ]:


def probar_parametros(modelo, params):
    modelo.entrenar(params)
    return modelo.validar()


# In[ ]:


def random_prueba(parametros):
    prueba = parametros.copy()
    return {key:choice(values) for key,values in prueba.items()}


# In[ ]:


def generar_n_pruebas(n, parametros):
    pruebas = []
    set_pruebas = set()
    while len(pruebas) < n:
        prueba = random_prueba(parametros)
        prueba_tuple = params_to_tuple(prueba)
        if prueba_tuple in set_pruebas: continue
        pruebas.append(prueba)
        set_pruebas.add(prueba_tuple)
    return pruebas


# In[ ]:


def buscar_hiperparametros():
    resultados = {}
    modelo = XGBoostRegressor()
    modelo.cargar_datos()
    cantidad_pruebas = 10
    opciones = {
        'learning_rate': [0.1, 0.01],
        'max_depth': [10, 15, 20, 25],
        'number_estimators': [500, 750, 1000, 1500],
        'gamma': [0, 0.5, 1, 2, 4],
        'min_child_weight': [5, 7, 10],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0, 0.5, 1],
        'base_score': [200000, 500000, 1000000, 2000000]
    }
    pruebas = generar_n_pruebas(cantidad_pruebas, opciones)
    for prueba in pruebas:
        print(prueba)
        puntaje = probar_parametros(modelo, prueba)
        print(puntaje)
        resultados[params_to_tuple(prueba)] = puntaje
        print(resultados)
    return resultados


# ## Mejores hiperparámetros
# {'learning_rate': 0.1, 'max_depth': 15, 'number_estimators': 500, 'gamma': 0.5, 'min_child_weight': 5, 'reg_alpha': 0.5, 'reg_lambda': 1, 'base_score': 500000}
# 
# {'learning_rate': 0.1, 'max_depth': 15, 'number_estimators': 500, 'gamma': 1, 'min_child_weight': 7, 'reg_alpha': 1, 'reg_lambda': 1, 'base_score': 2000000}
# 
# {'learning_rate': 0.1, 'max_depth': 15, 'number_estimators': 1000, 'gamma': 2, 'min_child_weight': 7, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'base_score': 1000000}
# 

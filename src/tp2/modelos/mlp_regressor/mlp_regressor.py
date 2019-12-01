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


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from operator import concat
from functools import reduce


# In[ ]:


import numpy as np
from random import choice


# In[ ]:


class MLP_Regressor(Modelo):
    """
    """

     
    def cargar_datos(self):
        """
        """
        excluir = {
            "idzona",
            "precio_metro_cubierto",
            "precio_metro_total",
            "gps", "lat", "lng",
            'ciudad', 'provincia'
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
        categoricas = {"tipodepropiedad"}
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
        self.train_data = self.llenar_nans(agregar_faltantes(self.train_data))
        self.test_data = self.llenar_nans(agregar_faltantes(self.test_data))
        self.submit_data = self.llenar_nans(agregar_faltantes(self.submit_data))
        return True

    def _split_data_label(self, df, label=None):
        if not label:
            label = self.feature
        data = df.loc[:, df.columns != label]
        label = df[label].values if label in df.columns else None
        return data, label
    
    def llenar_nans(self, df):
        return df.fillna(df.mean(skipna=True, numeric_only=True))
    
    @Modelo.cronometrar()
    def entrenar(self, params=None):
        """
        """
        data_train, label_train = self._split_data_label(self.train_data)
        self.scaler = StandardScaler()
        self.scaler.fit(data_train)
        X_train = self.scaler.transform(data_train)
        
        hiperparametros = {
            'learning_rate_init': 0.1,
            'activation': 'relu',
            'alpha': 0.001,
            'max_iter': 600,
            'shuffle': False
        }
        if params:
            hiperparametros.update(params)

        self.model = MLPRegressor(**hiperparametros)
        self.model.fit(X_train, label_train)
        super().entrenar()
        return True
    
    
    @Modelo.cronometrar()
    def predecir(self, df):
        """
        """
        data = df.copy()
        data_test, label_test = self._split_data_label(data)
        X_data = self.scaler.transform(data_test)
        predictions = self.model.predict(X_data)
        data["target"] = predictions
        return data


# In[ ]:


def test():
    modelo = MLP_Regressor()    
    modelo.cargar_datos()
    modelo.entrenar()
    return modelo


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
    modelo = MLP_Regressor()
    modelo.cargar_datos()
    cantidad_pruebas = 10
    opciones = {
        'learning_rate_init': [0.1, 0.01],
        'activation': ['logistic','tanh','relu'],
        'alpha': [0.0001, 0.001],
        'max_iter': [200, 300, 400],
        'shuffle': [False, True]
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
# 
# {'learning_rate_init': 0.1, 'activation': 'relu', 'alpha': 0.001, 'max_iter': 400, 'shuffle': False}
# 
# {'learning_rate_init': 0.1, 'activation': 'relu', 'alpha': 0.0001, 'max_iter': 300, 'shuffle': False}
# 
# {'learning_rate_init': 0.01, 'activation': 'relu', 'alpha': 0.0001, 'max_iter': 300, 'shuffle': True}

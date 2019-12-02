#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)
from modelo import Modelo
from datos import FEATURES_DISPONIBLES


# In[2]:


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[14]:


class RegresionLineal(Modelo):
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
            "ciudad"
        }
        super().cargar_datos(FEATURES_DISPONIBLES - excluir)
    
    @Modelo.cronometrar()
    def entrenar(self):
        super().entrenar()
        self.train_data = self.preparar_datos(self.train_data)
        self.test_data = self.preparar_datos(self.test_data)
        self.submit_data = self.preparar_datos(self.submit_data)
        self.agregar_columnas_faltantes()
        train_data = self.train_data.loc[:, self.train_data.columns != self.feature]
        train_label = np.log(self.train_data[self.feature])
        self.regression = LinearRegression(normalize=False).fit(train_data, train_label)
        return True
    
    def preparar_datos(self, df):
        """
        
        """
        df = df.drop(columns=["fecha", "titulo", "descripcion"])
        categoricas = ["tipodepropiedad", "provincia"]
        return self.one_hot_encode(df, categoricas)
    

    @Modelo.cronometrar()
    def predecir(self, df):
        """

        """
        datos = df.copy()
        a_predecir = datos.loc[:, datos.columns != 'precio']
        datos['target'] = np.exp(self.regression.predict(a_predecir))
        return datos
            


# In[15]:


def test():
    modelo = RegresionLineal()
    modelo.cargar_datos()
    modelo.entrenar()
    return modelo


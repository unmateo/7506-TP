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
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[26]:


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
            "tipodepropiedad",
            "provincia",
            "ciudad"
        }
        super().cargar_datos(FEATURES_DISPONIBLES - excluir)
    
    @Modelo.cronometrar()
    def entrenar(self):
        super().entrenar()
        self.train_data = self.preparar_datos(self.train_data)
        train_data = self.train_data.loc[:, self.train_data.columns != 'precio']
        train_label = self.train_data["precio"]
        self.regression = LinearRegression(normalize=True).fit(train_data, train_label)
        self.test_data = self.preparar_datos(self.test_data)
        self.submit_data = self.preparar_datos(self.submit_data)
        return True
    
    def preparar_datos(self, df):
        """
        
        """
        df = df.drop(columns=["fecha", "titulo", "descripcion"]) 
        rellenas = self.rellenar_vacios(df)
        return rellenas
    
    def rellenar_vacios(self, df):
        """
            Rellena los valores vacíos del df con el promedio de esa columna.
            Devuelve el DataFrame modificado.
        """
        return df.fillna(df.mean())

    @Modelo.cronometrar()
    def predecir(self, df):
        """

        """
        datos = df.copy()
        a_predecir = datos.loc[:, datos.columns != 'precio']
        datos['target'] = self.regression.predict(a_predecir)
        return datos
            


# In[4]:


def test():
    modelo = RegresionLineal()
    modelo.cargar_datos()
    modelo.entrenar()
    modelo.validar()
    predicciones = modelo.predecir(modelo.submit_data)
    comentario = 'Primer intento con regresor lineal'
    #modelo.presentar(predicciones, comentario)
    return modelo


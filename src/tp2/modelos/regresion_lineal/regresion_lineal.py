#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)
from modelo import Modelo


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
        features = {
            "piscina", "usosmultiples", "gimnasio", "garages",
            "escuelascercanas", "centroscomercialescercanos",
            "banos", "habitaciones", "metroscubiertos", "metrostotales",
            "antiguedad", "ano", "precio"
        }
        super().cargar_datos(features)
    
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
        #encodeadas = self.encodear_categoricas(df)
        df.drop(columns=["fecha"], inplace=True)
        rellenas = self.rellenar_vacios(df)
        return rellenas
    
    def encodear_categoricas(self, df):
        """
            Recibe un set de datos y le aplica one hot encoding a sus
            variables categoricas.
        """
        
        categoricas = ["tipodepropiedad"]
        return self.one_hot_encode(df, categoricas)
    
    def rellenar_vacios(self, df):
        """
            Rellena los valores vacíos del df con el promedio de esa columna.
            Devuelve el DataFrame modificado.
        """
        return df.fillna(df.mean())

    @Modelo.cronometrar()
    def predecir(self, datos):
        """

        """
#        columnas = set(datos.columns)
        a_predecir = datos.loc[:, self.train_data.columns != 'precio']
        datos['target'] = self.regression.predict(a_predecir)
        return datos
            


# In[27]:


modelo = RegresionLineal()


# In[28]:


modelo.cargar_datos()


# In[29]:


modelo.entrenar()


# In[30]:


modelo.validar()


# In[ ]:


predicciones = modelo.predecir(modelo.submit_data)


# In[ ]:


comentario = 'Primer intento con regresor lineal'
#modelo.presentar(predicciones, comentario)


# In[13]:


import pandas as pd
df = pd.DataFrame()
df.drop()


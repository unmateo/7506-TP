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


# In[3]:


class RegresionLineal(Modelo):
    """

    """
    
    
    @Modelo.cronometrar()
    def entrenar(self):
        super().entrenar()
        # no puedo usar estas columnas porque no las  voy a tener para predecir
        self.excluir = {'precio_metro_total', 'precio_metro_cubierto'}
        datos = self.filtrar_numericas(self.train_data, self.excluir)
        self.train_data = datos
        train_label = datos["precio"]
        train_data = datos.loc[:, datos.columns != 'precio']
        self.regression = LinearRegression().fit(train_data, train_label)
        
        self.test_data = self.filtrar_numericas(self.test_data, self.excluir)
        self.excluir = self.excluir.union({'precio'})
        self.submit_data = self.filtrar_numericas(self.submit_data, self.excluir)
        

        return True
    
    def filtrar_numericas(self, df, excluir={}):
        """ 
            Recibe un set de datos y se queda sólo con las columnas numéricas.
            Excluye de la respuesta las keys que vengan en excluir.
            Llena los NaN con el promedio de esa columna.
        """
        columnas_numericas = {'antiguedad', 'habitaciones', 'garages', 'banos',
            'metroscubiertos', 'metrostotales', 'gimnasio', 'usosmultiples',
            'piscina', 'escuelascercanas', 'centroscomercialescercanos',
            'precio_metro_cubierto', 'precio_metro_total', 'ano', 'mes', 'dia', 'precio'
        }
        columnas_numericas.difference_update(excluir)
        datos = df.loc[:, columnas_numericas]
        return datos.fillna(datos.mean())
    
    @Modelo.cronometrar()
    def predecir(self, datos):
        """

        """
        columnas = set(datos.columns)
        columnas.difference_update(self.excluir)
        a_predecir = datos.loc[:, columnas]
        datos['target'] = self.regression.predict(a_predecir)
        return datos
            


# In[4]:


modelo = RegresionLineal()


# In[8]:


modelo.entrenar()


# In[9]:


modelo.validar()


# In[10]:


predicciones = modelo.predecir(modelo.submit_data)


# In[12]:


comentario = 'Primer intento con regresor lineal'
#modelo.presentar(predicciones, comentario)


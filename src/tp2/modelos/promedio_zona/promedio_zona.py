#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)
from modelo import Modelo


# In[38]:


import pandas as pd
from numpy import mean
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[54]:


class PromedioZona(Modelo):
    """
        Calculamos el precio promedio por metro cubierto y metro total
        para cada zona. Asignamos ese precio en caso de tener zona, o
        el promedio general en caso de no tenerlo.
    """
    
    def cargar_datos(self):
        """
        """
        features = {
            "idzona", "ciudad", "provincia",
            "precio_metro_cubierto", "precio_metro_total",
            "metroscubiertos", "precio", "metrostotales"
        }
        super().cargar_datos(features)
    
    @Modelo.cronometrar()
    def entrenar(self):
        super().entrenar()
        predicciones_por_zona = self._promedio_por_feature(self.train_data, "idzona")
        predicciones_por_ciudad = self._promedio_por_feature(self.train_data, "ciudad")
        predicciones_por_provincia = self._promedio_por_feature(self.train_data, "provincia")
        self.predicciones = [
            ("idzona", predicciones_por_zona),
            ("ciudad", predicciones_por_ciudad),
            ("provincia", predicciones_por_provincia),
        ]
        self.promedio_cubiertos = self.train_data["precio_metro_cubierto"].mean()
        self.promedio_totales = self.train_data["precio_metro_total"].mean()
        promedio_general = (self.promedio_cubiertos + self.promedio_totales) / 2
        metros_general = mean(self.train_data["metroscubiertos"].dropna())
        self.prediccion_default = metros_general * promedio_general
    
    @Modelo.cronometrar()
    def predecir(self, df):
        """
            Genera una copia del df.
            Aplica predecir_publicacion a cada fila.
            Asigna el resultado en la columna 'target'.
            Devuelve la copia.
        """
        datos = df.copy()
        prediccion = lambda publicacion: self.predecir_publicacion(publicacion)
        datos["target"] = datos.apply(prediccion, axis="columns")
        return datos

    def predecir_publicacion(self, publicacion):
        """ Predice el precio de una publicacion en base a los
            promedios de los siguientes campos, en orden de prioridad:
                - idzona
                - ciudad
                - provincia
                - general
        """
        if not self.entrenado:
            raise Exception("No se ha entrenado.")
        cubiertos = publicacion["metroscubiertos"]
        totales = publicacion["metrostotales"]
        for feature, predicciones in self.predicciones:
            predicciones_feature = predicciones.get(publicacion[feature])
            prediccion = self._predecir_por_feature(predicciones_feature, totales, cubiertos)
            if prediccion: return prediccion
        if cubiertos > 0:
            return cubiertos * self.promedio_cubiertos
        if totales > 0:
            return totales * self.promedio_totales
        return self.prediccion_default
    
    def _promedio_por_feature(self, df, feature, minimas_apariciones=5):
        """ Recibe un df y uno de sus features.
            Agrupa por ese feature.
            Calcula promedios de precio para cada grupo.
            Devuelve los resultados en un diccionario.
        """
        calculos = {
            "idzona": "count",
            "precio_metro_cubierto": "mean",
            "precio_metro_total": "mean",
        }
        por_feature = df.groupby([feature]).agg(calculos)
        suficientes_datos = por_feature.loc[por_feature["idzona"] > minimas_apariciones].drop(columns=["idzona"])
        return suficientes_datos.to_dict(orient="index")
    
    def _predecir_por_feature(self, predicciones, totales, cubiertos):
        """
        
        """
        if not predicciones:
            return None
        
        prediccion_totales = predicciones.get("precio_metro_total")
        if totales > 0 and prediccion_totales > 0:
            return totales * prediccion_totales

        prediccion_cubiertos = predicciones.get("precio_metro_cubierto")
        if cubiertos > 0 and prediccion_cubiertos > 0:
            return cubiertos * prediccion_cubiertos      

        if prediccion_totales > 0:
            return self.promedio_totales * prediccion_totales

        if prediccion_cubiertos > 0:
            return self.promedio_cubiertos * prediccion_cubiertos

        return None
            


# In[55]:


def test():
    modelo = PromedioZona()
    modelo.cargar_datos()
    modelo.entrenar()
    return modelo


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
this = os.getcwd()
path = this[:this.rfind("/")]
if not path in sys.path: sys.path.append(path)
from modelo import Modelo


# In[ ]:


os.getcwd()


# In[ ]:


class PromedioZona(Modelo):
    """
        Calculamos el precio promedio por metro cubierto y metro total
        para cada zona. Asignamos ese precio en caso de tener zona, o
        el promedio general en caso de no tenerlo.
    """
    
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
        metros_general = self.train_data["metroscubiertos"].mean()
        self.prediccion_default = metros_general * promedio_general
    
    @Modelo.cronometrar()
    def predecir(self, datos):
        """
            Aplica predecir_publicacion a cada fila del df que recibe.
            Asigna el resultado en la columna 'target'
        """
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
        return self.prediccion_default
    
    def _promedio_por_feature(self, df, feature, minimas_apariciones=5):
        """ Recibe un df y uno de sus features.
            Agrupa por ese feature.
            Calcula promedios de precio para cada grupo.
            Devuelve los resultados en un diccionario.
        """
        calculos = {
            "id": "count",
            "precio_metro_cubierto": "mean",
            "precio_metro_total": "mean",
        }
        por_feature = df.groupby([feature]).agg(calculos)
        suficientes_datos = por_feature.loc[por_feature["id"] > minimas_apariciones].drop(columns=["id"])
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
            


# In[ ]:


modelo = PromedioZona()


# In[ ]:


modelo.entrenar()


# In[ ]:


modelo.validar()


# In[ ]:


comentario = "Despues de usar datos de ciudad y provincia"
#modelo.presentar(predicciones, comentario)


# In[ ]:


#modelo.buscar_score(comentario)


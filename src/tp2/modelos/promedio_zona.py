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


class PromedioZona(Modelo):
    """
        Calculamos el precio promedio por metro cubierto y metro total
        para cada zona. Asignamos ese precio en caso de tener zona, o
        el promedio general en caso de no tenerlo.
    """
    
    @Modelo.cronometrar()
    def entrenar(self):
        super().entrenar()
        grouped = modelo.train_data.groupby(["idzona"]).agg({"id": "count", "precio_metro_cubierto":"mean", "precio_metro_total":"mean"})
        grouped.index = grouped.index.astype(int)
        suficientes_datos = grouped.loc[grouped["id"] > 5].drop(columns=["id"])
        self.predicciones = suficientes_datos.to_dict(orient="index")
        self.promedio_cubiertos = suficientes_datos["precio_metro_cubierto"].mean()
        self.promedio_totales = suficientes_datos["precio_metro_total"].mean()
        promedio_general = (self.promedio_cubiertos + self.promedio_totales) / 2
        metros_general = self.train_data["metroscubiertos"].mean()
        self.prediccion_default = metros_general * promedio_general
    
    @Modelo.cronometrar()
    def predecir(self, datos):
        prediccion = lambda publicacion: self.predecir_publicacion(publicacion)
        datos["target"] = datos.apply(prediccion, axis="columns")
        return datos

    def predecir_publicacion(self, publicacion):
        """
            
        
        """
        if not self.entrenado:
            raise Exception("No se ha entrenado.")
        zona = publicacion["idzona"]
        cubiertos = publicacion["metroscubiertos"]
        totales = publicacion["metrostotales"]
        predicciones_zona = self.predicciones.get(zona)
        if not predicciones_zona:
            if totales > 0:
                return self.promedio_totales * totales
            if cubiertos > 0:
                return self.promedio_cubiertos * cubiertos
            return self.prediccion_default

        prediccion_totales = predicciones_zona.get("precio_metro_total")
        if totales > 0 and prediccion_totales > 0:
            return totales * prediccion_totales

        prediccion_cubiertos = predicciones_zona.get("precio_metro_cubierto")
        if cubiertos > 0 and prediccion_cubiertos > 0:
            return cubiertos * prediccion_cubiertos

        if totales > 0:
            return totales * self.promedio_totales

        if cubiertos > 0:
            return cubiertos * self.promedio_cubiertos

        return self.prediccion_default
            


# In[3]:


modelo = PromedioZona()
modelo.entrenar()
modelo.validar()
predicciones = modelo.predecir(modelo.submit_data)
# modelo.presentar(predicciones)


# In[7]:


modelo.tiempos


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


descripcion = """
    Calculamos el precio promedio por metro cubierto y metro total
    para cada zona. Asignamos ese precio en caso de tener zona, o
    el promedio general en caso de no tenerlo.
"""

class PromedioZona(Modelo):
    
    
    
    def entrenar(self):
        super().entrenar()
        self.test = self.test[["id","idzona", "metroscubiertos", "metrostotales"]]
        self.train["promedio_cubiertos"] = self.train["precio"] / self.train["metroscubiertos"]
        self.train["promedio_totales"] = self.train["precio"] / self.train["metrostotales"]
        grouped = modelo.train.groupby(["idzona"]).agg({"id": "count", "promedio_cubiertos": "mean", "promedio_totales": "mean"})
        grouped.index = grouped.index.astype(int)
        suficientes_datos = grouped.loc[grouped["id"] > 5].drop(columns=["id"])
        self.predicciones = suficientes_datos.to_dict(orient="index")
        self.promedio_cubiertos = suficientes_datos["promedio_cubiertos"].mean()
        self.promedio_totales = suficientes_datos["promedio_totales"].mean()
        promedio_general = (self.promedio_cubiertos + self.promedio_totales) / 2
        metros_general = self.train["metroscubiertos"].mean()
        self.prediccion_default = metros_general * promedio_general
    
    def predecir(self):
        prediccion = lambda publicacion: self.predecir_publicacion(publicacion)
        self.test["prediccion"] = self.test.apply(prediccion, axis="columns")
        #import pdb; pdb.set_trace()
        self.resultados = self.test[["id", "prediccion"]].set_index("id")
        return True

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
        
        prediccion_totales = predicciones_zona.get("promedio_totales")
        if totales > 0 and prediccion_totales > 0:
            return totales * prediccion_totales
        
        prediccion_cubiertos = predicciones_zona.get("promedio_cubiertos")
        if cubiertos > 0 and prediccion_cubiertos > 0:
            return cubiertos * prediccion_cubiertos
        
        if totales > 0:
            return totales * self.promedio_totales
        
        if cubiertos > 0:
            return cubiertos * self.promedios_cubiertos
        
        return self.prediccion_default
            


# In[3]:


modelo = PromedioZona(descripcion)
modelo.entrenar()
modelo.predecir()
modelo.validar()
modelo.presentar()


# In[7]:


modelo.resultados.head()


# In[8]:


modelo.resultados.to_csv("test.csv", header=["target"])


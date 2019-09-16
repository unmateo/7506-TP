#!/usr/bin/env python
# coding: utf-8

# ## Campos involucrados
# 
# - direccion
# - idzona
# - ciudad
# - provincia
# - lat
# - lng
# 
# ## Objetivos
# 
# - Normalizar (provincia, ciudad, dirección)
# - Agregar información (geometry)
# - Métodos para graficar (cantidad+densidad)
# - Método para definir si x publicación está en una geometry
# 
# - Agregar información externa (distrito electoral, etc.)
# 

# In[1]:


import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


# In[3]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')


# In[7]:


pais = geopandas.read_file("./MEX_adm/MEX_adm0.shp")
estados = geopandas.read_file("./MEX_adm/MEX_adm1.shp")
municipios = geopandas.read_file("./MEX_adm/MEX_adm2.shp")
ciudades = geopandas.read_file("./México_Centros_Urbanos/México_Centros_Urbanos.shp")


# In[56]:


df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
con_gps = df.loc[~ (df["lat"].isna() & df["lng"].isna())]
con_gps["coord_point"] = con_gps.apply(lambda x: Point(x["lng"],x["lat"]), axis=1)


# In[76]:


def esta_en_mexico(point: Point) -> bool:
    """ 
        Recibe un punto (lat,lng) y devuelve (muy aproximadamente) True si está dentro de Mexico, False si no.
        Hacerlo con ```df["coord"].map(lambda x: mexico_polygon.contains(x))``` sería muy lento.
    """
    MEX_MIN_LNG, MEX_MAX_LNG = (-120, -85)
    MEX_MIN_LAT, MEX_MAX_LAT = (14,33)
    return (MEX_MIN_LNG < point.x < MEX_MAX_LNG) and (MEX_MIN_LAT < point.y < MEX_MAX_LAT)


# In[82]:


con_gps["en_mexico"] = con_gps["coord_point"].map(esta_en_mexico)


# In[83]:


con_gps["en_mexico"].value_counts()


# In[87]:


fuera_de_mexico = con_gps.loc[~con_gps["en_mexico"]]
en_mexico = con_gps.loc[con_gps["en_mexico"]]


# In[88]:


validos = geopandas.GeoDataFrame(en_mexico, geometry="coord_point")


# In[61]:


mexico_polygon = pais.iloc[0]["geometry"]


# In[131]:


def dibujar_mexico(puntos):
    grafico = pais.plot(figsize=(18,9))
    estados.plot(ax=grafico, color="white")
#     municipios.plot(ax=grafico, color="white")
    # ciudades.plot(ax=grafico, color="yellow")
    puntos.plot(ax=grafico, color="green")


# In[122]:


# le cambio los nombres a las siguientes provincias, para que coincidan con mi info geografica 
provincias_mapper = {
    "Baja California Norte": "Baja California",
    "Edo. de México": "México",
    "San luis Potosí": "San Luis Potosí"
}
validos["provincia"] = validos["provincia"].map(lambda x: provincias_mapper.get(x, x))
set(validos["provincia"].dropna().unique()) == set(estados["NAME_1"]) #verifico


# In[163]:


# estos tienen datos gps pero no de provincia
sin_provincia = validos.loc[validos["provincia"].isna()]

def buscar_provincia(punto: Point, provincias):
    """
        Devuelve en qué provincia de mexico se encuentra el punto.
    """
    for provincia, geometry in provincias[["NAME_1","geometry"]].values:
        if geometry.contains(punto): return provincia
    return None

# agrego las provincias faltantes
sin_provincia["provincia"] = sin_provincia["coord_point"].map(lambda x: buscar_provincia(x, estados))


# In[164]:


validos.loc[validos["provincia"].isna()]


# In[161]:


sin_provincia


# In[162]:


dibujar_mexico(sin_provincia)


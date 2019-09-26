#!/usr/bin/env python
# coding: utf-8

# ## Campos involucrados
# 
# - escuelascercanas
# - centroscomercialescercanos
# - idzona
# 
# ## Ideas
# 
# - Agregar info externa (transporte cercano? barrios "de moda"?)
# - Caracterización de zona
#   - Centro Geográfico
#   - ¿diámetro?
#   - Promedios
#     - metroscubiertos
#     - precio
#     - ...
# - Armar un dataframe con info de zonas?
# 
# 
# ## Problema
# 
# - Dado un grupo de coordenadas geográficas, no es tan sencillo calcular áreas y distancias (habría que proyectar). 
# 
# __Hay publicaciones cuyas coordenadas son demasiado lejanas a las del resto de la zona__
# ¿Cómo las encuentro?
# - Distancia al centro de la zona (¿promedio de latitud-longitud u otra medida?)
# 

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point, Polygon
import numpy


# In[2]:


# importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
# importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')


# In[3]:


# cargo el dataset
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[5]:


def agg_polygon(point_series):
    """
        Recibe un pd.Series de geometry.Point
        Devuelve un Polygon de los puntos de la serie, o NaN si
        no tiene suficientes puntos (3).
    """
    values = point_series.loc[~point_series.isna()].values
    if len(values) < 3: return numpy.NaN
    return Polygon([[p.x, p.y] for p in values])


# ## Armo un Dataframe donde las filas son las zonas

# In[6]:


calculations = ["mean","std","max","min"]
aggregations = {"id": "count",                "precio_metro_total": calculations,                "precio_metro_cubierto": calculations,                "antiguedad": calculations,                "habitaciones": calculations,                "metroscubiertos": calculations,                "metrostotales": calculations,                "lat": calculations,                "lng": calculations,                "precio": calculations,                "habitaciones": calculations,                "garages": calculations,                "banos": calculations,                "gps": agg_polygon               }
zonas = df.groupby(["idzona"]).agg(aggregations)
zonas.columns = [x+"_"+y for x,y in zonas.columns]
zonas.rename({"gps_agg_polygon": "polygon"}, axis="columns", inplace=True)


# In[18]:


zonas["lat_dif"] = zonas["lat_max"] - zonas["lat_min"]
zonas["lng_dif"] = zonas["lng_max"] - zonas["lng_min"]


# In[62]:


zonas.shape


# In[53]:


zonas[["lat_dif","lng_dif"]].describe()


# In[64]:


zonas_ok = zonas.loc[(zonas["lat_dif"] < zonas["lat_dif"].mean()) & (zonas["lng_dif"] < zonas["lng_dif"].mean())]
zonas_ok.shape


# In[7]:


pais = geopandas.read_file("./MEX_adm/MEX_adm0.shp")
estados = geopandas.read_file("./MEX_adm/MEX_adm1.shp")


# In[65]:


con_polygon = zonas_ok.loc[~zonas["polygon"].isna()]
geoDF = geopandas.GeoDataFrame(con_polygon, geometry="polygon")
base = pais.plot(figsize=(18,9))
estados_plot = estados.plot(ax=base, color="white")
plot = geoDF.plot(ax=estados_plot, cmap="Greens_r")


# In[75]:


zonas_ok.loc[:,"centroid"] = zonas_ok.loc[~zonas_ok["polygon"].isna()]["polygon"].map(lambda x: x.centroid)


# In[78]:


zonas_ok.sort_values(by="id_count", ascending=False)


# In[81]:


publicaciones_84028 = df.loc[df["idzona"]==84028.0] 
publicaciones_84028.head(1)


# In[88]:


geoDF = geopandas.GeoDataFrame(publicaciones_84028, geometry="gps")
queretaro = estados.loc[estados["NAME_1"]=="Querétaro"].plot(figsize=(18,9),color="gray")
plot = geoDF.plot(ax=queretaro, cmap="Greens_r")


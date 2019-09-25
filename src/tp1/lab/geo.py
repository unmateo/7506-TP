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


# In[2]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')


# In[3]:


pais = geopandas.read_file("./MEX_adm/MEX_adm0.shp")
estados = geopandas.read_file("./MEX_adm/MEX_adm1.shp")
municipios = geopandas.read_file("./MEX_adm/MEX_adm2.shp")
ciudades = geopandas.read_file("./México_Centros_Urbanos/México_Centros_Urbanos.shp")
mexico_polygon = pais.iloc[0]["geometry"]


# In[4]:


df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df["tiene_gps"] = ~ (df["lat"].isna() & df["lng"].isna())
crear_punto = lambda x: Point(x["lng"],x["lat"]) if x["tiene_gps"] else None
df["coord"] = df.apply(crear_punto, axis=1)


# In[5]:


def esta_en_mexico(point: Point) -> bool:
    """ 
        Recibe un punto (lat,lng) y devuelve (muy aproximadamente) True si está dentro de Mexico, False si no.
        Hacerlo con ```df["coord"].map(lambda x: mexico_polygon.contains(x))``` sería muy lento.
    """
    MEX_MIN_LNG, MEX_MAX_LNG = (-120, -85)
    MEX_MIN_LAT, MEX_MAX_LAT = (14,33)
    return (MEX_MIN_LNG < point.x < MEX_MAX_LNG) and (MEX_MIN_LAT < point.y < MEX_MAX_LAT)


# In[6]:


df["en_mexico"] = df.loc[df["tiene_gps"]]["coord"].map(esta_en_mexico)


# In[11]:


df["en_mexico"].value_counts()


# In[49]:


geoDF = geopandas.GeoDataFrame(df.loc[df["tiene_gps"] & df["en_mexico"]], geometry="coord")


# In[50]:


def dibujar_mexico(puntos):
    grafico = pais.plot(figsize=(18,9))
    estados.plot(ax=grafico, color="white")
#     municipios.plot(ax=grafico, color="white")
    # ciudades.plot(ax=grafico, color="yellow")
    puntos.plot(ax=grafico, color="green")


# In[51]:


def fix_provincias(df, provincias) -> bool:
    # le cambio los nombres a las siguientes provincias, para que coincidan con mi info geografica 
    provincias_mapper = {
        "Baja California Norte": "Baja California",
        "Edo. de México": "México",
        "San luis Potosí": "San Luis Potosí"
    }
    df["estado"] = df["provincia"].map(lambda x: provincias_mapper.get(x, x))
    return set(validos["provincia"].dropna().unique()) == set(provincias["NAME_1"]) #verifico


# In[52]:


fix_provincias(geoDF, estados)


# In[53]:


def buscar_provincia(punto: Point, provincias):
    """
        Devuelve en qué provincia de mexico se encuentra el punto.
    """
#     if not punto: return None
    for provincia, geometry in provincias[["NAME_1","geometry"]].values:
        if geometry.contains(punto): return provincia

# agrego las provincias faltantes
geoDF.loc[geoDF["estado"].isna(), "estado"] = geoDF.loc[geoDF["estado"].isna()]["coord"].map(lambda x: buscar_provincia(x, estados))


# In[54]:


publicaciones_por_estado = geoDF.loc[~geoDF["estado"].isna()].groupby(["estado"]).agg({"estado":"count"})


# In[55]:


def choropleth_estados(estados, serie, nombre, titulo=""):
    estados[nombre] = estados["NAME_1"].map(serie)
    plot = estados.plot(column=nombre, legend=True, figsize=(24,8))    
    plot.set_title(titulo)
    return plot


# In[56]:


plot = choropleth_estados(estados, publicaciones_por_estado["estado"], "publicaciones", "Cantidad de publicaciones por estado")


# # Presento un análisis del valor del metro cuadrado en relacion a la ciudad

# ### Primero realizo una limpieza de los datos. Selecciono las ciudades con mayor cantidad de publicaciones

# In[7]:


mas_publicadas = df.groupby("ciudad").agg({"id":"count"})
mas_publicadas.columns = ["total"]
mas_publicadas=mas_publicadas.sort_values("total", ascending=False).head(100)
print(mas_publicadas)


# In[ ]:


mas_publicadas=pd.merge(df,mas_publicadas, on='ciudad', how='inner').ciudad.value_counts().head(101)
mas_publicadas


# In[14]:


#Realizo un calculo del promedio del valor del metro cuadrado por 
por_ciudad=df.groupby("ciudad").agg({"metrostotales":"sum"})
por_ciudad["precios"] = df.groupby("ciudad").agg({"precio":"sum"})
por_ciudad["valormetrocuadrado"] = por_ciudad["precios"] / por_ciudad["metrostotales"]


# ### Limpio el dataset de valores nulos en metrostotales y/o precios

# In[10]:


por_ciudad=por_ciudad.loc[(por_ciudad.metrostotales != 0.0)]
por_ciudad=por_ciudad.loc[(por_ciudad.precios != 0.0)]


# # Busco las ciudades extremo, la más cara y la más barata

# In[39]:


por_ciudad = por_ciudad.sort_values("valormetrocuadrado")
print(por_ciudad)


# ### Ahora armo un dataframe con las 20 ciudades más caras y las 20 más baratas.

# In[36]:


top_20_ciudades_mas_caras = por_ciudad.tail(20)
top_20_ciudades_mas_caras.reset_index(inplace=True)
top_20_ciudades_mas_caras


# In[34]:


top_20_ciudades_mas_baratas = por_ciudad.head(20)
top_20_ciudades_mas_baratas.reset_index(inplace=True)
top_20_ciudades_mas_baratas


# In[56]:


ciudad_mas_barata = (top_20_ciudades_mas_baratas.loc[0,:].ciudad,top_20_ciudades_mas_baratas.loc[0,:].valormetrocuadrado)
print("Ciudad mas barata {}".format(ciudad_mas_barata))
ciudad_mas_cara = (top_20_ciudades_mas_caras.loc[0,:].ciudad,top_20_ciudades_mas_caras.loc[0,:].valormetrocuadrado)
print("Ciudad mas cara {}".format(ciudad_mas_cara))
amplitud = ciudad_mas_cara[1] - ciudad_mas_barata[1]
print("Amplitud de precio {}".format(amplitud))


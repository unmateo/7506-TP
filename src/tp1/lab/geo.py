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


# In[ ]:





# In[5]:


df["en_mexico"] = df.loc[df["tiene_gps"]]["coord"].map(esta_en_mexico)


# In[6]:


df["en_mexico"].value_counts()


# In[7]:


geoDF = geopandas.GeoDataFrame(df.loc[df["tiene_gps"] & df["en_mexico"]], geometry="coord")


# In[8]:


def dibujar_mexico(puntos):
    grafico = pais.plot(figsize=(18,9))
    estados.plot(ax=grafico, color="white")
#     municipios.plot(ax=grafico, color="white")
    # ciudades.plot(ax=grafico, color="yellow")
    puntos.plot(ax=grafico, color="green")


# In[9]:


def fix_provincias(df, provincias) -> bool:
    # le cambio los nombres a las siguientes provincias, para que coincidan con mi info geografica 
    provincias_mapper = {
        "Baja California Norte": "Baja California",
        "Edo. de México": "México",
        "San luis Potosí": "San Luis Potosí"
    }
    df["estado"] = df["provincia"].map(lambda x: provincias_mapper.get(x, x))
    return set(df["provincia"].dropna().unique()) == set(provincias["NAME_1"]) #verifico


# In[10]:


fix_provincias(geoDF, estados)


# In[11]:


def buscar_provincia(punto: Point, provincias):
    """
        Devuelve en qué provincia de mexico se encuentra el punto.
    """
#     if not punto: return None
    for provincia, geometry in provincias[["NAME_1","geometry"]].values:
        if geometry.contains(punto): return provincia

# agrego las provincias faltantes
geoDF.loc[geoDF["estado"].isna(), "estado"] = geoDF.loc[geoDF["estado"].isna()]["coord"].map(lambda x: buscar_provincia(x, estados))


# In[12]:


publicaciones_por_estado = geoDF.loc[~geoDF["estado"].isna()].groupby(["estado"]).agg({"estado":"count"})
publicaciones_por_estado.columns = ["publicaciones"]


# In[13]:


# agrego datos de población al df de estados
df_poblacion = pd.read_csv("./poblacion_por_estado.csv", index_col="NAME_1")
estados = estados.merge(on="NAME_1", right=df_poblacion)


# In[14]:


# agrego datos de publicaciones al df de estados
estados = estados.merge(left_on="NAME_1", right_on="estado", right=publicaciones_por_estado)


# In[15]:


def choropleth_estados(estados, serie, nombre, titulo=""):
    estados[nombre] = estados["NAME_1"].map(serie)
    plot = estados.plot(column=nombre, legend=True, figsize=(24,8), cmap="Greens")    
    plot.set_title(titulo, fontdict={"fontsize": 18})
    plot.set_xlabel("Longitud")
    plot.set_ylabel("Latitud")
    return plot


# In[16]:


plot = choropleth_estados(estados, publicaciones_por_estado["publicaciones"], "publicaciones", "Cantidad de publicaciones por estado")
plot.figure.savefig("../graficos/map_publicaciones_por_estado.png")


# In[17]:


plot = estados.plot(column="poblacion", legend=True, figsize=(24,8), cmap="Greens")    
plot.set_title("Población de México por estado", fontdict={"fontsize": 18})
plot.set_xlabel("Longitud")
plot.set_ylabel("Latitud")
plot.figure.savefig("../graficos/map_poblacion_por_estado.png")


# In[18]:


estados["publicaciones_poblacion"] = estados["publicaciones"] / estados["poblacion"]
plot = estados.plot(column="publicaciones_poblacion", legend=True, figsize=(24,8), cmap="Greens")    
plot.set_title("Publicaciones por habitante en cada estado", fontdict={"fontsize": 18})
plot.set_xlabel("Longitud")
plot.set_ylabel("Latitud")
plot.figure.savefig("../graficos/map_publicaciones_por_habitante.png")


# # Presento un análisis del valor del metro cuadrado en relacion a la ciudad
# ### Primero realizo una limpieza de los datos. Selecciono las ciudades con mayor cantidad de publicaciones

# In[19]:


mas_publicadas = df.groupby("ciudad").agg({"id":"count"})
mas_publicadas.columns = ["total"]
mas_publicadas=mas_publicadas.sort_values("total", ascending=False).head(100)
mas_publicadas.reset_index(inplace=True)
print(mas_publicadas)


# In[20]:


lista_de_ciudades = mas_publicadas.ciudad
lista_de_ciudades = lista_de_ciudades.to_list()
lista_de_ciudades
df=df[df["ciudad"].isin(lista_de_ciudades)]
df


# In[21]:


#Realizo un calculo del promedio del valor del metro cuadrado por 
por_ciudad=df.groupby("ciudad").agg({"metrostotales":"sum"})
por_ciudad=por_ciudad.loc[por_ciudad.metrostotales != 0.0]
por_ciudad["precios"] = df.groupby("ciudad").agg({"precio":"sum"})
por_ciudad["valormetrocuadrado"] = por_ciudad["precios"] / por_ciudad["metrostotales"]
por_ciudad.reset_index(inplace=True)


# ### Limpio el dataset de valores nulos en metrostotales y/o precios

# In[22]:


por_ciudad=por_ciudad.loc[(por_ciudad.metrostotales != 0.0)]
por_ciudad=por_ciudad.loc[(por_ciudad.precios != 0.0)]
por_ciudad = por_ciudad.sort_values("valormetrocuadrado")
por_ciudad.reset_index(drop=True, inplace=True)
por_ciudad


# # Busco las ciudades extremo, la más cara y la más barata

# ### Ahora armo un dataframe con las 10 ciudades más caras y las 10 más baratas.

# In[23]:


top_10_ciudades_mas_caras = por_ciudad.tail(10)
top_10_ciudades_mas_caras.reset_index(drop=True, inplace=True)


# In[24]:


top_10_ciudades_mas_baratas = por_ciudad.head(10)
top_10_ciudades_mas_baratas.reset_index(inplace=True)


# In[25]:


vertical_stack = pd.concat([top_10_ciudades_mas_baratas, top_10_ciudades_mas_caras], axis=0, sort=False)
vertical_stack.reset_index(drop=True, inplace=True)
vertical_stack
bar_plot(vertical_stack["valormetrocuadrado"])


# In[ ]:


ciudad_mas_barata = (top_10_ciudades_mas_baratas.loc[0,:].ciudad,top_10_ciudades_mas_baratas.loc[0,:].valormetrocuadrado)
print("Ciudad mas barata {}".format(ciudad_mas_barata))
ciudad_mas_cara = (top_10_ciudades_mas_caras.loc[0,:].ciudad,top_10_ciudades_mas_caras.loc[0,:].valormetrocuadrado)
print("Ciudad mas cara {}".format(ciudad_mas_cara))
amplitud = ciudad_mas_cara[1] - ciudad_mas_barata[1]
print("Amplitud de precio {}".format(amplitud))


# In[ ]:


tiene_gps= df[~(df['gps'].isnull())]
tiene_gps=tiene_gps.groupby('ciudad').agg({"lat":"mean","lng":"mean"})


# In[ ]:





# In[ ]:





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

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point


# In[2]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[3]:


df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
df["precio_metro_total"] = df["precio"] / df["metrostotales"]


# In[4]:


promedios = df.groupby(["idzona"]).agg({"id": "count", "precio_metro_total": ["mean","std"], "precio_metro_cubierto": ["mean","std"]}).fillna(0).astype(int)
promedios.columns = ["publicaciones", "promedio_metrostotales", "desvio_metrostotales", "promedio_metroscubiertos","desvio_metroscubiertos"]
promedios.index = promedios.index.astype(int)
promedios.head()


# In[5]:


promedios.loc[promedios.index == 74]


# In[6]:


# me quedo con las zonas que tengan más publicaciones y ordeno por precio 
top_100_publicaciones = promedios.sort_values(by="publicaciones", ascending=False).head(100).sort_values(by="promedio_metroscubiertos", ascending=False)
# 10 barrios más caros
top_10 = top_100_publicaciones.head(10)
# 10 barrios más baratos
last_10 = top_100_publicaciones.tail(10)


# ## Armo un Dataframe donde las filas son las zonas

# In[7]:


calculations = ["mean","std","max","min"]
aggregations = {"id": "count",                "precio_metro_total": calculations,                "precio_metro_cubierto": calculations,                "antiguedad": calculations,                "habitaciones": calculations,                "metroscubiertos": calculations,                "metrostotales": calculations,                "lat": calculations,                "lng": calculations,                "precio": calculations,                "habitaciones": calculations,                "garages": calculations,                "banos": calculations,               }
zonas = df.groupby(["idzona"]).agg(aggregations)
zonas.columns = [x+"_"+y for x,y in zonas.columns]


# In[8]:


zonas.head()


# In[9]:


zonas["lat_dif"] = zonas["lat_max"] - zonas["lat_min"]
zonas["lng_dif"] = zonas["lng_max"] - zonas["lng_min"]


# In[10]:


zonas["has_gps"] = (~zonas["lat_mean"].isna()) & (zonas["lat_mean"] > 10) & (zonas["lng_mean"] < -80)


# In[11]:


zonas.has_gps.value_counts()


# In[12]:


# dimensiones (en cantidad de publicaciones) de las zonas que no tienen info gps
zonas.loc[~zonas["has_gps"]]["id_count"].sort_values(ascending=False).head()


# In[13]:


zonas_con_gps = zonas.loc[(zonas.has_gps) & (zonas.id_count > 10)]
zonas_con_gps = zonas_con_gps.loc[z]

def doble_violin(serie_izq, serie_der, titulo_izq, titulo_der):
    fig, ax = plt.subplots(1,2)
    plot_lng = sns.violinplot(serie_izq, orient="v", ax=ax[0])
    plot_lng.set_title(titulo_izq)
    plot_lat = sns.violinplot(serie_der, orient="v", color="red", ax=ax[1])
    plot_lat.set_title(titulo_der)
    return fig

plot = doble_violin(zonas_con_gps.lng_dif, zonas_con_gps.lat_dif, "Diferencia de Longitud", "Diferencia de Latitud")


# In[14]:


# en base a esto, me quedo con las zonas que tienen:
max_dif_lng, max_dif_lat = 0.2, 0.2
zonas_ok = zonas_con_gps.loc[(zonas_con_gps.id_count > 5) & (zonas_con_gps.lat_dif < max_dif_lat) & (zonas_con_gps.lng_dif < max_dif_lng)]
plot = doble_violin(zonas_ok.lng_dif, zonas_ok.lat_dif, "Diferencia de Longitud", "Diferencia de Latitud")


# In[15]:


zonas_ok.loc[zonas_ok.lat_mean < 10]
zonas_ok.loc[zonas_ok.lat_mean < 10]


# In[16]:


df.loc[df.idzona==70108.0][["lat","lng"]]


# In[22]:


zonas_ok.loc[:,"coord"] = zonas_ok.apply(lambda x: Point(x["lng_mean"],x["lat_mean"]), axis=1)


# In[24]:


zonas_ok["en_mexico"] = zonas_ok["coord"].map(esta_en_mexico)


# In[25]:


geoDF = geopandas.GeoDataFrame(zonas_ok, geometry="coord")


# In[26]:


pais = geopandas.read_file("./MEX_adm/MEX_adm0.shp")
estados = geopandas.read_file("./MEX_adm/MEX_adm1.shp")


# In[27]:


base = pais.plot(figsize=(18,9))
base.set_title("Zonas")
estados_plot = estados.plot(ax=base, color="white")
geoDF.plot(ax=estados_plot)


# In[28]:


geoDF.lng_mean.describe()


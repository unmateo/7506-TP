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


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[2]:


df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
df["precio_metro_total"] = df["precio"] / df["metrostotales"]


# In[3]:


promedios = df.groupby(["idzona"]).agg({"id": "count", "precio_metro_total": ["mean","std"], "precio_metro_cubierto": ["mean","std"]}).fillna(0).astype(int)
promedios.columns = ["publicaciones", "promedio_metrostotales", "desvio_metrostotales", "promedio_metroscubiertos","desvio_metroscubiertos"]
promedios.index = promedios.index.astype(int)
promedios.head()


# In[4]:


promedios.loc[promedios.index == 74]


# In[5]:


# me quedo con las zonas que tengan más publicaciones y ordeno por precio 
top_100_publicaciones = promedios.sort_values(by="publicaciones", ascending=False).head(100).sort_values(by="promedio_metroscubiertos", ascending=False)
# 10 barrios más caros
top_10 = top_100_publicaciones.head(10)
# 10 barrios más baratos
last_10 = top_100_publicaciones.tail(10)


# In[6]:


top_10


# In[7]:


last_10


# In[8]:


df.loc[df.idzona.isin(top_10.index)].describe()


# In[9]:


df.loc[df.idzona.isin(last_10.index)].describe()


# In[10]:


df.loc[df.idzona==113862].describe()


# ## Armo un Dataframe donde las filas son las zonas

# In[11]:


calculations = ["mean","std","max","min"]
aggregations = {"id": "count",                "precio_metro_total": calculations,                "precio_metro_cubierto": calculations,                "antiguedad": calculations,                "habitaciones": calculations,                "metroscubiertos": calculations,                "metrostotales": calculations,                "lat": calculations,                "lng": calculations,                "precio": calculations,                "habitaciones": calculations,                "garages": calculations,                "banos": calculations,               }
zonas = df.groupby(["idzona"]).agg(aggregations)
zonas.columns = [x+"_"+y for x,y in zonas.columns]


# In[12]:


zonas.head()


# In[13]:


zonas.head()


# In[14]:


zonas.columns


# In[15]:


zonas["lat_dif"] = zonas["lat_max"] - zonas["lat_min"]
zonas["lng_dif"] = zonas["lng_max"] - zonas["lng_min"]


# In[16]:


zonas["has_gps"] = ~zonas["lat_mean"].isna()


# In[17]:


zonas.has_gps.value_counts()


# In[18]:


# dimensiones (en cantidad de publicaciones) de las zonas que no tienen info gps
zonas.loc[~zonas["has_gps"]]["id_count"].sort_values(ascending=False)


# In[60]:


zonas[["has_gps","lat_max"]]


# In[68]:


import seaborn as sns
zonas_con_gps = zonas.loc[(zonas.has_gps) & (zonas.id_count > 10)]


# In[78]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2)
plot_lng = sns.violinplot(zonas_con_gps.lng_dif, orient="v", ax=ax[0])
plot_lng.set_title("Diferencia de Longitud")
plot_lat = sns.violinplot(zonas_con_gps.lat_dif, orient="v", color="red", ax=ax[1])
plot_lat.set_title("Diferencia de Latitud")


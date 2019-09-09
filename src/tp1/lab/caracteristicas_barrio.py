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
# - Precio promedio por metro cuadrado según zona

# In[1]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[6]:


df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
df["precio_metro_total"] = df["precio"] / df["metrostotales"]


# In[55]:


promedios = df.groupby(["idzona"]).agg({"id": "count", "precio_metro_total": ["mean","std"], "precio_metro_cubierto": ["mean","std"]}).fillna(0).astype(int)
promedios.columns = ["publicaciones", "promedio_metrostotales", "desvio_metrostotales", "promedio_metroscubiertos","desvio_metroscubiertos"]
promedios.index = promedios.index.astype(int)
promedios.head()


# In[46]:


promedios.loc[promedios.index == 74]


# In[63]:


# me quedo con las zonas que tengan más publicaciones
top_100_publicaciones = promedios.sort_values(by="publicaciones", ascending=False).head(100)
top_100_publicaciones.sort_values(by="promedio_metroscubiertos", ascending=False).head(10)


# In[66]:


df.loc[df.idzona==275358]


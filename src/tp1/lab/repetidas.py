#!/usr/bin/env python
# coding: utf-8

# ### Analizaremos la existencia de publicaciones repetidas

# In[35]:


import pandas as pd
#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')


# ### Consideramos que una publicación es igual a otra si comparten ciudad, precio, direccion, tipo de propiedad y metros totales. 

# In[57]:


repetidas = df.groupby(['ciudad','provincia','precio','direccion','metrostotales','tipodepropiedad']).agg({"id":"count"})
repetidas=repetidas.loc[repetidas.id>1]
repetidas


# ### La cantidad de publicaciones repetidas según nuestro criterio no es significativa frente al total de los datos. Quisieramos mencionar que en el caso de diferentes departamentos con iguales caracteristicas en un mismo edificio, las publicaciones matchearán como repetidas.

# In[58]:


repetidas=repetidas.groupby("tipodepropiedad").agg({"id":"count"})
repetidas=repetidas.rename(columns={"id":"total"})
get_barplot(repetidas["total"], title="Tipo de propiedad de las publicaciones repetidas", x_label="Tipo de propiedad", y_label="Total",)


# #### El grafico nos permite ver que la cantidad de apartamentos repetidos es muy baja, de modo que la influencia de departamentos iguales en un edificio es casi nula.

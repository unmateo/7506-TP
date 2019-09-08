#!/usr/bin/env python
# coding: utf-8

# ## Ideas
# 
# - Cantidad de publicaciones de cada tipo
#   - En total
#   - Por año
# - 

# In[1]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[67]:


por_tipo = df.groupby(["tipodepropiedad"]).agg({"tipodepropiedad":"count"}).unstack(fill_value=0).sort_values(ascending=False)
por_tipo.index = por_tipo.index.droplevel()
total_publicaciones = por_tipo.sum()
por_tipo_porcentual = por_tipo.apply(lambda x: round(100 * x/total_publicaciones, 2))
por_tipo_porcentual


# In[58]:


por_tipo_porcentual.head(5).sum()


# En base a los resultados de la tabla anterior, nos quedamos con los 5 tipos de propiedad más frecuentes en las publicaciones (96.41% del total) para poder sintetizar mejor los gráficos.

# In[76]:


top_5_tipos  = por_tipo_porcentual.head(5).index.values


# In[85]:


df_top_5 = df.loc[df["tipodepropiedad"].isin(top_5_tipos)]


# In[87]:


por_tipo_ano = df_top_5.groupby(["ano","tipodepropiedad"]).agg({"tipodepropiedad":"count"}).unstack(fill_value=0)
por_tipo_ano.columns = por_tipo_ano.columns.droplevel()


# In[89]:


totales = por_tipo_ano.sum(axis=1)


# In[90]:


# por_tipo_porcentual = por_tipo.apply(lambda x: x[totales], axis=1)
por_tipo_ano_porcentual = por_tipo_ano.apply(lambda x: round(100 * x/totales[x.index], 2), axis=0)


# In[92]:


por_tipo_ano_porcentual


# In[105]:


por_tipo_ano_porcentual.plot(kind='line', xticks=por_tipo_ano_porcentual.index.values)


# In[197]:


import matplotlib.pyplot as plt
def get_lineplot(data, xlabel="", ylabel="", **kwargs):
    plot = data.plot(figsize=(12,6), fontsize=12, **kwargs)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    return plot
plot = get_lineplot(por_tipo_ano_porcentual, xticks=por_tipo_ano_porcentual.index.values, title = "Porcentaje de publicaciones por tipo", xlabel="Año", ylabel="Porcentaje")


# In[ ]:





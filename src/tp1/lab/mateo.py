#!/usr/bin/env python
# coding: utf-8

# In[72]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')


# In[ ]:


df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[89]:


df.columns


# In[25]:


waffle_tipo = get_waffleplot(df.tipodepropiedad.value_counts(normalize=True).head(10), "Publicaciones por tipo de propiedad")


# In[26]:


waffle_tipo.savefig("../graficos/waffle_publicaciones_por_tipo.png")


# In[2]:


grouped = df.groupby(["ano","mes"]).aggregate({"id": "count"}).unstack()
grouped.columns = grouped.columns.droplevel()


# In[46]:


grouped


# In[38]:


df.groupby(["fecha"]).agg({"id": "count"}).tail()


# In[86]:


top_dias = df.groupby(["fecha"]).agg({"id": "count"}).sort_values(by="id",ascending=False).head(30)
top_dias.columns = ["cantidad_publicaciones"]


# In[73]:


plot = get_heatmap(grouped, title="Cantidad de publicaciones por año/mes")


# In[74]:


plot.figure.savefig("../graficos/heatmap_publicaciones_por_mes.png")


# In[5]:


get_barplot(df.tipodepropiedad.value_counts().head(5))


# In[53]:


top_3 = df["tipodepropiedad"].isin(["Apartamento", "Casa", "Casa en condominio"])
grouped_by_type = df.loc[top_3].groupby(["tipodepropiedad","mes","ano"]).aggregate({"precio": "count"}).unstack(fill_value=0).unstack(fill_value=0)
grouped_by_type.columns = grouped_by_type.columns.droplevel()
grouped_by_type


# In[54]:


grouped_by_type.T.plot(figsize=(18,12))


# In[60]:


df.groupby(["ano","mes"]).aggregate({"id": "count"}).plot(figsize=(18,12), logy=False)


# In[81]:


tipo_zona = df.groupby(["tipodepropiedad"]).aggregate({"idzona": "count", "id": "count"}).sort_values(by="id", ascending=False).head(5)
tipo_zona["proporcion"] = 100 * tipo_zona["idzona"] / tipo_zona["id"]
tipo_zona.sort_values(by="idzona", ascending=False)


# In[91]:


# todas las columnas que tienen latitud tienen tambien longitud
df.loc[df["lat"].isna() != df["lng"].isna()].shape


# In[102]:


df["has_gps"] = ~ (df["lat"].isna() & df["lng"].isna())
tipo_zona = df.groupby(["tipodepropiedad"]).aggregate({"has_gps": "sum", "id": "count"}).sort_values(by="id", ascending=False)
tipo_zona["proporcion"] = 100 * tipo_zona["has_gps"] / tipo_zona["id"]
tipo_zona.sort_values(by="has_gps", ascending=False)


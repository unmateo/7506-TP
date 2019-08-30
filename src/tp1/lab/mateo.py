#!/usr/bin/env python
# coding: utf-8

# In[12]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')


# In[4]:


df = levantar_datos("../../"+DATASET_RELATIVE_PATH)


# In[19]:


df.info()


# In[15]:


grouped = df.groupby(["ano","mes"]).aggregate({"precio": "count"}).unstack()
grouped.columns = grouped.columns.droplevel()


# In[16]:


grouped


# In[18]:


get_heatmap(grouped, title="Cantidad de publicaciones por año/mes")


# In[23]:


get_barplot(df.tipodepropiedad.value_counts().head(5))


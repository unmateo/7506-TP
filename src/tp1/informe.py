#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[30]:


import pandas as pd
import numpy

import nltk
from nltk.corpus import stopwords
from string import punctuation
import re
from unidecode import unidecode
from collections import Counter

import geopandas
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


get_ipython().run_line_magic('run', '"../utils/dataset_parsing.ipynb"')
get_ipython().run_line_magic('run', '"../utils/graphs.ipynb"')
pd.set_option("display.max_colwidth", -1)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# # Levantamos los datos

# In[11]:


df = levantar_datos()
df.info()


# # Analizamos las características generales del dataset

# In[ ]:





# ## Cantidad de publicaciones

# In[13]:


grouped = df.groupby(["ano","mes"]).aggregate({"id": "count"}).unstack()
grouped.columns = grouped.columns.droplevel()
plot = get_heatmap(grouped, title="Cantidad de publicaciones por año/mes")


# ## Publicaciones por tipo de propiedad

# In[12]:


waffle_tipo = get_waffleplot(df.tipodepropiedad.value_counts(normalize=True).head(10), "Publicaciones por tipo de propiedad")


# ### Distribucion de las publicaciones por año

# In[20]:


por_anio = df.groupby("ano").agg({"ano":"count"}).rename(columns={'ano':'cantidad'})
total_publicaciones = df.shape[0]
por_anio_porcentual = por_anio.apply(lambda x: round(100 * x/total_publicaciones, 2), axis=0)
#histograma de ditribucion por año
get_barplot(por_anio_porcentual.cantidad, title="Distribución de las publicaciones por año", x_label="Años", y_label="Cantidad (%)")


# In[22]:


por_tipo = df.groupby(["tipodepropiedad"]).agg({"tipodepropiedad":"count"}).unstack(fill_value=0).sort_values(ascending=False)
por_tipo.index = por_tipo.index.droplevel()
total_publicaciones = por_tipo.sum()
por_tipo_porcentual = por_tipo.apply(lambda x: round(100 * x/total_publicaciones, 2))
por_tipo_porcentual
por_tipo_porcentual.plot(kind='bar', figsize=(15, 5), rot=70, 
                                       title="Cantidad de propiedades por tipo de propiedad");


# #### En base a los resultados de la tabla anterior, nos quedamos con los 5 tipos de propiedad más frecuentes en las publicaciones (96.41% del total) para poder sintetizar mejor los gráficos.

# In[25]:


top_5_tipos  = por_tipo_porcentual.head(5).index.values
df_top_5 = df.loc[df["tipodepropiedad"].isin(top_5_tipos)]
por_tipo_ano = df_top_5.groupby(["ano","tipodepropiedad"]).agg({"tipodepropiedad":"count"}).unstack(fill_value=0)
por_tipo_ano.columns = por_tipo_ano.columns.droplevel()
totales = por_tipo_ano.sum(axis=1)
por_tipo_ano_porcentual = por_tipo_ano.apply(lambda x: round(100 * x/totales[x.index], 2), axis=0)
por_tipo_ano_porcentual


# #### Representamos los datos de la tabla superior en el siguiente grafico

# In[26]:


por_tipo_ano_porcentual.plot(kind='line', xticks=por_tipo_ano_porcentual.index.values, title= "Tipo de vivienda por año")


# ## Precios a lo largo del tiempo

# #### Se buscó información de las cotizaciones del peso mexicano respecto del dolar en el período que se corresponse a los datos de las publicaciones. El objetivo es analizar si los precios de los inmuebles en dólares se mantienen constantes o si aumentan a medida que avanza el tiempo.

# In[30]:


#Se carga el df de cotizaciones
cotizaciones = pd.read_csv("../enunciado/data/cotizacion.csv", dtype={'cotizacion':'float'}, parse_dates=["fecha"])
cotizaciones['dia'] = cotizaciones['fecha'].dt.day
cotizaciones['mes'] = cotizaciones['fecha'].dt.month
cotizaciones['anio'] = cotizaciones['fecha'].dt.year


# In[31]:


#Se calcula cotización promedio por mes
cotizaciones_por_mes = cotizaciones.groupby(['anio', 'mes']).agg({'cotizacion':'mean'})
cotizaciones_por_mes = cotizaciones_por_mes.reset_index()
cotizaciones_por_mes


# #### Se agrega la columna mes_anio tanto en el df de cotizaciones como en el df de publicaciones para poder joinear y calcular el precio en dolares de las propiedades

# In[36]:


cotizaciones_por_mes['anio_mes'] = cotizaciones_por_mes['anio'].astype('str') + cotizaciones_por_mes['mes'].astype('str').str.zfill(width=2)
df['anio_mes'] = df['ano'].astype('str') + df['mes'].astype('str').str.zfill(width=2)

df = pd.merge(df, cotizaciones_por_mes, on='anio_mes')
df['precio_en_dolares'] = df['precio'] / df['cotizacion']
df[['precio', 'precio_en_dolares', 'cotizacion']]


# #### Graficamos el precio promedio en dolares a lo largo del período estudiado

# In[39]:


precio_promedio_en_dolares = df.groupby(['ano','mes_x']).agg({'precio_en_dolares':'mean'})
ax = precio_promedio_en_dolares.plot(figsize=(18,12), title="Variación de precios en dólares")
ax.set_xlabel("Año y mes")
ax.set_ylabel("Precio (Dólares)")


# #### Graficamos el precio promedio en pesos mexicanos a lo largo del período estudiado

# In[42]:


precio_promedio_en_pesos_mexicanos = df.groupby(['ano','mes_x']).agg({'precio':'mean'})
ax = precio_promedio_en_pesos_mexicanos.plot(figsize=(18,12), title="Variación de precios en el tiempo")
ax.set_xlabel("Año y mes")
ax.set_ylabel("Precio (Pesos Mexicanos)")


# #### Graficamos la variación del precio del dólar en el período

# In[45]:


cotizaciones_por_anio_mes = cotizaciones_por_mes.set_index(['anio', 'mes'])['cotizacion']
ax = cotizaciones_por_anio_mes.plot(figsize=(15,10), title="Variación de cotización del dólar en el tiempo")
ax.set_xlabel("Año y mes")
ax.set_ylabel("Cotización (Pesos Mexicanos/Dólar)")


# ## Publicaciones repetidas

# ### Consideramos que una publicación es igual a otra si comparten ciudad, precio, direccion, tipo de propiedad y metros totales. 

# In[51]:


repetidas = df.groupby(['ciudad','provincia','precio','direccion','metrostotales','tipodepropiedad']).agg({"id":"count"})
repetidas=repetidas.loc[repetidas.id>1]
repetidas


# #### La cantidad de publicaciones repetidas según nuestro criterio no es significativa frente al total de los datos. Quisieramos mencionar que en el caso de diferentes departamentos con iguales caracteristicas en un mismo edificio, las publicaciones matchearán como repetidas.

# In[52]:


repetidas=repetidas.groupby("tipodepropiedad").agg({"id":"count"})
repetidas=repetidas.rename(columns={"id":"total"})
get_barplot(repetidas["total"], title="Tipo de propiedad de las publicaciones repetidas", x_label="Tipo de propiedad", y_label="Total",)


# #### El grafico nos permite ver que la cantidad de apartamentos repetidos es muy baja, de modo que la influencia de departamentos iguales en un edificio es casi nula.

# # Conformación del precio

# ### Hipótesis
# 
# - En las zonas urbanas debería haber mayor cantidad de apartamentos que cualquier otro tipo de propiedad
# - Si consideramos características similares los departamentos deberían ser más caros que las casas
# - Las casas tienen más metros cuadrados que los departamentos (en general)
# - El factor más importante para el precio de una vivienda son el tipo de propiedad y los metros cuadrados
# - Las publicaciones de lugares comerciales deberían estar concentradas en lugares más céntricos
# 

# In[53]:


import seaborn as sns


# ## Porcentaje de publicaciones de cada tipo de propiedad en cada provincia
# Se quiere saber qué es lo que más se publica en cada provincia. Seguramente la mayor cantidad de publicaciones sean de departamentos o casas. En centros urbanos muy poblados, como el distrito federal, seguramente haya mayor cantidad de publicaciones de departamentos, mientras que en zonas no tan urbanizadas la cantidad de publicaciones de casas sean predominantes.

# In[58]:


#Se obtiene un dataframe con la cantidad de publicaciones de cada provincia
publicaciones_por_provincia = df.groupby('provincia').agg({'id':'count'})
publicaciones_por_provincia_y_tipo = df.groupby(['provincia', 'tipodepropiedad']).agg({'id':'count'}).reset_index()
publicaciones_por_provincia_y_tipo = pd.merge(publicaciones_por_provincia_y_tipo, publicaciones_por_provincia, on='provincia')
publicaciones_por_provincia_y_tipo = publicaciones_por_provincia_y_tipo.rename(columns={'id_x':'cantidad_tipo', 'id_y':'cantidad_total'})
publicaciones_por_provincia_y_tipo['porcentaje'] = publicaciones_por_provincia_y_tipo['cantidad_tipo'] / publicaciones_por_provincia_y_tipo['cantidad_total']
publicaciones_por_provincia_y_tipo = publicaciones_por_provincia_y_tipo.set_index(['provincia', 'tipodepropiedad'])


# In[59]:


publicaciones_por_provincia_y_tipo_for_heatmap = publicaciones_por_provincia_y_tipo.pivot_table(index='provincia', columns='tipodepropiedad', values='porcentaje', aggfunc='sum')
get_heatmap(publicaciones_por_provincia_y_tipo_for_heatmap, xlabel="Tipo de propiedad", ylabel="Provincia", title="Porcentajes por tipo de propiedad y provincia")


# #### Se puede visualizar en el heatmap que en general la cantidad de publicaciones de casas es muy grande, en relación con los demás tipos de propiedades.
# 
# Efectivamente en el distrito federal la cantidad de departamento es muy alta en comparación con las casas. Lo mismo ocurre en Guerrero.

# ## Relación entre tipo de propiedad y precio
# A continuación intentaremos determinar el impacto que tiene el tipo de propiedad en el precio. Para ello nos concetraremos primero en el grupo de tipos de propiedad correspondiente a las viviendas. Este grupo tendrá los inmuebles de tipo: Apartamento, Casa, Casa en condominio, Casa uso de suelo, Departamento Compartido y Duplex.

# In[61]:


# Obtenemos un df con las publicaciones de viviendas
tipo_vivienda = ['Apartamento', 'Casa', 'Casa en condominio', 'Casa uso de suelo', 'Departamento Compartido', 'Duplex']
publicaciones_viviendas = df.set_index('tipodepropiedad')
publicaciones_viviendas = publicaciones_viviendas.filter(items=tipo_vivienda, axis=0)
publicaciones_viviendas.reset_index(inplace=True)


# Por ahora tomemos este df tal como está y comparemos los precios utilizando un Boxplot.

# In[63]:


get_boxplot(publicaciones_viviendas, 'tipodepropiedad', 'precio', (15,5), label_x='Tipo de propiedad', label_y='Precio', title='Precio según tipo de propiedad')


# Tener en cuenta que la escala de precios es 1e7, por lo tanto 0.2 es 2,000,000.00.
# A simple vista los apartamentos, las casas y las casas en condominio tienen precios similares (en promedio).
# El problema de este análisis es que no se tiene en cuenta que existen otros factores que podrían variar el precio, por ejemplo los metros cuadrados. Es decir si una casa y un apartamento tienen metros cuadrados similares, ¿El precio también es similar?.

# ### Relación entre tipo de propiedad, metros cuadrados y precio
# 
# Primero verifiquemos que todas las propiedades tienen los valores de metros cuadrados.

# In[64]:


publicaciones_viviendas.isnull().sum()


# In[65]:


publicaciones_viviendas.loc[(publicaciones_viviendas['metroscubiertos'].isnull()) & (publicaciones_viviendas['metrostotales'].isnull())]


# Como se observa hay publicaciones que no tienen los metros cuadrados cubiertos o bien no tienen los metros cuadrados totales, pero no hay ninguna que no tenga ninguno de los 2 datos. Esto podría indicar que en caso que los metros totales estén en null, significa que los metros cubiertos son los metros totales y viceversa. Para no dejar fuera del análisis estas publicaciones consideraremos que lo dicho anteriormente es cierto, por lo tanto actualizaremos los valores.

# In[67]:


publicaciones_viviendas.loc[publicaciones_viviendas['metroscubiertos'].isnull(), ['metroscubiertos']] = publicaciones_viviendas['metrostotales']
publicaciones_viviendas.loc[publicaciones_viviendas['metrostotales'].isnull(), ['metrostotales']] = publicaciones_viviendas['metroscubiertos']
#Verificamos que no haya valores en 0
publicaciones_viviendas.loc[publicaciones_viviendas['metrostotales'] == 0]


# Ahora que todas las propiedades tienen valores en el campo metros cuadrados es necesario determinar un rango de análisis. Análiamos con un histograma en qué rangos se encuentran aproximadamente la mayor cantidad de publicaciones.

# In[69]:


get_hist(publicaciones_viviendas["metrostotales"], bins=50)


# Para que la cantidad de muestras entre los distintos tipos de propiedades sea similar vamos a tomar el rango 120 a 130

# In[71]:


publicaciones_entre_120_y_130 = publicaciones_viviendas.loc[(publicaciones_viviendas["metrostotales"] >= 120) & (publicaciones_viviendas["metrostotales"] <= 130)]
publicaciones_entre_120_y_130.groupby("tipodepropiedad").agg({"id":"count"})


# In[72]:


get_boxplot(publicaciones_entre_120_y_130, 'tipodepropiedad', 'precio', (15,5))


# De este gráfico sólo podemos considerar válidos los datos de departamentos, casas y casas en condominio, dado que la cantidad de muestras de los demás tipos de propiedades no es representativo.
# Si consideramos propiedades de metros similares vemos que los departamentos son más caros. Entonces por qué en el gráfico anterior mostraba que las casas y los departamentos tienen precios similares y aquí no ocurre eso? Es de suponer que la casas tienen más metros cuadrados que los departamentos en general, por lo tanto si tomamos las propiedades sin limitar las demás características las casas tienden a tener precios similares a los departamentos por la cantidad de metros cuadrados. Lo anterior se puede verificar con el siguiente gráfico.

# In[74]:


get_boxplot(publicaciones_viviendas, 'tipodepropiedad', 'metrostotales', (15,5), title="Metros totales según tipo de propiedad", label_x="Tipo de propiedad", label_y="Metros totales")


# En el siguiente gráfico se puede observar el cambio de precio según la cantidad de metros cuadrados, por tipo de propiedad.

# In[75]:


publicaciones_apartamentos = publicaciones_viviendas.loc[publicaciones_viviendas['tipodepropiedad'] == 'Apartamento']
publicaciones_casas = publicaciones_viviendas.loc[publicaciones_viviendas['tipodepropiedad'] == 'Casa']
publicaciones_casas_condominio = publicaciones_viviendas.loc[publicaciones_viviendas['tipodepropiedad'] == 'Casa en condominio']
f, ax = plt.subplots(figsize=(10,5))
ax.set(xlim=(0, 500), ylim=(0, 14000000))
#sns.scatterplot(x='metrostotales', y='precio', data=publicaciones_viviendas.loc[publicaciones_viviendas['tipodepropiedad'] == 'Apartamento'])
sns.regplot(x=publicaciones_apartamentos['metrostotales'],y=publicaciones_apartamentos['precio'],color='blue',scatter=False)
sns.regplot(x=publicaciones_casas['metrostotales'],y=publicaciones_casas['precio'],color='magenta',scatter=False)
sns.regplot(x=publicaciones_casas_condominio['metrostotales'],y=publicaciones_casas_condominio['precio'],color='green',scatter=False)

plt.legend(labels=['Apartamentos','Casas','Casas en condominio'])
plt.title('Relación metros totales y precio', size=24)
plt.xlabel('Metros total (m2)', size=18)
plt.ylabel('Precio (Pesos Mexicanos)', size=18);


# ### Antigüedad

# In[76]:


f, ax=plt.subplots(figsize=(10,5))
ax.set(xlim=(0, 100), ylim=(0, 14000000))
sns.regplot(x='antiguedad',y='precio',data=publicaciones_viviendas, color='blue',scatter=False)

plt.title('Relación entre antigüedad y precio', size=24)
plt.xlabel('Antigüedad (años)', size=18)
plt.ylabel('Precio (Pesos Mexicanos)', size=18);


# In[77]:


plt.subplots(figsize=(10,5))
sns.scatterplot(x='antiguedad', y='precio', data=publicaciones_viviendas)


# Como se aprecia en la línea de regresión el precio no varía practicamente con la antiguedad. Al haber pocas publicaciones con muchos años de antigüedad este valor también es más impreciso.
# 
# El aumento de precio a medida que la antigüedad es mayor también puede deberse a que hay mayor cantidad de metros cuadrados en construcciones más antiguas.

# In[78]:


f, ax=plt.subplots(figsize=(10,5))
ax.set(xlim=(0, 100), ylim=(0, 500))
sns.regplot(x='antiguedad',y='metrostotales',data=publicaciones_viviendas, color='blue',scatter=False)

plt.title('Relación entre antigüedad y metros totales', size=24)
plt.xlabel('Antigüedad (años)', size=18)
plt.ylabel('Metros totales (m2)', size=18);


# ### Baños y habitaciones

# In[79]:


convert_dict={'banos':'int'}
publicaciones_viviendas_con_bano = publicaciones_viviendas.loc[~publicaciones_viviendas['banos'].isnull(), ['banos','metroscubiertos']].astype(convert_dict)
ax = publicaciones_viviendas_con_bano.pivot(columns='banos')['metroscubiertos'].plot(kind = 'hist', stacked=True, figsize=(10,10), title="Relación cantidad de baños y metros cubiertos")
ax.set_xlabel("Metros cubiertos (m2)")


# In[80]:


convert_dict={'habitaciones':'int'}
publicaciones_viviendas_con_habitaciones = publicaciones_viviendas.loc[~publicaciones_viviendas['habitaciones'].isnull(), ['habitaciones','metroscubiertos']].astype(convert_dict)
ax = publicaciones_viviendas_con_habitaciones.pivot(columns='habitaciones')['metroscubiertos'].plot(kind = 'hist', stacked=True, figsize=(10,10), title="Relación cantidad de habitaciones y metros cubiertos")
ax.set_xlabel("Metros cubiertos (m2)")


# In[81]:


precio_viviendas_con_bano_90_a_100 = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['metrostotales'] >= 90) & (publicaciones_viviendas['metrostotales'] <= 100)]
get_boxplot(precio_viviendas_con_bano_90_a_100, 'banos', 'precio', (15,5), title="Precio según cantidad de baños", label_x="Cantidad de baños", label_y="Precio")


# In[82]:


precio_viviendas_con_bano_70_a_80 = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['metrostotales'] >= 70) & (publicaciones_viviendas['metrostotales'] <= 80)]
get_boxplot(precio_viviendas_con_bano_70_a_80, 'banos', 'precio', (15,5), title="Precio según cantidad de baños", label_x="Cantidad de baños", label_y="Precio")


# In[83]:


precio_viviendas_con_bano_50_a_60 = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['metrostotales'] >= 50) & (publicaciones_viviendas['metrostotales'] <= 60)]
get_boxplot(precio_viviendas_con_bano_50_a_60, 'banos', 'precio', (15,5), title="Precio según cantidad de baños", label_x="Cantidad de baños", label_y="Precio")


# In[84]:


publicaciones_1_bano = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['banos'] == 1)]
publicaciones_2_bano = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['banos'] == 2)]
publicaciones_3_bano = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['banos'] == 3)]
publicaciones_4_bano = publicaciones_viviendas.loc[(~publicaciones_viviendas['banos'].isnull()) & (publicaciones_viviendas['banos'] == 4)]
f, ax = plt.subplots(figsize=(15,10))
ax.set(xlim=(0, 500), ylim=(0, 14000000))
#sns.scatterplot(x='metrostotales', y='precio', data=publicaciones_viviendas.loc[publicaciones_viviendas['tipodepropiedad'] == 'Apartamento'])
sns.regplot(x=publicaciones_1_bano['metrostotales'],y=publicaciones_1_bano['precio'],color='blue',scatter=False)
sns.regplot(x=publicaciones_2_bano['metrostotales'],y=publicaciones_2_bano['precio'],color='magenta',scatter=False)
sns.regplot(x=publicaciones_3_bano['metrostotales'],y=publicaciones_3_bano['precio'],color='green',scatter=False)
sns.regplot(x=publicaciones_4_bano['metrostotales'],y=publicaciones_4_bano['precio'],color='yellow',scatter=False)

plt.legend(labels=['1 baño','2 baños','3 baños', '4 baños'])
plt.title('Relación metros totales y precio según cantidad de baños', size=24)
plt.xlabel('Metros total (m2)', size=18)
plt.ylabel('Precio (Pesos Mexicanos)', size=18);


# In[85]:


publicaciones_1_habitacion = publicaciones_viviendas.loc[(~publicaciones_viviendas['habitaciones'].isnull()) & (publicaciones_viviendas['habitaciones'] == 1)]
publicaciones_2_habitacion = publicaciones_viviendas.loc[(~publicaciones_viviendas['habitaciones'].isnull()) & (publicaciones_viviendas['habitaciones'] == 2)]
publicaciones_3_habitacion = publicaciones_viviendas.loc[(~publicaciones_viviendas['habitaciones'].isnull()) & (publicaciones_viviendas['habitaciones'] == 3)]
publicaciones_4_habitacion = publicaciones_viviendas.loc[(~publicaciones_viviendas['habitaciones'].isnull()) & (publicaciones_viviendas['habitaciones'] == 4)]
publicaciones_5_habitacion = publicaciones_viviendas.loc[(~publicaciones_viviendas['habitaciones'].isnull()) & (publicaciones_viviendas['habitaciones'] == 5)]
publicaciones_6_habitacion = publicaciones_viviendas.loc[(~publicaciones_viviendas['habitaciones'].isnull()) & (publicaciones_viviendas['habitaciones'] == 6)]

f, ax = plt.subplots(figsize=(15,10))
ax.set(xlim=(0, 500), ylim=(0, 14000000))
#sns.scatterplot(x='metrostotales', y='precio', data=publicaciones_viviendas.loc[publicaciones_viviendas['tipodepropiedad'] == 'Apartamento'])
sns.regplot(x=publicaciones_1_habitacion['metrostotales'],y=publicaciones_1_habitacion['precio'],color='blue',scatter=False)
sns.regplot(x=publicaciones_2_habitacion['metrostotales'],y=publicaciones_2_habitacion['precio'],color='magenta',scatter=False)
sns.regplot(x=publicaciones_3_habitacion['metrostotales'],y=publicaciones_3_habitacion['precio'],color='green',scatter=False)
sns.regplot(x=publicaciones_4_habitacion['metrostotales'],y=publicaciones_4_habitacion['precio'],color='yellow',scatter=False)
sns.regplot(x=publicaciones_5_habitacion['metrostotales'],y=publicaciones_5_habitacion['precio'],color='violet',scatter=False)
sns.regplot(x=publicaciones_6_habitacion['metrostotales'],y=publicaciones_6_habitacion['precio'],color='brown',scatter=False)

plt.legend(labels=['1 habitación','2 habitaciones','3 habitaciones', '4 habitaciones', '5 habitaciones', '6 habitaciones'])
plt.title('Relación metros totales y precio según cantidad de habitaciones', size=24)
plt.xlabel('Metros total (m2)', size=18)
plt.ylabel('Precio (Pesos Mexicanos)', size=18);


# ## Amenities y cercanías

# Analizamos la influencia de los amenities y las cercanías a escuelas y/o centro comerciales

# In[88]:


df['tiene_amenities'] = (df["gimnasio"] + df["usosmultiples"] + df["piscina"]) > 0
df['tiene_cercanias'] = (df["centroscomercialescercanos"] + df["escuelascercanas"]) > 0


# In[91]:


ameneties=df.groupby(["tiene_amenities"]).agg({"id":"count"})
ameneties=ameneties.rename(columns={"id":"cantidad"})
get_barplot(ameneties["cantidad"], title="¿Tiene amenities?", y_label="Cantidad de publicaciones")


# Vemos como se distribuyen los amenities entre las publicaciones según el tipo de propiedad del que se trate

# In[92]:


amenities_por_tipo = df.groupby(["tipodepropiedad","tiene_amenities"]).agg({"id":"count"}).unstack(fill_value=0)
amenities_por_tipo.columns = ["No", "Sí"]
amenities_por_tipo = (amenities_por_tipo.div(amenities_por_tipo.sum(axis=1), axis=0) * 100).apply(lambda x: round(x,0)).sort_values(by="Sí", ascending=True)
plot = amenities_por_tipo.plot(kind = 'barh', stacked=True, figsize=(10,10))
plot.legend(labels=["Sin amenities", "Con amenities"],loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_title("Porcentaje de publicaciones con amenities por tipo de propiedad", fontdict={"fontsize": 18})
plot.set_xlabel("Porcentaje")
plot.set_ylabel("Tipo de propiedad")
plt.tight_layout()


# Vemos como afecta al precio la precencia de amenities a las publicaciones según el tipo de propiedad del que se trate

# In[96]:


tipos_con_amenities = amenities_por_tipo.loc[amenities_por_tipo["No"] < 100].index.values
precio_amenities_por_tipo = df.loc[df["tipodepropiedad"].isin(tipos_con_amenities)].groupby(["tipodepropiedad","tiene_amenities"]).agg({"precio_metro_cubierto":"mean"}).dropna().unstack()
precio_amenities_por_tipo.columns = ["Sin amenities", "Con amenities"]
plot = precio_amenities_por_tipo.plot(kind = 'barh', stacked=False, figsize=(10,10), title="Precio de las publicaciones con y sin amenities por tipo de propiedad")
plot.set_xlabel("Precio metro_cubierto")
plot.set_ylabel("Tipo de propiedad")
plot.set_title("Precio de las publicaciones con y sin amenities por tipo de propiedad", fontdict={"fontsize": 18})


# Analizamos la influencia de cercanías a escuelas y/o centro comerciales

# In[98]:


cercanias = df.groupby(["tiene_cercanias"]).agg({"id":"count"})
cercanias = cercanias.rename(columns={"id":"cantidad"})
get_barplot(cercanias["cantidad"], title="¿Tiene alguna cercanía?", y_label="Cantidad de publicaciones")


# Vemos como se distribuyen las cercanías entre las publicaciones según el tipo de propiedad del que se trate

# In[99]:


cercanias_por_tipo = df.groupby(["tipodepropiedad","tiene_cercanias"]).agg({"id":"count"}).unstack(fill_value=0)
cercanias_por_tipo.columns = ["No", "Sí"]
cercanias_por_tipo = (cercanias_por_tipo.div(cercanias_por_tipo.sum(axis=1), axis=0) * 100).apply(lambda x: round(x,0)).sort_values(by="Sí", ascending=True)
plot = cercanias_por_tipo.plot(kind = 'barh', stacked=True, figsize=(10,10))
plot.legend(labels=["Sin cercanías", "Con cercanías"],loc='center left', bbox_to_anchor=(1, 0.5))
plot.set_title("Porcentaje de publicaciones con cercanías por tipo de propiedad", fontdict={"fontsize": 18})
plot.set_xlabel("Porcentaje")
plot.set_ylabel("Tipo de propiedad")
plt.tight_layout()  


# Vemos como afectan al precio las cercanías, según el tipo de propiedad del que se trate

# In[100]:


tipos_con_cercanias = cercanias_por_tipo.loc[cercanias_por_tipo["No"] < 100].index.values
precio_cercanias_por_tipo = df.loc[df["tipodepropiedad"].isin(tipos_con_cercanias)].groupby(["tipodepropiedad","tiene_amenities"]).agg({"precio_metro_cubierto":"mean"}).dropna().unstack()
precio_cercanias_por_tipo.columns = ["Sin cercanias", "Con cercanias"]
plot = precio_cercanias_por_tipo.plot(kind = 'barh', stacked=False, figsize=(10,10), title="Precio de las publicaciones con y sin cercanias por tipo de propiedad")
plot.set_xlabel("Precio metro_cubierto")
plot.set_ylabel("Tipo de propiedad")
plot.set_title("Precio de las publicaciones con y sin cercanias por tipo de propiedad", fontdict={"fontsize": 18})


# ## Título y Descripción

# In[15]:


# creo el set data stopwords
spanish_stopwords = set(stopwords.words('spanish'))
non_words = set(punctuation)
non_words.update({'¿', '¡'})
non_words.update(map(str,range(10)))

# agrego palabras comunes que detecte y no son relevantes
stopwords_extra = {"fracc", "cada", "mas", "ntilde", "consta", "tres", "dos", "solo", "cuenta", "areas", "tipo", "nbsp", "oacute", "hrs", "aacute", "palapa", "easybroker", "tarja", "cuatro", "uacute", "cancel", "asi", "hace", "tan", "dia", "ningun" }


# In[16]:


def is_meaningful(word: str) -> bool:
    """
        Recibe una palabra, remueve puntuaciones y verifica que lo que queda no esté en el set de stopwords
    """
    return len(word) > 2 and not word in spanish_stopwords

def remove_html(field: str) -> str:
    """
        Recibe un texto y devuelve una copia sin los tags html
    """
    return re.compile(r'<[^>]+>').sub('', field) if field else field

def normalize(field: str) -> str:
    """
        Recibe un texto y devuelve una copia sin acentos, ñ ni puntuaciones.
    """
    return ''.join([" " if c in non_words else unidecode(c) for c in field]).strip() if field else ""

def limpiar_campo(field: str) -> str:
    """
        Recibe un campo string que podría tener muchas palabras.
        Devuelve un string que contiene sólo las palabras significativas.
    """
    if not isinstance(field,str): return ""
    without_html = remove_html(field)
    normalized = normalize(without_html)
    meaningful = " ".join(set(filter(is_meaningful, normalized.split())))
    return meaningful

def get_word_counter(series):
    """
        Faltaría analizar stemming
    """
    counter = Counter()
    for title in series.values:
        counter.update(set(title.split()))
    return counter


# In[17]:


# agrego al df los campos de titulo y descripcion procesados

df["descripcion_limpia"] = df["descripcion"].map(limpiar_campo)
df["len_descripcion"] = df["descripcion_limpia"].map(lambda x: len(x.split()))
df["titulo_limpio"] = df["titulo"].map(limpiar_campo)
df["len_titulo"] = df["titulo_limpio"].map(lambda x: len(x.split()))


# In[18]:


plot = df["len_descripcion"].plot(kind="hist",figsize=(24,12),xticks=range(0,500,20), bins=25, logy=True, cmap="summer")
fontdict={"fontsize":18}
plot.set_title("Palabras en descripcion (log)", fontdict=fontdict)
plot.set_xlabel("Cantidad de palabras en descripcion", fontdict=fontdict)
plot.set_ylabel("Cantidad de publicaciones", fontdict=fontdict)
plot.figure.savefig("./graficos/hist_palabras_descripcion")


# In[19]:


plot = get_barplot(df["len_titulo"].value_counts().sort_index(), title="Palabras en título", x_label="Cantidad de palabras en título", y_label="Cantidad de publicaciones")
plot.figure.savefig("./graficos/barplot_palabras_titulo.png")


# In[20]:


titulo_palabras = get_word_counter(df["titulo_limpio"])
wc_titulo = get_wordcloud(titulo_palabras)
wc_titulo.to_file("./graficos/wordcloud_titulo.png")


# In[21]:


descripcion_palabras = get_word_counter(df["descripcion_limpia"])
wc_descripcion = get_wordcloud(descripcion_palabras)
wc_descripcion.to_file("./graficos/wordcloud_descripcion.png")


# In[22]:


palabras_positivas = {"vigilancia","hermosa","diseño","vistas","playa","conservacion","tenis","balcon","panoramica","exclusivos","golf","canchas","remodelada","acondicionado","lujo","jacuzzi","diseno","exclusiva","magnifica","exclusivo","country","precioso","estilo","seguridad","verdes","juegos","servicio","excelente","terraza","jardin","hermosa","vista","bonita","renta", "granito","porcelanato","mejores"}
palabras_negativas = {"oportunidad","remato","oferta","remodelar", "inversion"}


# In[23]:


df["palabras_positivas_descripcion"] = df["descripcion_limpia"].map(lambda x: " ".join([y for y in x.split() if y in palabras_positivas]))
df["cantidad_palabras_positivas_descripcion"] = df["palabras_positivas_descripcion"].map(lambda x: len(x.split()))
df[["cantidad_palabras_positivas_descripcion","precio_metro_total"]].corr()


# In[24]:


df["palabras_negativas_descripcion"] = df["descripcion_limpia"].map(lambda x: " ".join([y for y in x.split() if y in palabras_negativas]))
df["cantidad_palabras_negativas_descripcion"] = df["palabras_negativas_descripcion"].map(lambda x: len(x.split()))
df[["cantidad_palabras_negativas_descripcion","precio_metro_total"]].corr()


# In[25]:


counter_positivas = get_word_counter(df["palabras_positivas_descripcion"])
wc_positivas = get_wordcloud(counter_positivas)
wc_positivas.to_file("./graficos/wordcloud_positivas.png")


# In[26]:


counter_negativas = get_word_counter(df["palabras_negativas_descripcion"])
wc_negativas = get_wordcloud(counter_negativas)
wc_negativas.to_file("./graficos/wordcloud_negativas.png")


# In[27]:


plot = get_barplot(df.cantidad_palabras_negativas_descripcion.value_counts(), title="Palabras negativas en descripción", x_label="Cantidad de palabras negativas en descripción", y_label="Cantidad de publicaciones")
plot.figure.savefig("./graficos/barplot_palabras_negativas_descripcion.png")


# In[28]:


plot = get_barplot(df.cantidad_palabras_positivas_descripcion.value_counts().sort_index(), title="Palabras positivas en descripción", x_label="Cantidad de palabras positivas en descripción", y_label="Cantidad de publicaciones")
plot.figure.savefig("./graficos/barplot_palabras_positivas_descripcion.png")


# # Información geográfica

# In[96]:


# creo los geodataframes con información geográfica de México
pais = geopandas.read_file("lab/MEX_adm/MEX_adm0.shp")
estados = geopandas.read_file("lab/MEX_adm/MEX_adm1.shp")


# In[97]:


# le cambio los nombres a las siguientes provincias, para que coincidan con mi info geografica 
def fix_provincias(df, provincias) -> bool:
    provincias_mapper = {
        "Baja California Norte": "Baja California",
        "Edo. de México": "México",
        "San luis Potosí": "San Luis Potosí"
    }
    df["provincia"] = df["provincia"].map(lambda x: provincias_mapper.get(x, x))
    return set(df["provincia"].dropna().unique()) == set(provincias["NAME_1"]) #verifico
fix_provincias(df, estados)


# In[34]:


# creo las columnas para distinguir los puntos en el mapa
df["tiene_gps"] = ~ (df["lat"].isna() & df["lng"].isna())
crear_punto = lambda x: Point(x["lng"],x["lat"]) if x["tiene_gps"] else None
df["coord"] = df.apply(crear_punto, axis=1)
df["en_mexico"] = df.loc[df["tiene_gps"]]["coord"].map(esta_en_mexico)
# selecciono los puntos que tienen informacion geografica
geoDF = geopandas.GeoDataFrame(df.loc[df["tiene_gps"] & df["en_mexico"]], geometry="coord")


# In[70]:


def choropleth_estados(estados, columna, titulo=""):
    """
        Dibuja en un gráfico un chloropeth de los estados de méxico según la columna que reciba
    """
    plot = estados.plot(column=columna, legend=True, figsize=(24,8), cmap="Greens")    
    plot.set_title(titulo, fontdict={"fontsize": 18})
    plot.set_xlabel("Longitud")
    plot.set_ylabel("Latitud")
    return plot


# In[69]:


df_poblacion = pd.read_csv("lab/poblacion_por_estado.csv", index_col="NAME_1")
estados["poblacion"] = estados["NAME_1"].map(df_poblacion["poblacion"])
plot = choropleth_estados(estados, "poblacion", "Población de México por estado")


# In[47]:


# cuento la cantidad de publicaciones y las agrego al df de estados
publicaciones_por_estado = geoDF.loc[~geoDF["estado"].isna()].groupby(["estado"]).agg({"estado":"count"})
publicaciones_por_estado.columns = ["publicaciones"]
estados = estados.merge(left_on="NAME_1", right_on="estado", right=publicaciones_por_estado)


# In[71]:


plot = choropleth_estados(estados, "publicaciones", "Cantidad de publicaciones por estado")
plot.figure.savefig("graficos/map_publicaciones_por_estado.png")


# In[72]:


estados["publicaciones_poblacion"] = estados["publicaciones"] / estados["poblacion"]
plot = choropleth_estados(estados, "publicaciones_poblacion", "Publicaciones por habitante en cada estado")


# ## Características de zona

# In[101]:


def agg_polygon(point_series):
    """
        Recibe un pd.Series de geometry.Point
        Devuelve un Polygon de los puntos de la serie, o NaN si
        no tiene suficientes puntos (3).
    """
    values = point_series.loc[~point_series.isna()].values
    if len(values) < 3: return numpy.NaN
    return Polygon([[p.x, p.y] for p in values])


# In[105]:


calculations = ["mean","std","max","min"]
aggregations = {"id": "count",                "precio_metro_total": calculations,                "precio_metro_cubierto": calculations,                "antiguedad": calculations,                "habitaciones": calculations,                "metroscubiertos": calculations,                "metrostotales": calculations,                "lat": calculations,                "lng": calculations,                "precio": calculations,                "habitaciones": calculations,                "garages": calculations,                "banos": calculations,                "gps": agg_polygon               }
zonas = df.groupby(["idzona"]).agg(aggregations)
zonas.columns = [x+"_"+y for x,y in zonas.columns]
zonas.rename({"gps_agg_polygon": "polygon"}, axis="columns", inplace=True)
zonas["lat_dif"] = zonas["lat_max"] - zonas["lat_min"]
zonas["lng_dif"] = zonas["lng_max"] - zonas["lng_min"]


# In[86]:


# analizo precios promedio por zona
minima_cantidad_publicaciones = zonas["id_count"].mean() + zonas["id_count"].std()
zonas_con_mas_publicaciones = zonas.loc[zonas["id_count"] > minima_cantidad_publicaciones ]
titulo = "Precio promedio de metros totales en las {} zonas con más de {} publicaciones".format(zonas_con_mas_publicaciones.shape[0], int(minima_cantidad_publicaciones))
plot = get_hist(zonas_con_mas_publicaciones["precio_metro_total_mean"], title=titulo, size=(24,12), xlabel="Precio promedio", ylabel="Cantidad de zonas")
plot.figure.savefig("graficos/hist_precios_zonas")


# In[87]:


titulo = "Desvío estándar de Precio promedio de metros totales en las {} zonas con más de {} publicaciones".format(zonas_con_mas_publicaciones.shape[0], int(minima_cantidad_publicaciones))
plot = get_hist(zonas_con_mas_publicaciones["precio_metro_total_std"], title=titulo, size=(24,12), xlabel="Desvío estándar", ylabel="Cantidad de zonas")
plot.figure.savefig("graficos/hist_desvio_precios_zonas")


# In[98]:


# agrego info de zonas a df estados
zonas_por_estado = df.groupby(["provincia"]).agg({"idzona":"nunique"})
zonas_por_estado.columns = ["cantidad_zonas"]
estados = estados.merge(left_on="NAME_1", right_on="provincia", right=zonas_por_estado, how="left")
estados["cantidad_zonas"] = estados["cantidad_zonas"].fillna(0).astype(int)


# In[99]:


plot = estados.plot(column="cantidad_zonas", legend=True, figsize=(24,8), cmap="Greens")    
plot.set_title("Cantidad de zonas por estado", fontdict={"fontsize": 18})
plot.set_xlabel("Longitud")
plot.set_ylabel("Latitud")
plot.figure.savefig("graficos/map_zonas_por_estado.png")


# In[112]:


def plot_mexico(df, geometry, columna, titulo):
    geoDF = geopandas.GeoDataFrame(df, geometry=geometry)
    base = pais.plot(figsize=(24,12))
    estados_plot = estados.plot(ax=base, color="white")
    plot = geoDF.plot(ax=estados_plot, cmap="viridis_r",legend=True, column=columna)
    plot.set_title(titulo, fontdict={"fontsize": 18})
    plot.set_xlabel("Longitud", fontdict={"fontsize": 18})
    plot.set_ylabel("Latitud", fontdict={"fontsize": 18})
    return plot


zonas_ok = zonas.loc[(zonas["lat_dif"] < zonas["lat_dif"].mean()) & (zonas["lng_dif"] < zonas["lng_dif"].mean())]
zonas_ok.loc[:,"centroid"] = zonas_ok.loc[~zonas["polygon"].isna()]["polygon"].map(lambda x: x.buffer(0).representative_point())
con_centroide = zonas_ok.loc[(~zonas_ok["centroid"].isna())]
en_mexico = con_centroide.loc[con_centroide["centroid"].map(esta_en_mexico)]
publicaciones_minimas = en_mexico["id_count"].mean() + en_mexico["id_count"].std()
en_mexico = en_mexico.loc[en_mexico["id_count"] > publicaciones_minimas]

msg_minimo = " ({} zonas con más de {} publicaciones)".format(en_mexico.shape[0], int(publicaciones_minimas))

id_count = plot_mexico(en_mexico, "centroid", "id_count", "Cantidad de publicaciones por cada zona"+msg_minimo)
id_count.figure.savefig("graficos/map_zonas_mas_publicaciones.png")


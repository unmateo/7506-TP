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


# ## Precios a lo largo del tiempo

# In[ ]:





# ## Distribución geográfica

# In[ ]:





# ## Publicaciones repetidas

# In[ ]:





# # Conformación del precio

# In[ ]:





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


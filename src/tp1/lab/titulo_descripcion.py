#!/usr/bin/env python
# coding: utf-8

# ## Campos involucrados
# 
# - titulo
# - descripcion
# 
# ## Ideas
# 
# - wordcloud
# - normalizacion
# - stemming
# - palabras positivas (respecto al precio)
# - palabras negativas (respecto al precio)
# 
# ## Hipótesis
# 
# - ciertas palabras indican mayor precio (luminoso, jardín, hermoso, vista...)
# - a más palabras, mayor precio
# 
# ## Resultados
# - la correlacion entre longitud de descripcion y precio es bastante baja (0.1)

# In[1]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns
pd.set_option("display.max_colwidth", -1)


# In[2]:


import nltk  
from nltk.corpus import stopwords  
from string import punctuation  


# In[3]:


spanish_stopwords = set(stopwords.words('spanish'))
non_words = set(punctuation)
non_words.update({'¿', '¡'})
non_words.update(map(str,range(10)))


# In[179]:


import re
from unidecode import unidecode

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


# In[182]:


df["descripcion_limpia"] = df["descripcion"].map(limpiar_campo)
df["len_descripcion"] = df["descripcion_limpia"].map(lambda x: len(x.split()))


# In[183]:


df["titulo_limpio"] = df["titulo"].map(limpiar_campo)
df["len_titulo"] = df["titulo_limpio"].map(lambda x: len(x.split()))


# In[184]:


# df["titulo_limpio"].sample(10)
df["descripcion_limpia"].sample(10)


# In[185]:


df[["len_descripcion","precio"]].corr()


# In[148]:


from collections import Counter

def get_word_counter(series):
    """
        Faltaría analizar stemming
    """
    counter = Counter()
    for title in series.values:
        counter.update(set(title.split()))
    return counter


# In[186]:


titulo_palabras = get_word_counter(df["titulo_limpio"])
descripcion_palabras = get_word_counter(df["descripcion_limpia"])


# In[151]:


print(len(titulo_palabras),len(descripcion_palabras))


# In[155]:


# titulo_palabras.most_common(10)


# In[96]:


# descripcion_palabras.most_common(10)


# In[156]:


palabras_positivas = {"conservacion","tenis","balcon","panoramica","exclusivos","golf","canchas","remodelada","acondicionado","lujo","jacuzzi","diseno","exclusiva","magnifica","exclusivo","country","precioso","estilo","seguridad","verdes","juegos","servicio","excelente","terraza","jardin","hermosa","vista","bonita","renta", "granito"}
palabras_negativas = {"oportunidad","remato","oferta","remodelar"}


# In[188]:


df["palabras_positivas_descripcion"] = df["descripcion_limpia"].map(lambda x: len([y for y in x.split() if y in palabras_positivas]))
df[["palabras_positivas_descripcion","precio"]].corr()


# In[189]:


df["palabras_negativas_descripcion"] = df["descripcion_limpia"].map(lambda x: len([y for y in x.split() if y in palabras_negativas]))
df[["palabras_negativas_descripcion","precio"]].corr()


# In[196]:


df.palabras_positivas_descripcion.value_counts()


# In[199]:


# df.loc[df.palabras_positivas_descripcion > 14]["descripcion"]


# In[203]:


df.loc[df.palabras_negativas_descripcion > 2]["descripcion"]


# In[214]:


df_corr_positivas = df[["descripcion_limpia","precio"]]
for palabra in palabras_positivas:
    df_corr_positivas[palabra] = df_corr_positivas["descripcion_limpia"].map(lambda x: int(palabra in x))
df_corr_positivas.corr()["precio"].sort_values(ascending=False)


# In[215]:


df_corr_negativas = df[["descripcion_limpia","precio"]]
for palabra in palabras_negativas:
    df_corr_negativas[palabra] = df_corr_negativas["descripcion_limpia"].map(lambda x: int(palabra in x))
df_corr_negativas.corr()["precio"].sort_values(ascending=True)


# In[216]:


test = df[["descripcion_limpia","precio","metrostotales"]]
for palabra in palabras_positivas:
    test[palabra] = test["descripcion_limpia"].map(lambda x: int(palabra in x))


# In[222]:





# In[239]:


top = list(set(test.corr()["metrostotales"].sort_values(ascending=False).head(8).index).union(set(test.corr()["precio"].sort_values(ascending=False).head(8).index)))


# In[244]:


test_corr = test[top].corr()
test_corr["dif"] = test_corr["precio"] - test_corr["metrostotales"]
test_corr["dif"] = abs(test_corr["dif"])


# In[246]:


test_corr["dif"].sort_values(ascending=False)
#estas se me ocurre que serian las palabras que mayor diferencia podrian hacer


# In[247]:


test_corr


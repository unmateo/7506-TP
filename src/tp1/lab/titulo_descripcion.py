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
# 
# ## Hipótesis
# 
# - ciertas palabras indican mayor precio (luminoso, jardín, hermoso, vista...)
# - a más palabras, mayor precio

# In[3]:


#importo las funciones para levantar los dataframes
get_ipython().run_line_magic('run', '"../../utils/dataset_parsing.ipynb"')
#importo las funciones para graficar
get_ipython().run_line_magic('run', '"../../utils/graphs.ipynb"')
df = levantar_datos("../../"+DATASET_RELATIVE_PATH)
df.columns


# In[4]:


import nltk  
from nltk.corpus import stopwords  
from string import punctuation  


# In[5]:


spanish_stopwords = set(stopwords.words('spanish'))
non_words = set(punctuation)
non_words.update({'¿', '¡'})
non_words.update(map(str,range(10)))


# In[79]:


def add_to_dict(word, dictionary):
    dictionary[word] = dictionary.get(word, 0) + 1 

def get_word_count(series):
    """
    
    """
    words_count = {}
    no_words_count = {}
    for title in series.values:
        if not isinstance(title,str): continue
        words = title.split()
        for word in words:
            text = ''.join([" " if c in non_words else c for c in word]).strip()
            if text in spanish_stopwords: 
                add_to_dict(text, no_words_count)
                continue
            add_to_dict(text, words_count)
    return words_count, no_words_count


# In[116]:


from collections import Counter

def get_word_counters(series):
    """
        Faltaría analizar stemming
    """
    words_counter = Counter()
    no_words_counter = Counter()
    for title in series.values:
        if not isinstance(title,str): continue
        words = filter(lambda x: len(x)>2, title.split())
        for word in words:
            text = ''.join([" " if c in non_words else c for c in word]).strip()
            text_ok = len(text) > 2 and not text in spanish_stopwords
            counter_to_update = words_counter if text_ok else no_words_counter 
            counter_to_update.update({text})
    return words_counter, no_words_counter


# In[117]:


titulo_palabras, titulo_no_palabras = get_word_counters(df["titulo"])


# In[119]:


descripcion_palabras, descripcion_no_palabras = get_word_counters(df["descripcion"])


# In[120]:


print(len(titulo_palabras), len(titulo_no_palabras), len(descripcion_palabras), len(descripcion_no_palabras))


# In[131]:


titulo_palabras.most_common(100)


# In[130]:


descripcion_palabras.most_common(100)


# In[125]:


import matplotlib.pyplot as plt


# In[128]:


titulo_palabras_top_200 = titulo_palabras.most_common(200)


# In[ ]:





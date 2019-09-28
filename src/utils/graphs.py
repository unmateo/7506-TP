#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mtl
import seaborn as sns
import numpy as np
from pywaffle import Waffle
from wordcloud import WordCloud
plt.rc("font",family="monospace")


# In[2]:


def get_heatmap(data, title="Titulo", xlabel="Mes", ylabel="Año" , show_values=False, **kwargs):
    fig, ax = plt.subplots(figsize=(20,15))
    defaults = {"cmap": "YlGn", "square": True, "cbar_kws": {"shrink":0.3}, "ax": ax, "annot": show_values}
    for k,v in defaults.items(): kwargs.setdefault(k, v)
    heatmap = sns.heatmap(data, **kwargs)
    heatmap.set_title(title, fontdict={"fontsize": 18})
    heatmap.set_ylabel(ylabel)
    heatmap.set_xlabel(xlabel)
    return heatmap


# In[ ]:


def get_barplot(series, h_align=False, title="", x_label="", y_label="", show_grid=False, size=(12,6)):
    max_value = series.max()
    if h_align:
        kind = "barh"
        x_lim = (0, max_value * 1.08)
        y_lim = None
    else:
        kind = "bar"
        y_lim = (0, max_value * 1.08)
        x_lim = None
    plot = series.plot(kind=kind, figsize=size, fontsize=12, cmap="summer_r", grid=show_grid,xlim=x_lim, ylim=y_lim )
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plot.set_title(title, fontsize=16)
    margin = max_value * 0.03
    for index, value in enumerate(series):
        x_pos = value+margin if h_align else index
        y_pos = index if h_align else value+margin
        plot.text(x_pos, y_pos, str(value), horizontalalignment='center', verticalalignment='center')
    return plot


# In[4]:


def get_boxplot(data_toshow, value_x, value_y, size, title="", label_x="", label_y=""):
    plt.subplots(figsize=size)
    plot = sns.boxplot(x=value_x, y=value_y, data=data_toshow,
                     palette="hls")
    plot.set_title(title, fontsize=18)
    plot.set_xlabel(label_x, fontsize=18)
    plot.set_ylabel(label_y, fontsize=18)
    return plot


# In[3]:


def get_hist(serie, title="", xlabel="", ylabel="", bins=50, size=(12, 6)):
    plt.subplots(figsize=size)
    plot = serie.plot.hist(bins=bins, color='lightblue')
    plot.set_title(title, fontsize=18)
    plot.set_xlabel(xlabel,fontsize=18)
    plot.set_ylabel(ylabel, fontsize=18)
    return plot


# In[1]:


def get_waffleplot(series, title=" ", precision=10, boolean=False):
    """
        Espera una serie de valores normalizados [0,1].
        
    """
    
    if len(series)>10: raise Exception("La serie no puede tener más de 10 elementos")
    if precision not in {10,20}: raise Exception("Los valores admitidos para precisión son {10,20}")
    cmap = plt.cm.Set3_r
    max_char = 25
    series.index = series.index.tolist()
    if not boolean:
        otros = 0.9999 - series.sum()
        if otros > 0: series["Otros valores"] = otros
        series.index = pd.Index(series.index.map(lambda x: x[:25].ljust(max_char)))
    colors = [mtl.colors.rgb2hex(cmap(i)) for i in range(len(series))]
    y_legend = (len(series)-1)/35+0.55
    plot = plt.figure(
        FigureClass=Waffle, 
        rows=precision,
        columns=precision,
        values=series*100,
        figsize= (12,6),
        title={"label":title, "horizontalalignment":"center", "fontsize":18, "position": (1,1), "pad": 20},
        labels = ["{} ({}%)".format(n, str(v*100)[:4]) for n, v in series.items()],
        legend = {'bbox_to_anchor': (2.2, y_legend), "fontsize":14},
        colors = colors
    )
    plot.set_tight_layout(False)
    return plot


# In[4]:


def get_wordcloud(frecuencias):
    """
        Recibe un Counter (o diccionario de frecuencias) y devuelve un wordcloud de esos términos.
    """
    cantidad = len(frecuencias)
    if 0 < cantidad < 51:
        width, height = 400, 200
    elif 50 < cantidad < 101:
        width, height = 600, 300
    else:
        width, height = 800, 400
    wc = WordCloud(max_font_size=60, max_words=80, background_color="white", width=width, height=height).generate_from_frequencies(frecuencias)
    plot = plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    return wc


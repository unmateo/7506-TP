#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mtl
import seaborn as sns
import numpy as np
# from pywaffle import Waffle
plt.rc("font",family="monospace")


# In[2]:


def get_heatmap(data, title="Titulo", xlabel="Mes", ylabel="Año" ,show_values=False):
    fig, ax = plt.subplots(figsize=(20,15))
    heatmap = sns.heatmap(data, cmap="summer_r", square=True, cbar_kws={"shrink":0.3}, ax=ax, annot=show_values)
    heatmap.set_title(title)
    heatmap.set_ylabel(ylabel)
    heatmap.set_xlabel(xlabel)
    return heatmap


# In[3]:


def get_barplot(series, h_align=False, title="", x_label="", y_label="", show_grid=False):
    max_value = series.max()
    if h_align:
        kind = "barh"
        x_lim = (0, max_value * 1.08)
        y_lim = None
    else:
        kind = "bar"
        y_lim = (0, max_value * 1.08)
        x_lim = None
    plot = series.plot(kind=kind, figsize=(12,6), fontsize=12, cmap="Greens_r", grid=show_grid,xlim=x_lim, ylim=y_lim )
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


def bar_plot(dataframe, x="", y="", rot="0"):
    return dataframe.plot.bar(x=x, y=y, rot=0)


# In[ ]:


# def get_waffleplot(series, title=" ", precision=10, boolean=False):
#     """Espera una serie de valores normalizados [0,1]."""
#     if len(series)>10: raise Exception("La serie no puede tener más de 10 elementos")
#     if precision not in {10,20}: raise Exception("Los valores admitidos para precisión son {10,20}")
#     cmap = plt.cm.Set3_r
#     max_char = 15
#     series.index = series.index.tolist()
#     if not boolean:
#         otros = 0.9999 - series.sum()
#         if otros > 0: series["Otros valores"] = otros
#         series.index = pd.Index(series.index.map(lambda x: x[:13].ljust(max_char)))
#     colors = [mtl.colors.rgb2hex(cmap(i)) for i in range(len(series))]
#     y_legend = (len(series)-1)/35+0.55
#     plot = plt.figure(
#         FigureClass=Waffle, 
#         rows=precision,
#         columns=precision,
#         values=series*100,
#         figsize= (12,6),
#         title={"label":title, "horizontalalignment":"center", "fontsize":18, "position": (1,1), "pad": 20},
#         labels = ["{} ({}%)".format(n, str(v*100)[:4]) for n, v in series.items()],
#         legend = {'bbox_to_anchor': (1.9, y_legend), "fontsize":14},
#         colors = colors
#     )
#     plot.set_tight_layout(False)
#     return plot


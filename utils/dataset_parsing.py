#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

FULL_DATASET = "../enunciado/data/train.csv"


# In[2]:


def levantar_datos(csv_file: str = FULL_DATASET) -> pd.DataFrame:
    """
        Recibe un .csv que debe tener el formato de columnas específico del tp.
        Devuelve un Dataframe optimizado con esa información.
    """
    df = pd.read_csv(csv_file)
    return df


# In[7]:


df.columns


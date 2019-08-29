#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

DATASET_RELATIVE_PATH = "enunciado/data/train.csv"


# In[10]:


def levantar_datos(csv_file: str = "../"+DATASET_RELATIVE_PATH, optimizar: bool = True) -> pd.DataFrame:
    """
        Recibe un .csv que debe tener el formato de columnas específico del tp.
        Si optimizar = True, asigna mejores tipos y agrega columnas útiles.
        Devuelve un Dataframe con esa información.
    """
    if not optimizar:
        return pd.read_csv(csv_file)
    dtypes = {
        "piscina": "bool",
        "escuelascercanas": "bool",
        "centroscomercialescercanos": "bool",
        "usosmultiples": "bool",
        "gimnasio": "bool",
        "garages": "float16",
        "banos": "float16",
        "tipodepropiedad": "category",
        "ciudad": "category",
        "provincia": "category",
        "antiguedad": "float16",
        "habitaciones": "float16",
        "metroscubiertos": "float32",
        "metrostotales": "float32",
    }
    df = pd.read_csv(csv_file, dtype=dtypes, parse_dates=["fecha"])
    df["mes"] = df["fecha"].dt.month
    df["ano"] = df["fecha"].dt.year
    df["dia"] = df["fecha"].dt.day
    return df


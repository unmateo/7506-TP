#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from shapely.geometry import Point

DATASET_RELATIVE_PATH = "enunciado/data/train.csv"


# In[ ]:


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
    agregar_columnas_tiempo(df)
    agregar_columnas_precio(df)
    agregar_columnas_gps(df)
    return df


# In[ ]:


def agregar_columnas_tiempo(df):
    df["mes"] = df["fecha"].dt.month
    df["ano"] = df["fecha"].dt.year
    df["dia"] = df["fecha"].dt.day


# In[ ]:


def agregar_columnas_precio(df):
    df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
    df["precio_metro_total"] = df["precio"] / df["metrostotales"]


# In[ ]:


def agregar_columnas_gps(df):
    df.loc[:,"gps"] = df.loc[~ df["lng"].isna()].apply(lambda x: Point(x["lng"],x["lat"]), axis=1)


# In[ ]:


def esta_en_mexico(point: Point) -> bool:
    """ 
        Recibe un punto (lat,lng) y devuelve (muy aproximadamente) True si está dentro de Mexico, False si no.
        Hacerlo con ```df["coord"].map(lambda x: mexico_polygon.contains(x))``` sería muy lento.
    """
    MEX_MIN_LNG, MEX_MAX_LNG = (-120, -85)
    MEX_MIN_LAT, MEX_MAX_LAT = (14,33)
    return (MEX_MIN_LNG < point.x < MEX_MAX_LNG) and (MEX_MIN_LAT < point.y < MEX_MAX_LAT)


# In[ ]:





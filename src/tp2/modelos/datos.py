import numpy
import pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

from procesamiento_palabras import (
    limpiar_campo,
    cantidad_palabras,
    cantidad_palabras_positivas,
    cantidad_palabras_negativas
)

TRAIN_CSV = "../../datos/train.csv"
TEST_CSV = "../../datos/test.csv"
DOLAR_CSV = "../../datos/dolar.csv"


FEATURES_DISPONIBLES = {
    "piscina", "usosmultiples", "gimnasio", "garages",
    "escuelascercanas", "centroscomercialescercanos",
    "banos", "habitaciones", "metroscubiertos", "metrostotales",
    "antiguedad", "tipodepropiedad",
    "idzona", "ciudad", "provincia", "gps", "lng", "lat",
    "fecha", "ano", "mes", "dia",
    "precio", "precio_metro_cubierto", "precio_metro_total",
    "titulo", "descripcion",
    "cantidad_palabras_titulo", "cantidad_palabras_descripcion",
    "palabras_positivas_descripcion", "palabras_negativas_descripcion",
    "dolar"
}

def levantar_datos(train_file=TRAIN_CSV, test_file=TEST_CSV, features=None, seed=42):
    """
        Levanta los datos  de la competencia.
        Limpia y agrega columnas basicas.
        Separa en train y test.
        Devuelve una tupla:
            train, test, submit
    """
    if not features:
        features = FEATURES_DISPONIBLES
    train, test = train_test_split(read_csv(train_file, features), random_state=seed)
    submit = read_csv(test_file, features-{"precio"})
    return train, test, submit

def read_csv(csv_file, features) -> pd.DataFrame:
    """
        Recibe un .csv que debe tener el formato de columnas específico del tp.
        Si optimizar = True, asigna mejores tipos y agrega columnas útiles.
        Devuelve un Dataframe con esa información.
    """

    types = {
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
        "metroscubiertos": "float16",
        "metrostotales": "float16",
        "idzona": "category",
        "precio": "float32",
        "titulo": "object",
        "descripcion": "object",
        "lng": "float16",
        "lat": "float16"
    }
    columns = [col for col in types.keys() if col in features] + ["fecha", "id"]
    dtypes ={col:dtype for col,dtype in types.items() if col in features}
    df = pd.read_csv(csv_file,
                     usecols=columns,
                     dtype=dtypes,
                     parse_dates=["fecha"],
                     index_col='id')

    if "ano" in features:
        df["ano"] = df["fecha"].dt.year.astype('int16')
    if "mes" in features:
        df["mes"] = df["fecha"].dt.month.astype('int8')
    if "dia" in features:
        df["dia"] = df["fecha"].dt.day.astype('int8')

    if "precio" in features:
        if "metroscubiertos" in features and "precio_metro_cubierto" in features:
            df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
        if "metrostotales" in features and "precio_metro_total" in features:
            df["precio_metro_total"] = df["precio"] / df["metrostotales"]
    
    if {"gps","lng","lat"}.issubset(features):
        agregar_columnas_gps(df)

    if {"titulo", "cantidad_palabras_titulo"}.issubset(features):
        
        df["titulo"] = df['titulo'].map(limpiar_campo)
        df["cantidad_palabras_titulo"] = cantidad_palabras(df["titulo"])
    
    if "descripcion" in features:
        
        df['descripcion'] = df['descripcion'].map(limpiar_campo)
        
        if "cantidad_palabras_descripcion" in features:
            df["cantidad_palabras_descripcion"] = cantidad_palabras(df['descripcion']).astype('int16')
        
        if "palabras_positivas_descripcion" in features:
            df["palabras_positivas_descripcion"] = cantidad_palabras_positivas(df['descripcion']).astype('int16')
        
        if "palabras_negativas_descripcion" in features:
            df["palabras_negativas_descripcion"] = cantidad_palabras_negativas(df['descripcion']).astype('int16')
    
    if {"ano", "mes", "dolar"}.issubset(features):
        df = agregar_dolar(df)

    return df


def agregar_columnas_gps(df):
    """
        Creo los puntos para todas las filas que tengan latitud/longitud.
        Anulo los puntos que estén fuera de México (pongo NaN).
    """
    filter_mexico = lambda x: x if x is not numpy.NaN and not x.is_empty and esta_en_mexico(x) else numpy.NaN
    df.loc[:,"gps"] = df.loc[~ df["lng"].isna()].apply(lambda x: Point(x["lng"],x["lat"]), axis=1)
    df.loc[:,"gps"] = df.loc[~ df["gps"].isna()]["gps"].map(filter_mexico)

def esta_en_mexico(point: Point) -> bool:
    """ 
        Recibe un punto (lat,lng) y devuelve (muy aproximadamente) True si está dentro de Mexico, False si no.
        Hacerlo con ```df["coord"].map(lambda x: mexico_polygon.contains(x))``` sería muy lento.
    """
    if point.is_empty: return False
    MEX_MIN_LNG, MEX_MAX_LNG = (-120, -85)
    MEX_MIN_LAT, MEX_MAX_LAT = (14,33)
    return (MEX_MIN_LNG < point.x < MEX_MAX_LNG) and (MEX_MIN_LAT < point.y < MEX_MAX_LAT)

def agregar_dolar(df, index='id'):
    cotizaciones = pd.read_csv(DOLAR_CSV, dtype={'cotizacion':'float16'}, parse_dates=["fecha"]).rename({"cotizacion": "dolar"}, axis= 'columns')
    cotizaciones['mes'] = cotizaciones['fecha'].dt.month
    cotizaciones['ano'] = cotizaciones['fecha'].dt.year
    cotizaciones_por_mes = cotizaciones.groupby(['ano', 'mes']).agg({'dolar':'mean'})
    cotizaciones_por_mes = cotizaciones_por_mes.reset_index()
    cotizaciones_por_mes['dolar'] = cotizaciones_por_mes['dolar'].astype('float16')
    return df.reset_index().merge(cotizaciones_por_mes, on=["ano","mes"], how="left").set_index(index)
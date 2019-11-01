import numpy
import pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

TRAIN_CSV = "../../datos/train.csv"
TEST_CSV = "../../datos/test.csv"

def levantar_datos(train_file=TRAIN_CSV, test_file=TEST_CSV):
    """
        Levanta los datos  de la competencia.
        Limpia y agrega columnas basicas.
        Separa en train y test.
        Devuelve una tupla:
            train, test, submit
    """
    train, test = train_test_split(read_csv(train_file))
    submit = read_csv(test_file)
    return train, test, submit

def read_csv(csv_file, optimizar: bool = True) -> pd.DataFrame:
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

def agregar_columnas_tiempo(df):
    df["mes"] = df["fecha"].dt.month
    df["ano"] = df["fecha"].dt.year
    df["dia"] = df["fecha"].dt.day

def agregar_columnas_precio(df):
    if "precio" in df.columns:
        df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
        df["precio_metro_total"] = df["precio"] / df["metrostotales"]

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


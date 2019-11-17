import numpy
import pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

TRAIN_CSV = "../../datos/train.csv"
TEST_CSV = "../../datos/test.csv"


FEATURES_DISPONIBLES = {
    "piscina", "usosmultiples", "gimnasio", "garages",
    "escuelascercanas", "centroscomercialescercanos",
    "banos", "habitaciones", "metroscubiertos", "metrostotales",
    "antiguedad", "tipodepropiedad",
    "idzona", "ciudad", "provincia", "gps", "lng", "lat"
    "fecha", "ano", "mes", "dia",
    "precio", "precio_metro_cubierto", "precio_metro_total"
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
        "metroscubiertos": "float32",
        "metrostotales": "float32",
        "idzona": "category",
        "precio": "float32"
    }
    columns = [col for col in types.keys() if col in features] + ["fecha", "id"]
    dtypes ={col:dtype for col,dtype in types.items() if col in features}
    df = pd.read_csv(csv_file,
                     usecols=columns,
                     dtype=dtypes,
                     parse_dates=["fecha"],
                     index_col='id')

    if "ano" in features:
        df["ano"] = df["fecha"].dt.year
    if "mes" in features:
        df["mes"] = df["fecha"].dt.month
    if "dia" in features:
        df["dia"] = df["fecha"].dt.day

    if "precio" in features:
        if "metroscubiertos" in features and "precio_metro_cubierto" in features:
            df["precio_metro_cubierto"] = df["precio"] / df["metroscubiertos"]
        if "metrostotales" in features and "precio_metro_total" in features:
            df["precio_metro_total"] = df["precio"] / df["metrostotales"]
    
    if {"gps","lng","lat"}.issubset(features):
        agregar_columnas_gps(df)

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


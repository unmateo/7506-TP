import pandas as pd

TRAIN_CSV = "../datos/train.csv"
TEST_CSV = "../datos/test.csv"

def levantar_datos(train=TRAIN_CSV, test=TEST_CSV):
    """
        Levanta los datos  de la competencia, hace una limpieza 
        común a todos los modelos  y devuelve los datos en un
        Dataframe de pandas.
    """
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    return train, test
import functools
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from datos import levantar_datos
from kaggle import api
import pandas as pd


class Modelo:

    LOG_RESULTADOS = "../../resultados.csv"
    LEN_DESCRIPCION = 80
    DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
    DIR_PREDICCIONES = "../../predicciones/"
    COMPETENCIA = "Inmuebles24"

    
    def cronometrar(nombre=None):
        """ 
            Mide y registra el tiempo de ejecucion de la funcion que envuelve.
            Si no recibe nombre, usa el de la funcion que envuelve.
        """
        def afuera(function):
            @functools.wraps(function)
            def adentro(self, *args, **kwargs):
                inicio = time.perf_counter()
                resultado = function(self, *args, **kwargs)
                fin = time.perf_counter()
                tiempo = round(fin - inicio,2)
                clave = nombre if nombre else function.__name__
                print("{} demoro {} segundos".format(clave, tiempo))
                self.tiempos[clave] = self.tiempos.get(clave, 0) + tiempo
                return resultado
            return adentro
        return afuera   
    
    @cronometrar("instanciar")
    def __init__(self):
        """ """
        self.cargado = False
        self.entrenado = False
        self.validado = False
        self.presentado = False
        self.modelo = self.__class__.__name__
        self.resultado_validacion = None
        self.resultado_kaggle = None
        self.tiempos = {}

    @cronometrar("cargar")
    def cargar_datos(self, features=None):
        """ Carga los datos que se usarán para entrenar y predecir. """
        train, test, submit = levantar_datos(features=features)
        self.train_data = train
        self.test_data = test
        self.submit_data = submit
        self.cargado = True
        return True

    @cronometrar()
    def entrenar(self):
        """ Entrena el modelo. Sobreescribir si es necesario """
        if not self.cargado:
            raise Exception("No se han cargado los datos.")
        self.entrenado = True
        return True
    
    @cronometrar()
    def predecir(self, data):
        """ Este método debe sobreescribirse.
            Debe recibir un DataFrame con el formato del
            TP, agregarle una columna 'target' y devolverlo.
        """
        raise NotImplementedError()

    @cronometrar()
    def validar(self):
        """ Valida el modelo localmente."""
        if not self.entrenado:
            raise Exception("No se ha entrenado.")
        self.validado = True
        predicciones = self.predecir(self.test_data)
        score = self.puntuar(predicciones["precio"], predicciones["target"])
        self.resultado_validacion = score
        return score
    
    @cronometrar()
    def puntuar(self, real, prediccion):
        """
            Recibe un array con valores reales y otro con predicciones.
            Devuelve un puntaje con la metrica de la competencia.
        """
        return mean_absolute_error(real, prediccion)

    @cronometrar()
    def guardar(self, predicciones):
        """
            Recibe una DataFrame con predicciones y
            lo guarda en un archivo csv con el formato
            requerido por la competencia.
        """
        columnas = ["target"]
        timestamp = datetime.now().strftime(self.DATETIME_FORMAT)
        nombre_archivo = "{}_{}.csv".format(self.modelo, timestamp)
        self.target = self.DIR_PREDICCIONES + nombre_archivo
        predicciones.to_csv(self.target, columns=columnas, index=True)
        return self.target
    
    @cronometrar()
    def submit(self, descripcion):
        """
            Hace el submit en la competencia.
            Busca el puntaje.
            Eleva la excepcion si algo falla.
            Devuelve el puntaje.
        """
        response = api.competition_submit(self.target, descripcion, self.COMPETENCIA)
        return self.buscar_score(descripcion)
    
    def buscar_score(self, descripcion):
        """ Busca el score de un submit por su descripcion. """
        submits = api.competitions_submissions_list(self.COMPETENCIA, page=1)
        candidatos = [ a for a in submits if a.get("description")==descripcion ]
        if not candidatos:
            msg = "No se encontro el score: {}".format(descripcion)
            raise Exception(msg)
        submit = candidatos[0]
        if submit.get("status") == "error":
            raise Exception(submit.get("errorDescription"))
        return submit.get("publicScore")
    
    @cronometrar()
    def presentar(self, predicciones, descripcion=""):
        """ 
            Hace el submit en la competencia.
            Registra los resultados.
        """
        if not self.validado:
            raise Exception("No se ha validado.")
        if len(descripcion) < 10:
            descripcion = "Resultado local: {}".format(self.resultado_validacion)
            print(descripcion)
        filename = self.guardar(predicciones)
        self.resultado_kaggle = self.submit(descripcion)
        self.registrar_resultado(descripcion)
        return self.resultado_kaggle

    def registrar_resultado(self, descripcion):
        """ Registra el resultado en el log de resultados.
            El formato a utilizar es:
                (timestamp, modelo, validacion, kaggle, comentario)
        """
        registro = (
                datetime.now().strftime(self.DATETIME_FORMAT),
                self.modelo,
                str(self.resultado_validacion),
                str(self.resultado_kaggle),
                descripcion
            )
        with open(self.LOG_RESULTADOS, "a") as log:
            log.write(",".join(registro) + "\n")
        return True


    def one_hot_encode(self, df, categoricas):
        """
            Recibe un DataFrame y una lista de sus columnas categóricas.
            Aplica one hot encoding a cada una, agregandolas al DataFrame
            y eliminando la columna categórica.
            
            Devuelve el DataFrame con estas modificaciones.
        """
        
        return pd.get_dummies(df, prefix=categoricas, columns=categoricas, dtype='bool')
        

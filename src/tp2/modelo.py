from datetime import datetime
from sklearn.metrics import mean_absolute_error
from datos import levantar_datos
from kaggle import api

class Modelo:

    LOG_RESULTADOS = "../resultados.csv"
    LEN_DESCRIPCION = 80
    DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
    DIR_PREDICCIONES = "predicciones/"
    COMPETENCIA = "Inmuebles24"

    def __init__(self, descripcion):
        """
        
        Arguments:
            descripcion {str}:
                Describe en qué se basa el modelo.
                Debe tener al menos 80 caracteres.
        """
        if not isinstance(descripcion, str) or \
                    len(descripcion) < self.LEN_DESCRIPCION:
            raise Exception("La descripcion debe "\
                "ser un string de al menos 80 caracteres")

        self.descripcion = descripcion
        self.cargado = self.cargar_datos()
        self.entrenado = False
        self.validado = False
        self.presentado = False
        self.modelo = self.__class__.__name__

    def cargar_datos(self):
        """ Carga los datos que se usarán para entrenar y predecir. """
        train, test, submit = levantar_datos()
        self.train_data = train
        self.test_data = test
        self.submit_data = submit
        return True

    def entrenar(self):
        """ Entrena el modelo. Sobreescribir si es necesario """
        if not self.cargado:
            raise Exception("No se han cargado los datos.")
        self.entrenado = True
        return True
    
    def predecir(self, data):
        """ Este método debe sobreescribirse.
            Debe recibir un DataFrame con el formato del
            TP, agregarle una columna 'target' y devolverlo.
        """
        raise NotImplementedError()

    def validar(self):
        """ Valida el modelo localmente."""
        if not self.entrenado:
            raise Exception("No se ha entrenado.")
        self.validado = True
        predicciones = self.predecir(self.test_data)
        score = self.puntuar(predicciones["precio"], predicciones["target"])
        self.resultado_validacion = score
        return score
    
    def puntuar(self, real, prediccion):
        """
            Recibe un array con valores reales y otro con predicciones.
            Devuelve un puntaje con la metrica de la competencia.
        """
        return mean_absolute_error(real, prediccion)

    def guardar_prediccion(self, predicciones):
        """
            Recibe una DataFrame con predicciones y
            lo guarda en un archivo csv con el formato
            requerido por la competencia.
        """
        columnas = ["id", "target"]
        timestamp = datetime.now().strftime(self.DATETIME_FORMAT)
        nombre_archivo = "{}_{}.csv".format(self.modelo, timestamp)
        self.target = self.DIR_PREDICCIONES + nombre_archivo
        predicciones.to_csv(self.target, columns=columnas, index=False)
        return self.target
    
    def submit_target(self, descripcion):
        """
            Hace el submit en la competencia.
            Busca el puntaje.
            Eleva la excepcion si algo falla.
            Devuelve el puntaje.
        """
        response = api.competition_submit(self.target, descripcion, self.COMPETENCIA)
        return self.buscar_score(descripcion)
    
    def buscar_score(self, descripcion):
        """
            Busca el score de un submit por su descripcion.
        """
        submits = api.competitions_submissions_list(self.COMPETENCIA, page=1)
        candidatos = [ a for a in submits if a.get("description")==descripcion ]
        if not candidatos:
            msg = "No se encontro el score: {}".format(descripcion)
            raise Exception(msg)
        submit = candidatos[0]
        if submit.get("status") == "error":
            raise Exception(submit.get("errorDescription"))
        return submit.get("publicScore")
        
    def presentar(self, descripcion=""):
        """ 
            Predice el set de test.
            Hace el submit en la competencia.
            Registra los resultados.
        """
        if not self.validado:
            raise Exception("No se ha validado.")
        if len(descripcion) < 10:
            descripcion = "Resultado local: {}".format(self.resultado_validacion)
            print(descripcion)
        predicciones = self.predecir(self.submit_data)
        filename = self.guardar_prediccion(predicciones)
        self.resultado_kaggle = self.submit_target(descripcion)
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
            log.write(",".join(registro))
        return True
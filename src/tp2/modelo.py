from datos import levantar_datos
from datetime import datetime

class Modelo:

    LOG_RESULTADOS = "../resultados.csv"
    LEN_DESCRIPCION = 80
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

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

    def cargar_datos(self):
        """ Carga los datos que se usarán para entrenar y predecir. """
        train, test = levantar_datos()
        self.train = train
        self.test = test
        return True

    def entrenar(self):
        """ Entrena el modelo. Sobreescribir si es necesario """
        if not self.cargado:
            raise Exception("No se han cargado los datos.")
        self.entrenado = True
        return True
    
    def predecir(self):
        """ Este método debe sobreescribirse.
            Debe devolver un df con dos columnas: id y precio.
            Cada fila debe corresponder a una fila de self.test
        """
        raise NotImplementedError()

    def validar(self):
        """ Valida el modelo localmente."""
        if not self.entrenado:
            raise Exception("No se ha entrenado.")
        self.validado = True
        self.resultado_validacion = 100
        return True

    def presentar(self, comentario=""):
        """ Hace el submit en la competencia."""
        if not self.validado:
            raise Exception("No se ha validado.")
        self.resultado_kaggle = 200
        #self.registrar_resultado(comentario)
        return True

    def registrar_resultado(self, comentario=""):
        """ Registra el resultado en el log de resultados.
        El formato a utilizar es:
            (timestamp, modelo, validacion, kaggle, comentario)
        """
        registro = (
                datetime.now().strftime(self.DATETIME_FORMAT),
                self.__class__.__name__,
                str(self.resultado_validacion),
                str(self.resultado_kaggle),
                comentario
            )
        with open(self.LOG_RESULTADOS, "a") as log:
            log.write(",".join(registro))
        return True
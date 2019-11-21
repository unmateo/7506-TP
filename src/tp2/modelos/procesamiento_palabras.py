"""
    En base a lo analizado en el TP1, agregamos features
    a partir de los campos que incluyen palabras.
"""
from functools import reduce
from nltk.corpus import stopwords
from string import punctuation
import re
from unidecode import unidecode
from collections import Counter


# creo el set data stopwords
spanish_stopwords = set(stopwords.words('spanish'))
non_words = set(punctuation)
non_words.update({'¿', '¡'})
non_words.update(map(str,range(10)))

# agrego palabras comunes que detecte y no son relevantes
stopwords_extra = {"fracc", "cada", "mas", "ntilde", "consta", "tres", "dos", "solo", "cuenta", "areas", "tipo", "nbsp", "oacute", "hrs", "aacute", "palapa", "easybroker", "tarja", "cuatro", "uacute", "cancel", "asi", "hace", "tan", "dia", "ningun" }

palabras_positivas = {"vigilancia","hermosa","diseño","vistas","playa","conservacion","tenis","balcon","panoramica","exclusivos","golf","canchas","remodelada","acondicionado","lujo","jacuzzi","diseno","exclusiva","magnifica","exclusivo","country","precioso","estilo","seguridad","verdes","juegos","servicio","excelente","terraza","jardin","hermosa","vista","bonita","renta", "granito","porcelanato","mejores"}
palabras_negativas = {"oportunidad","remato","oferta","remodelar", "inversion"}


# FUNCIONES FEATURE

def cantidad_palabras(serie_de_palabras):
    """
        Devuelve una serie con la cantidad de palabras
        totales que tiene en cada fila, sacando stopwords.
    """
    return serie_de_palabras.map(lambda x: len(x.split()))

def cantidad_palabras_positivas(serie_de_palabras):
    """
        Devuelve una serie con la cantidad de palabras
        positivas que se encontraron en cada fila
    """
    return serie_de_palabras.map(lambda x: _contar_palabras(x, palabras_positivas))

def cantidad_palabras_negativas(serie_de_palabras):
    """
        Devuelve una serie con la cantidad de palabras
        positivas que se encontraron en cada fila
    """
    return serie_de_palabras.map(lambda x: _contar_palabras(x, palabras_negativas))
                          

# FUNCIONES AUXILIARES

def is_meaningful(word: str) -> bool:
    """
        Recibe una palabra, remueve puntuaciones y verifica que lo que queda no esté en el set de stopwords
    """
    return len(word) > 2 and not word in spanish_stopwords

def remove_html(field: str) -> str:
    """
        Recibe un texto y devuelve una copia sin los tags html
    """
    return re.compile(r'<[^>]+>').sub('', field) if field else field

def normalize(field: str) -> str:
    """
        Recibe un texto y devuelve una copia sin acentos, ñ ni puntuaciones.
    """
    return ''.join([" " if c in non_words else unidecode(c) for c in field]).strip() if field else ""

def limpiar_campo(field: str) -> str:
    """
        Recibe un campo string que podría tener muchas palabras.
        Devuelve un string que contiene sólo las palabras significativas.
    """
    if not isinstance(field,str): return ""
    without_html = remove_html(field)
    normalized = normalize(without_html)
    meaningful = " ".join(set(filter(is_meaningful, normalized.split())))
    return meaningful

def _contar_palabras(campo, set_palabras):
    """ Cuenta las palabras que hay en el string campo que pertenezcan a set_palabras. """
    return reduce(lambda x,y: x+int(y in set_palabras) , campo.split(), 0)

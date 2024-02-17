import  os
from enum import Enum, StrEnum, auto

# Rutas globales
DATA_DIR = r"..\Dataset"
"""Directorio base donde se encuentran los DATASETS"""

INITIAL_DIR = r"..\Dataset\Inicial"
"""Directorio usado en el ayuntamiento para procesar los datos iniciales y generar los datasets anonimizados"""
ESC_DIR = os.path.join(INITIAL_DIR,"escrituras")
"""Directorio donde se encuentras los PDF de escrituras"""
FICH_DIR = os.path.join(INITIAL_DIR,"fichas")
"""Directorio con las fichas notariales"""
PDF_FICHAS_DIR = os.path.join(INITIAL_DIR, "pdf_fichas")
"""Ruta donde se combinan los PDF y los XML y se generan los textos y textos normalizados"""

# Ficheros de diccionarios usados en la anonimización
RANDOM_NOMBRES = os.path.join(DATA_DIR,"RANDOM","nombres.jsonl")
"""Diccionario de nombres usados en las fichas"""
RANDOM_APELLIDOS = os.path.join(DATA_DIR,"RANDOM","apellidos.jsonl")
"""Diccionario de apellidos usados en las fichas"""
RANDOM_EMPRESAS = os.path.join(DATA_DIR,"RANDOM","empresas.jsonl")
"""Diccionario de empresas usadas en las fichas"""
RANDOM_VIAS = os.path.join(DATA_DIR,"RANDOM","vias.jsonl")
"""Diccionario de vías usadas en las fichas"""

class Tipo_Contenido(StrEnum):
    """Enumerado para trabajar con los tipos texto para diferenciar las preguntas del contexto"""
    # Tipos de preguntas #
    PROTOCOLO = auto()
    FECHA = auto()
    NOTARIO = auto()
    TIPO_DOCUMENTO = 'tipo'
    # Tipos que no están codificados como preguntas #
    CONTEXT = auto()    
    
    @staticmethod
    def tipo(question:str)->StrEnum|None:
        """Indica de qué tipo es esta pregunta"""
        for tipo_c in Tipo_Contenido:
            if tipo_c in [Tipo_Contenido.PROTOCOLO, Tipo_Contenido.FECHA, Tipo_Contenido.NOTARIO, Tipo_Contenido.TIPO_DOCUMENTO] \
                and tipo_c in question.lower():
                return tipo_c            
        # Si no es un tipo dentro de los esperados para preguntas, es CONTEXT
        return Tipo_Contenido.CONTEXT
    @staticmethod
    def pregunta(tipo:StrEnum):
        """Devuelve la pregunta que debe hacerse para ser de este tipo, si el tipo no está pensado para ser preguntado, devuelve None"""
        if tipo == Tipo_Contenido.PROTOCOLO:
            return "¿cuál es el número de protocolo?"
        elif tipo == Tipo_Contenido.NOTARIO:
            return "¿qué notario ha firmado el documento?"
        elif tipo == Tipo_Contenido.FECHA:
            return "¿en qué fecha se ha firmado el documento?"
        elif tipo == Tipo_Contenido.TIPO_DOCUMENTO:
            return "¿cuál es el tipo de documento?"
        else:
            return None
        
class Tipo_Diccionario(Enum):
    """Enumerado con los nombres de los diccionarios creados a partir de datos de las fichas"""
    NOMBRE  = auto()
    APELLIDO = auto()
    VIA = auto()
    EMPRESA = auto()


# NER

# Diccionarios para el entrenamiento NER
id2label = {
    0: 'O',
    1: 'B-PROTO', 2: 'I-PROTO',
    3: 'B-FDOC',  4: 'I-FDOC',
    5: 'B-NOT',   6: 'I-NOT',
    7: 'B-TDOC',  8: 'I-TDOC'
}

label2id = {
    'O': 0,
    'B-PROTO': 1, 'I-PROTO': 2,
    'B-FDOC' : 3, 'I-FDOC' : 4,
    'B-NOT'  : 5, 'I-NOT'  : 6,
    'B-TDOC' : 7, 'I-TDOC' :8
}

TIPO_ETIQUETA = {
    Tipo_Contenido.PROTOCOLO: 'PROTO', 
    Tipo_Contenido.FECHA: 'FDOC',
    Tipo_Contenido.NOTARIO: 'NOT',
    Tipo_Contenido.TIPO_DOCUMENTO: 'TDOC'
    }


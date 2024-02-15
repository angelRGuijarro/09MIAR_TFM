import  os
from enum import Enum, StrEnum, auto

# Rutas globales
# Directorio base donde se encuentran los DATASETS
DATA_DIR = r"..\Dataset"

# Directorio usado en el ayuntamiento para procesar los datos originales y generar los datasets anonimizados
INITIAL_DIR = r"..\Dataset\Inicial"
ESC_DIR = os.path.join(INITIAL_DIR,"escrituras")
FICH_DIR = os.path.join(INITIAL_DIR,"fichas")
PDF_FICHAS_DIR = os.path.join(INITIAL_DIR, "pdf_fichas")

# Ficheros de diccionarios usados en la anonimización
RANDOM_NOMBRES = os.path.join(DATA_DIR,"RANDOM","nombres.jsonl")
RANDOM_APELLIDOS = os.path.join(DATA_DIR,"RANDOM","apellidos.jsonl")
RANDOM_EMPRESAS = os.path.join(DATA_DIR,"RANDOM","empresas.jsonl")
RANDOM_VIAS = os.path.join(DATA_DIR,"RANDOM","vias.jsonl")

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
        # Si no es un tipo dentro de los esperados para preguntas, es None
        return None
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
    NOMBRE  = auto()
    APELLIDO = auto()
    VIA = auto()
    EMPRESA = auto()

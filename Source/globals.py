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

# Carpetas para modelos y entrenamiento
TRAINING_DIR = os.path.join("..","training")
MODELS_DIR = os.path.join("..","Models")


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

def ner_predicted_labels(predictions, text, tokenizer):
    """Genera una estructura con los textos originales y el índice de inicio para cada etiqueta predicha"""
    tokens = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    offset_mapping = tokens['offset_mapping']
    # Indica con qué palabra se corresponde cada token y guarda el índice de inicio
    word_ids = tokens.word_ids()
    
    words = {}
    start_indices = {}  # Nuevo diccionario para guardar los índices de inicio
    for idx, (o_map, w_id) in enumerate(zip(offset_mapping, word_ids)):
        if w_id is not None:
            if w_id not in words:
                words[w_id] = ""
                start_indices[w_id] = offset_mapping[idx][0]  # Guarda el índice de inicio de la palabra
            words[w_id] += text[o_map[0]:o_map[1]]
    
    start_indices.pop(None, None)
    words_list = list(words.values())
    
    labels = {}
    for p in predictions:
        word_idx = word_ids[p['index']]
        labels[word_idx] = (p['entity'], words_list[word_idx], start_indices[word_idx])
    
    return [{'id_word': idx, 'label': l[0], 'word': l[1], 'start': l[2]} for (idx, l) in labels.items()]

def group_by_labels(resultado_procesado):   
    """Dado el resultado con los textos originales de las etiquetas predichas, los agrupa por etiqueta en una estructura más compacta"""
    # Asegurarse de que resultado_procesado esté ordenado por 'start'
    resultado_procesado.sort(key=lambda x: x['start'])

    agrupaciones = {}
    secuencia_actual = []
    etiqueta_actual = None
    inicio_actual = None

    for item in resultado_procesado:
        label, word, start = item['label'], item['word'], item['start']

        # Caso 1: Inicio de una nueva secuencia de etiqueta
        if label.startswith("B-"):
            # Guardar la secuencia anterior si existe
            if secuencia_actual:
                etiqueta_limpia = etiqueta_actual[2:]  # Quitar "B-" o "I-"
                if etiqueta_limpia not in agrupaciones:
                    agrupaciones[etiqueta_limpia] = {'label': etiqueta_limpia, 'matches': []}
                agrupaciones[etiqueta_limpia]['matches'].append({'start': inicio_actual, 'text': " ".join(secuencia_actual)})

            # Reiniciar para la nueva secuencia
            secuencia_actual = [word]
            etiqueta_actual = label
            inicio_actual = start

        # Caso 2: Continuación de una secuencia existente
        elif label.startswith("I-") and etiqueta_actual and etiqueta_actual[2:] == label[2:]:
            secuencia_actual.append(word)

    # Guardar la última secuencia si existe
    if secuencia_actual:
        etiqueta_limpia = etiqueta_actual[2:]  # Quitar "B-" o "I-"
        if etiqueta_limpia not in agrupaciones:
            agrupaciones[etiqueta_limpia] = {'label': etiqueta_limpia, 'matches': []}
        agrupaciones[etiqueta_limpia]['matches'].append({'start': inicio_actual, 'text': " ".join(secuencia_actual)})

    return list(agrupaciones.values())
import  os

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

# Carpetas para modelos y entrenamiento
TRAINING_DIR = os.path.join("..","training")
MODELS_DIR = os.path.join("..","Models")
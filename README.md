# 09MIAR_TFM
Aplicación de modelos LLM para extracción de datos en documentos de escritura pública.

## Pasos para configurar el entorno
Con python >= 3.11

py -m venv .venv

.\.venv\Scripts\activate

py -m pip install --upgrade pip

py -m pip install -r requisitos.txt

py -m pip install PyPDF2, num2words, pandas, pyarrow, pyahocorasick, scikit-learn

## Datos
Para generar el Dataset inicial debe crearse una carpeta 'Inicial' en .\Dataset, copiar ahí los archivos de fichas y escrituras, según las rutas especificadas en el fichero 'globals.py' (o modificar esas rutas).
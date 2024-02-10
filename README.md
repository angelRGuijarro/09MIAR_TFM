# 09MIAR_TFM
Aplicación de modelos LLM para extracción de datos en documentos de escritura pública.

## Pasos para configurar el entorno
Con python >= 3.11

py -m venv .venv

.\.venv\Scripts\activate

pip install --upgrade pip

pip install -r requisitos.txt

# Se necesita torch para utilizar GPU. El comando de instalación puede variar dependiendo de su sistema
# Visite la web de PyTorch (https://pytorch.org/get-started/locally/) para más información
pip install torch --index-url https://download.pytorch.org/whl/cu121

## Datos
Para generar el Dataset inicial debe crearse una carpeta 'Inicial' en .\Dataset, copiar ahí los archivos de fichas y escrituras, según las rutas especificadas en el fichero 'globals.py' (o modificar esas rutas).
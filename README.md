# 09MIAR_TFM
Aplicación de modelos LLM para extracción de datos en documentos de escritura pública.

## Pasos para configurar el entorno
Con python = 3.11

py -m venv .venv

.\.venv\Scripts\activate

pip install --upgrade pip

pip install -r requisitos.txt

Durante el desarrollo ha surgido la siguiente incidencia:
En la Pull Request #28637 (https://github.com/huggingface/transformers/pull/28637) de la librería transformers se arregla un problema de permisos con el trainer en equipos windows, el probelma es que esta mejora se despliega con la versión 4.37.2, que no está soportado con MLflow (2.10.2), por eto he tenido que fijar la versión de transformers a la 4.37.1 y hay que modificar el código de trainer.py según el siguiente código.

        # Ensure rename completed in cases where os.rename is not atomic
-       fd = os.open(output_dir, os.O_RDONLY)
-       os.fsync(fd)
-       os.close(fd)
+       # And can only happen on non-windows based systems
+       if os.name != "nt":
+           fd = os.open(output_dir, os.O_RDONLY)
+           os.fsync(fd)
+           os.close(fd)


# Se necesita torch para utilizar GPU. El comando de instalación puede variar dependiendo de su sistema
# Visite la web de PyTorch (https://pytorch.org/get-started/locally/) para más información
pip install torch --index-url https://download.pytorch.org/whl/cu121

## Datos
Para generar el Dataset inicial debe crearse una carpeta 'Inicial' en .\Dataset, copiar ahí los archivos de fichas y escrituras, según las rutas especificadas en el fichero 'globals.py' (o modificar esas rutas).

## Modelos
Crear una carpeta ../Models/PlanTL-GOB-ES donde clonar el repositorio de los modelos
git lfs install
git clone https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-sqac
git clone https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus
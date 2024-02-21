# 09MIAR_TFM
Aplicación de modelos LLM para extracción de datos en documentos de escritura pública.

## Resumen
Código de mi TFM para el Máster en Inteligencia Artificial de la Universidad Internacional de Valencia (VIU). El propósito de este código es aplicar LLMs en la extracción estructurada de información en documentos no estructurados. concretamente en documentos de escritura pública. Los ficheros de código, en formato Jupyter Notebook (.ipynb) han sido nombrados en el orden de realización del trabajo. Siguiendo estos cuadernos, utilizando sus propios documentos podría completar el entrenamiento.

## Pasos para configurar el entorno
Con python = 3.11
```
py -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requisitos.txt
```
Durante el desarrollo ha surgido la siguiente incidencia:
En la Pull Request #28637 (https://github.com/huggingface/transformers/pull/28637) de la librería transformers se arregla un problema de permisos con el trainer en equipos windows, el probelma es que esta mejora se despliega con la versión 4.37.2, que no está soportado con MLflow (2.10.2), por eto he tenido que fijar la versión de transformers a la 4.37.1 y hay que modificar el código de trainer.py según el siguiente código.

```python
        # Ensure rename completed in cases where os.rename is not atomic
-       fd = os.open(output_dir, os.O_RDONLY)
-       os.fsync(fd)
-       os.close(fd)
+       # And can only happen on non-windows based systems
+       if os.name != "nt":
+           fd = os.open(output_dir, os.O_RDONLY)
+           os.fsync(fd)
+           os.close(fd)
```

### Se necesita torch para utilizar GPU. El comando de instalación puede variar dependiendo de su sistema
### Visite la web de PyTorch (https://pytorch.org/get-started/locally/) para más información
```pip install torch --index-url https://download.pytorch.org/whl/cu121```

## Datos
Para generar el Dataset inicial debe crearse una carpeta 'Inicial' en .\Dataset, copiar ahí los archivos de fichas y escrituras, según las rutas especificadas en el fichero 'globals.py' (o modificar esas rutas).

## Modelos pre-entrenados
Crear una carpeta ../Models/PlanTL-GOB-ES donde clonar el repositorio de los modelos
```
git lfs install
git clone https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-sqac
git clone https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus
```
### PlanTL-GOB-ES/roberta-large-bne-sqac
Asier Gutiérrez Fandiño, Jordi Armengol Estapé, Marc Pàmies, Joan Llop Palao, Joaquin Silveira Ocampo, Casimiro Pio Carrino, Carme Armentano Oller, Carlos Rodriguez Penagos, Aitor Gonzalez Agirre, Marta Villegas. **MarIA: Spanish Language Models**. *Procesamiento del Lenguaje Natural*, 68. Sociedad Española para el Procesamiento del Lenguaje Natural, 2022. DOI: [10.26342/2022-68-3](https://doi.org/10.26342/2022-68-3). Disponible en: [https://upcommons.upc.edu/handle/2117/367156#.YyMTB4X9A-0.mendeley](https://upcommons.upc.edu/handle/2117/367156#.YyMTB4X9A-0.mendeley).
### PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus
Asier Gutiérrez Fandiño, Jordi Armengol Estapé, Marc Pàmies, Joan Llop Palao, Joaquin Silveira Ocampo, Casimiro Pio Carrino, Carme Armentano Oller, Carlos Rodriguez Penagos, Aitor Gonzalez Agirre, Marta Villegas. **MarIA: Spanish Language Models**. *Procesamiento del Lenguaje Natural*, 68. Sociedad Española para el Procesamiento del Lenguaje Natural, 2022. DOI: [10.26342/2022-68-3](https://doi.org/10.26342/2022-68-3). Disponible en: [https://upcommons.upc.edu/handle/2117/367156#.YyMTB4X9A-0.mendeley](https://upcommons.upc.edu/handle/2117/367156#.YyMTB4X9A-0.mendeley).


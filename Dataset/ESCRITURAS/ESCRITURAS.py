import json
import datasets
import hashlib

class ESCRITURAS(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="QA", \
                               description="""Dataset para entrenamiento de modelos QA en extracción de datos de escrituras.
                                    id:         Identificador único en el dataset para cada pregunta-contexto.
                                    context:    Texto del documento donde se encuentra el registro.
                                    question:   Texto de la pregunta que queremos resolver.
                                        answers:Una lista de respuestas a esa pregunta que se extraen del contexto.
                                            answer_start:   Caracter de inicio donde se encuentra la respuesta.
                                            text:           Texto de la respuesta. 
                                """),
        datasets.BuilderConfig(name="NER", \
                               description="""Dataset para entrenamiento de modelos NER en extracción de datos de escrituras.
                                    Las etiquetas utilizadas se corresponden con los siguientes elementos:
                                    'B-PROTO','I-PROTO':    Número de PROTOCOLO.
                                    'B-FDOC','I-FDOC':      FECHA de firma del DOCUMENTO.
                                    'B-NOT','I-NOT':        NOTARIO, nombre y apellidos.
                                    'B-TDOC','I-TDOC':      TIPO de DOCUMENTO.
                                """)
    ]

    def _info(self):
        if self.config.name == "QA":
            info = datasets.DatasetInfo(
                features=datasets.Features({
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence({
                        "answer_start": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                    })
                })
            )
        elif self.config.name == "NER":
            info = datasets.DatasetInfo(
                description=self.config.description,
                features=datasets.Features({
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(
                        names=[
                            'O',
                            'B-PROTO','I-PROTO', # protocolo
                            'B-FDOC','I-FDOC', # fecha
                            'B-NOT','I-NOT', # notario
                            'B-TDOC','I-TDOC', # tipo documento
                        ]
                    ))
                })
            )
        return info


    DATA_URLS = {
        "QA": {
            "train": "QA_train.json",
            "dev": 'QA_validation.json',
            "test": "QA_test.json",
        },
        "NER": {
            "train": "NER_train.json",
            "dev": "NER_validation.json",
            "test": "NER_test.json",
        }
    }


    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(self.DATA_URLS[self.config.name])
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            json_data = json.load(f)
            if self.config.name == "QA":
                for d in json_data['data']:
                        for escritura in d['paragraphs']:                    
                            id = escritura['id']
                            context = escritura['context']
                            for qa in escritura['qas']:
                                question = qa['question']
                                answers = qa['answers']
                                uid = f"{id}_{hashlib.sha256(question.encode('utf-8')).hexdigest()}"                    
                                yield uid, {
                                    "id": uid,
                                    "context": context,
                                    "question": question,
                                    "answers": {
                                        "answer_start": [answer["answer_start"] for answer in answers],
                                        "text": [answer["text"] for answer in answers]
                                    },
                                }
            elif self.config.name == "NER":
                for escritura in json_data:
                    id = escritura['id']
                    yield id, {'id': id,
                                'tokens': escritura['tokens'],
                                'ner_tags': escritura['ner_tags']}


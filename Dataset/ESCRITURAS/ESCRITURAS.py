import json
import datasets
import hashlib

class ESCRITURAS(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="QA", description="Dataset para entrenamiento de modelos QA en extracción de datos de escrituras."),
        datasets.BuilderConfig(name="NER", description="Dataset para entrenamiento de modelos NER en extracción de datos de escrituras.")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "id": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answers": datasets.features.Sequence({
                    "answer_start": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                }),
            }),
        )


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
            for d in json_data['data']:
                    escritura = d['paragraph']
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


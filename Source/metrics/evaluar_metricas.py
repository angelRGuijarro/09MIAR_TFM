from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from .evaluate import compute_exact, compute_f1
import torch
import mlflow
from statistics import mean

# Definición de funciones para evaluar las métricas que se usarán en distintos cuadernos

def evaluar_metricas_QA(model_path_or_name, dataset, batch_size=512, split='test'):
    """Escribe por pantalla y guarda en MLflow los valores de las métricas f1 y exact match para este modelo y dataset."""
    model = AutoModelForQuestionAnswering.from_pretrained(model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    model.eval()    
    ## Método 1
    # qc_dataset = [{'question':q, 'context':c} for q,c in zip(dataset['question'],dataset['context'])]
    # consulta = pipeline("question-answering", model=model, tokenizer=tokenizer, 
    #                 device=0 if torch.cuda.is_available() else None, batch_size=batch_size)
    # predicciones = consulta(qc_dataset)
    
    ## Método 2
    # consulta =  pipeline("question-answering", model=model, device=0 if torch.cuda.is_available() else None, batch_size=batch_size)
    # predicciones = dataset.map(lambda ejemplo: {'prediccion': consulta({'question':ejemplo['question'], 'context':ejemplo['context']})})
    
    # # Método 3
    consulta =  pipeline("question-answering", model=model, tokenizer=tokenizer, 
                    device=0 if torch.cuda.is_available() else None, batch_size=batch_size)        
    predicciones = consulta(question=dataset['question'], context=dataset['context'])

    # Ejecución y cálculo de métricas
    gold_answers = [answer['text'][0] for answer in dataset['answers']]
    pred_answers = [pred['answer'] for pred in predicciones]
    f1_scores = [compute_f1(g,p) for g,p in zip(gold_answers,pred_answers)]
    exact_scores = [compute_exact(g,p) for g,p in zip(gold_answers,pred_answers)]
    
    f1_mean = mean(f1_scores)
    exact_mean = mean(exact_scores)
    
    mlflow.log_param('dataset_split', split)
    mlflow.log_param('model_name',model.name_or_path)
    mlflow.log_metric('f1', f1_mean)
    mlflow.log_metric('exact', exact_mean)
    print(model.name_or_path)
    print('\tf1:', f1_mean)
    print('\texact:', exact_mean)

def evaluar_metricas_NER(model_path_or_name, dataset, batch_size=512):
    pass
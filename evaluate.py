import json
import spacy
import requests
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from scispacy.linking import EntityLinker
from bert_score import score as bert_score
import argparse 
import os

# configuration
parser = argparse.ArgumentParser(description='Evaluate Clinical Report Extraction')
parser.add_argument(
    '--test_file',
    type=str,
    required=True,
    help='Path to the test results JSON file'
)
parser.add_argument(
    '--task',
    type=str,
    default='naive',
    help='Task name (naive, advanced_icl, finetuned, etc.)'
)
parser.add_argument(
    '--model',
    type=str,
    default='Llama-3.1-8B-Instruct',
    help='Model name for output files'
)
parser.add_argument(
    '--lang',
    type=str,
    default='en',
    choices=['en', 'de'],
    help='Language (en or de)'
)
parser.add_argument(
    '--gpu_id',
    type=int,
    default=0,
    help='GPU ID for processing (default: 0)'
)
parser.add_argument(
    '--bert_gpu_id',
    type=int,
    default=1,
    help='GPU ID for BERTScore (default: 1)'
)

args = parser.parse_args()

# Configuration from arguments
LANG = args.lang
TASK = args.task
MODEL_NAME = args.model
TEST_FILE_PATH = args.test_file
GPU_ID = args.gpu_id
BERT_GPU_ID = args.bert_gpu_id
DATA_PATH = os.path.dirname(TEST_FILE_PATH)
FILENAME = os.path.basename(TEST_FILE_PATH)
BERT_SCORE_MODEL = {
    "en": "microsoft/deberta-xlarge-mnli",
    "de": "microsoft/deberta-xlarge-mnli"
}

ENTITY_MODEL = {
    "en": "en_core_sci_scibert",
    "de": "en_core_sci_scibert"
}

data = json.load( 
    open(
        file=TEST_FILE_PATH,
        mode="r",
        encoding="utf-8"
    )
)

# Load SciSpacy model for clinical NER and Initialize UMLS Entity Linker
spacy.require_gpu(gpu_id=3)
nlp = spacy.load(ENTITY_MODEL[LANG])
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}, last=True)
linker = EntityLinker(resolve_abbreviations=True, name="umls")

# Initialize accumulators for metrics
rouge_metrics = {}
bert_metrics = {}
entity_metrics = {}

# Cache snomed results
cached_snomed_results = {}

# Fields to evaluate (excluded "age", "gender")
fields = [
    "life_style", "family_history", "social_history", "medical_surgical_history",
    "signs_symptoms", "comorbidities", "diagnostic_techniques_procedures",
    "diagnosis", "laboratory_values", "pathology", "pharmacological_therapy",
    "interventional_therapy", "patient_outcome_assessment"
]

fields_with_medical_entities = {
    "signs_symptoms", "comorbidities", "diagnostic_techniques_procedures",
    "diagnosis", "pathology", "pharmacological_therapy",
    "interventional_therapy", "patient_outcome_assessment"
}

# Initialize metrics dictionary
for field in fields:
    rouge_metrics[field] = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bert_metrics[field] = []
    if field in fields_with_medical_entities:
        entity_metrics[field] = {'precision': [], 'recall': [], 'f1': []}

def get_snomed_code(search_term):
    if search_term in cached_snomed_results:
        return cached_snomed_results[search_term]

    response = requests.get(
        url='https://xxx.healthai.fit.fraunhofer.de/browser/MAIN/descriptions',
        headers={
            'accept': 'application/json',
            'Accept-Language': 'en-X-900000000000509007,en-X-900000000000508004,en'
        },
        params={
            'term': search_term,
            'groupByConcept': 'false',
            'searchMode': 'STANDARD',
            'offset': 0,
            'limit': 50
        }
    )

    if response.status_code == 200:
        response_data = response.json()
        if 'items' in response_data and len(response_data['items']) > 0:
            first_item = response_data['items'][0]
            cached_snomed_results[search_term] = {
                'term': first_item['term'],
                'concept_id': first_item['concept']['conceptId']
            }
            return cached_snomed_results[search_term]
        else:
            return None
    else:
        return None

# Function to extract entities and link them to SNOMED CT concepts
def extract_snomed_entities(text):
    doc = nlp(text)
    snomed_concepts = set()
    for ent in doc.ents:
        cui = ent._.kb_ents[0][0] if len(ent._.kb_ents) > 0 else None
        if cui:
            # Retrieve the UMLS concept
            concept = linker.kb.cui_to_entity[cui]

            # Use carlac fit sevice to get snomed data
            # if concept and len(concept) > 0:
            #     search_term = concept[1]
            # else:
            #     search_term = ent.text
            # snomed_data = get_snomed_code(search_term)
            # if snomed_data:
            #     snomed_concepts.add(snomed_data["concept_id"])

            if concept and len(concept) > 0:
                snomed_concepts.add(concept[0])

    return snomed_concepts

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
entity_comparison_data = []
# Iterate over each example
for example in tqdm(data):
    if 'model_prediction' not in example:
        continue
    gold_summary = example['summary']
    model_pred = example.get('model_prediction', None)

    # Check if model_prediction is a valid dictionary/json
    if not isinstance(model_pred, dict):
        continue

    for field in fields:
        gold_text = gold_summary.get(field, "").strip()
        pred_text = model_pred.get(field, "")

        if not isinstance(pred_text, str):
            continue
        else:
            pred_text = pred_text.strip()

        # Do not consider NA values for evaluation
        if gold_text == "N/A":
            continue

        # ROUGE Scores
        rouge = scorer.score(gold_text, pred_text)
        rouge_metrics[field]['rouge1'].append(rouge['rouge1'].fmeasure)
        rouge_metrics[field]['rouge2'].append(rouge['rouge2'].fmeasure)
        rouge_metrics[field]['rougeL'].append(rouge['rougeL'].fmeasure)

        # Entity-Level Metrics with SNOMED CT concepts
        if field in fields_with_medical_entities:
            gold_entities = extract_snomed_entities(gold_text)
            try:
                pred_entities = extract_snomed_entities(pred_text)
            except Exception as e:
                print(e)
                pred_entities = set()

            #save the entities for the further study    
            entity_comparison_data.append({
                "patient_id": example.get("patient_id", "ID_NOT_FOUND"),
                "field": field,
                "gold_entities": list(gold_entities), # Convert set to list for JSON serialization
                "pred_entities": list(pred_entities)
            })

            common_entities = gold_entities.intersection(pred_entities)
            entity_precision = len(common_entities) / len(pred_entities) if pred_entities else 0
            entity_recall = len(common_entities) / len(gold_entities) if gold_entities else 0
            entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) else 0
            entity_metrics[field]['precision'].append(entity_precision)
            entity_metrics[field]['recall'].append(entity_recall)
            entity_metrics[field]['f1'].append(entity_f1)

# Compute BERTScore for each field
for field in tqdm(fields):
    refs= []
    preds = []
    for example in data:
        if 'model_prediction' not in example:
            continue
        gold_summary = example['summary']
        model_pred = example['model_prediction']

        if not isinstance(model_pred, dict) or not isinstance(model_pred.get(field, ""), str):
            continue

        refs.append(gold_summary.get(field, "").strip())
        preds.append(model_pred.get(field, "").strip())

    P, R, F1 = bert_score(
        preds,
        refs,
        model_type=BERT_SCORE_MODEL[LANG],
        lang=LANG,
        rescale_with_baseline=True,
        device = 1
    )
    bert_metrics[field] = F1.numpy()

# Save results
results_file_name = f"{DATA_PATH}/{FILENAME}_results.txt"
with open(results_file_name, mode="w", encoding="utf-8") as results_file:
    results_file.write("=== ROUGE Scores ===\n")
    for field in fields:
        rouge1 = np.mean(rouge_metrics[field]['rouge1'])
        rouge2 = np.mean(rouge_metrics[field]['rouge2'])
        rougeL = np.mean(rouge_metrics[field]['rougeL'])
        results_file.write(f"{field}: ROUGE-1={rouge1:.4f}, ROUGE-2={rouge2:.4f}, ROUGE-L={rougeL:.4f}\n")
    results_file.write("\n")

    results_file.write("=== BERTScore ===\n")
    for idx, field in enumerate(fields):
        f1 = np.mean(bert_metrics[field])
        results_file.write(f"{field}: BERTScore F1={f1:.4f}\n")
    results_file.write("\n")

    results_file.write("=== Entity-Level (UMLS) Metrics ===\n")
    for field in fields_with_medical_entities:
        precision = np.mean(entity_metrics[field]['precision'])
        recall = np.mean(entity_metrics[field]['recall'])
        f1 = np.mean(entity_metrics[field]['f1'])
        results_file.write(f"{field}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")
    results_file.write("\n")

print(f"Results have been saved to {results_file_name}")

# save entities comparision 
entity_comparison_file_name = f"{DATA_PATH}/{FILENAME}_entity_comparison.json"
with open(entity_comparison_file_name, mode="w", encoding="utf-8") as f:
    json.dump(entity_comparison_data, f, indent=4)

print(f"Entity comparison results have been saved to {entity_comparison_file_name}")

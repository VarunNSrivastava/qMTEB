import gradio as gr
import pandas as pd
import json
import os


TASKS_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASKS_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]


MODELS = [
    "all-MiniLM-L6-v2"
]


def get_model_size(model_name):
    return os.path.getsize(f"models/{model_name}") / (1024.0 * 1024.0)


def compute_model_score(model_name):
    results_dir = "results"
    model_dir = os.path.join(results_dir, model_name)

    scores = []

    # Get scores for clustering tasks
    for task in TASKS_CLUSTERING:
        task_file = os.path.join(model_dir, f"{task}.json")
        with open(task_file, 'r') as f:
            data = json.load(f)
            v_measure = data['test']['v_measure']
            scores.append(v_measure)

    # Get scores for pair classification tasks
    for task in TASKS_PAIR_CLASSIFICATION:
        task_file = os.path.join(model_dir, f"{task}.json")
        with open(task_file, 'r') as f:
            data = json.load(f)
            max_ap = data['test']['max']['ap']
            scores.append(max_ap)

    # Compute average score
    average_score = sum(scores) / len(scores)
    return average_score


DATA = {
        "Model": MODELS,
        "Model Size (MB)": [
                get_model_size(f"{model}/pytorch_model.bin") for model in MODELS
            ],
        "Score": [
                compute_model_score(model) for model in MODELS
            ],
        "q8 Model Size (MB)": [
                get_model_size(f"optimum/{model}-self-optimum-q8/model.onnx") for model in MODELS
            ],
        "q8 Score": [
                compute_model_score(f"optimum/{model}-q8") for model in MODELS
            ],
    }

with open('data.json', 'w') as json_file:
    json.dump(DATA, json_file)




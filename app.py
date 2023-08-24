import gradio as gr
import pandas as pd
import json
import os

# Given list of tasks for clustering and pair classification
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

def display_table():
    # Create a sample dataframe
    data = {
        "Model": ["ModelA", "ModelB", "ModelC"],
        "Model Size (MB)": [293, 793, 1000],
        "Score": [0.92, 0.85, 0.89],
        "Quantized Score": [0.91, 0.84, 0.88]
    }
    df = pd.DataFrame(data)

    df.index.name = "Rank"
    html_table = df.to_html()

    html_content = f"""
    <style>
        .wide_table {{
            width: 100%;
        }}
    </style>
    {html_table}
    """

    return html_content


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


# score = compute_model_score("ModelA")

# Create Gradio interface
iface = gr.Interface(fn=display_table, live=True, inputs=[], outputs="html")

iface.launch()

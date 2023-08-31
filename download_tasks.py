"""Downloads MTEB tasks"""
import os

TASK_LIST = [
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

    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]


os.environ["TRANSFORMERS_CACHE"]="./transformers_cache/"
os.environ["HF_DATASETS_CACHE"]="./hf_datasets_cache/"
os.environ["HF_MODULES_CACHE"]="./hf_modules_cache/"
os.environ["HF_METRICS_CACHE"]="./hf_metrics_cache/"

from mteb import MTEB
evaluation = MTEB(tasks=TASK_LIST, task_langs=["en"])

for task in evaluation.tasks:
    print(f"Loading {task}")
    task.load_data()

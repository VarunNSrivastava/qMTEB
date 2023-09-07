import argparse
import logging
import os
import time

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

os.environ["HF_DATASETS_OFFLINE"] = "1"  # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 1 for offline
os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache/"
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets_cache/"
os.environ["HF_MODULES_CACHE"] = "./hf_modules_cache/"
os.environ["HF_METRICS_CACHE"] = "./hf_metrics_cache/"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"



TASK_LIST_CLUSTERING = [
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

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST = TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--startid", type=int)
    # parser.add_argument("--endid", type=int)

    parser.add_argument("--modelpath", type=str, default="./models/")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--device", type=str, default="mps")  # sorry :>
    args = parser.parse_args()
    return args


def main(args):
    """
    ex: python run_array.py --modelpath ./models/all-MiniLM-L6-v2
    """
    model = SentenceTransformer(args.modelpath, device=args.device)
    model_name = args.modelpath.split("/")[-1].split("_")[-1]
    if not model_name:
        print(f"Model name is empty. Make sure not to end modelpath with a /")
        return

    print(f"Running on {model._target_device} with model {model_name}.")

    for task in TASK_LIST:
        print("Running task: ", task)
        # this args. notation seems anti-pythonic
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        retries = 5
        for attempt in range(retries):
            try:
                evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=["test"])
                break
            except ConnectionError:
                if attempt < retries - 1:
                    print(f"Connection error occurred during task {task}. Waiting for 1 minute before retrying...")
                    time.sleep(60)
                else:
                    print(f"Failed to execute task {task} after {retries} attempts due to connection errors.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

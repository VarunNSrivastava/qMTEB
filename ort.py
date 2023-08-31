from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch

tokenizer = AutoTokenizer.from_pretrained("./models/optimum/all-MiniLM-L6-v2")
model = ORTModelForFeatureExtraction.from_pretrained("./models/optimum/all-MiniLM-L6-v2")

inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

# no mean pooling
print(list(last_hidden_state.shape))

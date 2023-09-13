import torch
from transformers import AutoModel
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 1 for offline

model_fp32 = AutoModel.from_pretrained("./models/all-MiniLM-L6-v2")

model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.float16)

torch.save(model_int8.state_dict(), "./models/all-MiniLM-L6-v2-unquantized-q16/pytorch_model.bin")

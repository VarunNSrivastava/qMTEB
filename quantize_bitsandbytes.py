import os
from transformers import AutoModel
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# Make sure transformers works offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 1. Initialize the empty model
model_fp32 = AutoModel.from_pretrained("./models/all-MiniLM-L6-v2")
with init_empty_weights():
    empty_model = model_fp32

# 2. Get the path to the weights of your model. For now, we'll assume it's in the same folder.
weights_location = "./models/all-MiniLM-L6-v2-unquantized/pytorch_model.bin"

# 3. Set quantization configuration (8-bit for this example)
bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold=6)

# 4. Quantize the empty model
quantized_model = load_and_quantize_model(empty_model, weights_location=weights_location,
                                          bnb_quantization_config=bnb_quantization_config, device_map="auto")

# 5. Save the quantized model
accelerator = Accelerator()
new_weights_location = "./models/all-MiniLM-L6-v2-unquantized-q8"
accelerator.save_model(quantized_model, new_weights_location)

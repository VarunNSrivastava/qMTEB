import torch
from torch.ao.quantization import quantize_dynamic
from optimum.fx.optimization import Transformation
from transformers import AutoModel, AutoTokenizer
from transformers.utils.fx import symbolic_trace

# Define the Dynamic Quantization Transformation
class DynamicQuantization(Transformation):
    def __init__(self, dtype=torch.qint8, qconfig_spec=None, mapping=None):
        super().__init__()
        self.dtype = dtype
        self.qconfig_spec = qconfig_spec
        self.mapping = mapping

    def transform(self, graph_module):
        # Use torch's quantize_dynamic function to quantize the module
        quantized_module = quantize_dynamic(
            graph_module, qconfig_spec=self.qconfig_spec, dtype=self.dtype, mapping=self.mapping, inplace=False
        )
        return quantized_module

# Load the model
model_path = "./models/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Symbolically trace the model
# Note: For certain models, you might need to modify the input_names
input_names = ["input_ids", "attention_mask"]
traced_model = symbolic_trace(model, input_names=input_names)

# Apply dynamic quantization
transformation = DynamicQuantization(dtype=torch.qint8)
quantized_model = transformation(traced_model)

print(type(quantized_model.))
#
# # Save the quantized model
# quantized_model_path = "./models/all-MiniLM-L6-v2-unquantized-q8/"
# quantized_model.save(quantized_model_path)
# tokenizer.save_pretrained(quantized_model_path)  # Save the tokenizer as well

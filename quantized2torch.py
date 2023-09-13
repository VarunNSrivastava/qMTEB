import sys
import os
from onnx2torch import convert
import torch

def onnx_to_torch_converter(dir_name):
    if not os.path.exists(dir_name):
        print(f"Directory {dir_name} does not exist!")
        return

    onnx_model_path = os.path.join(dir_name, "onnx", "model.onnx")

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model at {onnx_model_path} does not exist!")
        return

    torch_model = convert(onnx_model_path)

    torch_model_save_path = os.path.join(dir_name, "pytorch_model.bin")
    torch.save(torch_model.state_dict(), torch_model_save_path)
    print(f"PyTorch model saved at {torch_model_save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onnx2torch.py [directory_path]")
    else:
        dir_name = sys.argv[1]
        onnx_to_torch_converter(dir_name)

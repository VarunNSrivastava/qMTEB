import sys
import os

import onnx
from onnx_tf.backend import prepare

def onnx_to_torch_converter(dir_name):
    if not os.path.exists(dir_name):
        print(f"Directory {dir_name} does not exist!")
        return

    onnx_model_path = os.path.join(dir_name, "onnx", "model.onnx")

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model at {onnx_model_path} does not exist!")
        return

    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_model_save_path = os.path.join(dir_name, "tf_model")

    tf_rep.export_graph(tf_model_save_path)  # export the model

    print(f"PyTorch model saved at {tf_model_save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onnx2torch.py [directory_path]")
    else:
        dir_name = sys.argv[1]
        onnx_to_torch_converter(dir_name)

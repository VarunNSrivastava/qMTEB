import os
from dataclasses import dataclass, field
from typing import Optional, Set

import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)

from optimum.exporters.tasks import TasksManager
from transformers import (
    AutoConfig,
    HfArgumentParser
)

DEFAULT_QUANTIZE_PARAMS = {
    'per_channel': True,
    'reduce_range': True,
}

MODEL_SPECIFIC_QUANTIZE_PARAMS = {
    'whisper': {
        'per_channel': False,
        'reduce_range': False,
    }
}

MODELS_WITHOUT_TOKENIZERS = [
    'wav2vec2'
]


@dataclass
class ConversionArguments:
    """
    Arguments used for converting HuggingFace models to onnx.
    """

    model_id: str = field(
        metadata={
            "help": "Model identifier"
        }
    )
    quantize: bool = field(
        default=False,
        metadata={
            "help": "Whether to quantize the model."
        }
    )
    output_parent_dir: str = field(
        default='./models/',
        metadata={
            "help": "Path where the converted model will be saved to."
        }
    )

    task: Optional[str] = field(
        default='auto',
        metadata={
            "help": (
                "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
                f" {str(list(TasksManager._TASKS_TO_AUTOMODELS.keys()))}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
            )
        }
    )

    opset: int = field(
        default=None,
        metadata={
            "help": (
                "If specified, ONNX opset version to export the model with. Otherwise, the default opset will be used."
            )
        }
    )

    device: str = field(
        default='cpu',
        metadata={
            "help": 'The device to use to do the export.'
        }
    )
    skip_validation: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip validation of the converted model"
        }
    )

    per_channel: bool = field(
        default=None,
        metadata={
            "help": "Whether to quantize weights per channel"
        }
    )
    reduce_range: bool = field(
        default=None,
        metadata={
            "help": "Whether to quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode"
        }
    )

    output_attentions: bool = field(
        default=False,
        metadata={
            "help": "Whether to output attentions from the model. NOTE: This is only supported for whisper models right now."
        }
    )

    split_modalities: bool = field(
        default=False,
        metadata={
            "help": "Whether to split multimodal models. NOTE: This is only supported for CLIP models right now."
        }
    )


def get_operators(model: onnx.ModelProto) -> Set[str]:
    operators = set()

    def traverse_graph(graph):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = attr.g
                    traverse_graph(subgraph)

    traverse_graph(model.graph)
    return operators


def quantize(model_path):
    """
    Quantize the weights of the model from float32 to int8 to allow very efficient inference on modern CPU

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    directory_path = os.path.dirname(model_path)

    loaded_model = onnx.load_model(model_path)
    op_types = get_operators(loaded_model)
    weight_type = QuantType.QUInt8 if 'Conv' in op_types else QuantType.QInt8
    print("quantizing to", weight_type)

    quantize_dynamic(
        model_input=model_path,
        model_output=os.path.join(directory_path, 'model-q8.onnx'),
        weight_type=weight_type,
        optimize_model=False,
    )


def main():
    """
    Example usage:
    python quantize_onnx.py --model_id sentence-transformers/all-MiniLM-L6-v2-unquantized
    """
    parser = HfArgumentParser(
        (ConversionArguments,)
    )
    conv_args, = parser.parse_args_into_dataclasses()

    model_id = conv_args.model_id

    quantize(os.path.join(model_id, "model.onnx"))


if __name__ == '__main__':
    main()

"""Build a tiny ONNX "detector" for tests: it flags any window containing token id TRIGGER.

It has the same interface as the real exported model (input_ids, attention_mask -> logits
[batch, 2]), so the whole serving stack runs against real ONNX Runtime in CI without
downloading a 100 MB model.  logits = [0, 10*hit - 5]  ->  P(injection) ~ 0.993 or ~ 0.007.

Usage: python tests/fixtures/make_tiny_model.py tests/fixtures/tiny_model
"""

import json
import sys
from pathlib import Path

from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
)

TRIGGER = 7


def _value_info(name, elem_type, dims):
    shape = TensorShapeProto(dim=[TensorShapeProto.Dimension(dim_param=d) if isinstance(d, str)
                                  else TensorShapeProto.Dimension(dim_value=d) for d in dims])
    tt = TypeProto(tensor_type=TypeProto.Tensor(elem_type=elem_type, shape=shape))
    return ValueInfoProto(name=name, type=tt)


def _const(name, elem_type, value):
    if elem_type == TensorProto.INT64:
        t = TensorProto(name=name, data_type=elem_type, dims=[], int64_data=[value])
    else:
        t = TensorProto(name=name, data_type=elem_type, dims=[], float_data=[value])
    return t


def _node(op, inputs, outputs, **attrs):
    n = NodeProto(op_type=op, input=inputs, output=outputs)
    for key, val in attrs.items():
        if isinstance(val, list):
            n.attribute.append(AttributeProto(name=key, type=AttributeProto.INTS, ints=val))
        else:
            n.attribute.append(AttributeProto(name=key, type=AttributeProto.INT, i=val))
    return n


def build() -> ModelProto:
    nodes = [
        _node("Equal", ["input_ids", "trigger"], ["eq"]),
        _node("Cast", ["eq"], ["eq_f"], to=TensorProto.FLOAT),
        _node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        _node("Mul", ["eq_f", "mask_f"], ["hits"]),
        _node("ReduceMax", ["hits"], ["hit"], axes=[1], keepdims=1),
        _node("Mul", ["hit", "ten"], ["scaled"]),
        _node("Add", ["scaled", "minus_five"], ["inj"]),
        _node("Mul", ["hit", "zero"], ["safe"]),
        _node("Concat", ["safe", "inj"], ["logits"], axis=1),
    ]
    graph = GraphProto(
        name="tiny_detector",
        node=nodes,
        initializer=[_const("trigger", TensorProto.INT64, TRIGGER),
                     _const("ten", TensorProto.FLOAT, 10.0),
                     _const("minus_five", TensorProto.FLOAT, -5.0),
                     _const("zero", TensorProto.FLOAT, 0.0)],
        input=[_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
               _value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])],
        output=[_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
    )
    return ModelProto(ir_version=8, graph=graph, producer_name="sentinel-tests",
                      opset_import=[OperatorSetIdProto(domain="", version=17)])


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tiny_model")
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_bytes(build().SerializeToString())
    config = {"cls_id": 0, "sep_id": 2, "pad_id": 1, "max_length": 16, "stride": 4,
              "threshold": 0.5, "max_chunks": 8, "onnx_file": "model.onnx", "version": "tiny"}
    (out / "sentinel_config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {out}")

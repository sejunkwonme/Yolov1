import onnx
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from model import Yolov1Model
from onnx import numpy_helper

import numpy as np
from model import Yolov1Model

m = onnx.load("yolov1_raw.onnx")
print(m.opset_import)

for n in m.graph.initializer:
    print(n.name, n.data_type)

w = onnx.numpy_helper.to_array(n)
print(w.shape)

for n in m.graph.node:
    if n.op_type == "Conv":
        w_name = n.input[1]
        for init in m.graph.initializer:
            if init.name == w_name:
                arr = numpy_helper.to_array(init)
                print("Conv:", n.name, "weight shape:", arr.shape)

for n in m.graph.node:
    if n.op_type == "Conv":
        print("\nConv:", n.name)
        print("  inputs:", n.input)

for init in m.graph.initializer:
    if "val_" in init.name:
        arr = numpy_helper.to_array(init)
        print(init.name, arr.shape, arr.dtype, arr)

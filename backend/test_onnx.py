import numpy as np
import onnxruntime as ort
from pathlib import Path

model_path = Path("models/aquaclass_mobilenetv2.onnx")

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# Test image tensor
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run prediction
outputs = session.run(None, {input_name: dummy_input})

print("ONNX model loaded successfully!")
print("Input name:", input_name)
print("Output shape:", outputs[0].shape)
print("Inference test successful!")

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

# Project paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "models" / "aquaclass_mobilenetv2_best.pth"
OUTPUT_PATH = BASE_DIR / "models" / "aquaclass_mobilenetv2.onnx"

CLASSES = [
    "Amphiprion_clarkii",
    "Chaetodon_lunulatus",
    "Chromis_chrysura",
    "Dascyllus_reticulatus",
    "Plectroglyphidodon_dickii"
]

# Check weights
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

# Create MobileNetV2 architecture
model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(CLASSES))

# Load trained weights
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Dummy input: batch, channels, height, width
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    str(OUTPUT_PATH),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX export successful!")
print("Saved at:", OUTPUT_PATH)
print("File size:", round(OUTPUT_PATH.stat().st_size / (1024 * 1024), 2), "MB")

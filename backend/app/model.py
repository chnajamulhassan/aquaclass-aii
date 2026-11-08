import os
import onnxruntime as ort

CLASSES = [
    "Amphiprion_clarkii",
    "Chaetodon_lunulatus",
    "Chromis_chrysura",
    "Dascyllus_reticulatus",
    "Plectroglyphidodon_dickii"
]


def load_trained_model(model_path=None):
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(
            base_dir,
            "models",
            "aquaclass_mobilenetv2.onnx"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    return session
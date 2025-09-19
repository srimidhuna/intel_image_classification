import os
import traceback
import numpy as np
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model

model = load_model("my_model.h5", compile=False)

inp = None
try:
    inp = model.input_shape
except Exception:
    try:
        inp = model.layers[0].input_shape
    except Exception:
        inp = None

if inp and len(inp) == 4:
    _, H, W, C = inp
else:
    H, W, C = 200, 200, 3

H = int(H) if H is not None else 200
W = int(W) if W is not None else 200
C = int(C) if C is not None else 3

class_names = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]

def preprocess_image(img):
    if img is None:
        raise ValueError("No image provided")
    if C == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize((W, H))
    arr = np.asarray(img).astype(np.float32)
    if C == 1 and arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    if arr.shape[-1] != C:
        if arr.shape[-1] > C:
            arr = arr[..., :C]
        else:
            pad = np.zeros((*arr.shape[:2], C - arr.shape[-1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=-1)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def safe_softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def predict_fn(img):
    try:
        arr = preprocess_image(img)
        print("DEBUG input shape:", arr.shape)
        preds = model.predict(arr)
        preds = np.asarray(preds).squeeze()
        print("DEBUG raw preds shape:", preds.shape, "values:", preds.tolist() if hasattr(preds, "tolist") else preds)
        if preds.ndim == 0:
            probs = np.array([float(preds)])
        else:
            s = float(np.sum(preds))
            if s <= 0 or s > 1.0001:
                probs = safe_softmax(preds)
            else:
                probs = preds.astype(float)
        probs = probs.flatten()
        if len(class_names) != len(probs):
            if len(class_names) < len(probs):
                names = class_names + [f"Class_{i}" for i in range(len(class_names), len(probs))]
            else:
                names = class_names[: len(probs)]
        else:
            names = class_names
        out = {names[i]: float(probs[i]) for i in range(len(probs))}
        return out, ""
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return {}, tb

iface = gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=6), gr.Textbox(label="Debug")],
    title="Image Classifier",
    description=f"Model expects input shape ({H},{W},{C}). Replace class_names with your real labels."
)

if __name__ == "__main__":
    iface.launch(debug=True)




from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
import cv2

app = FastAPI(title="🔬 Microscope AI Production API (Drug Discovery Team)")
try:
    ort_session = ort.InferenceSession("microscope_unet_256.onnx", providers=['CPUExecutionProvider'])
except Exception:
    ort_session = None

def apply_clahe_per_channel(img_fused):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = np.zeros_like(img_fused)
    for c in range(img_fused.shape[2]):
        ch = img_fused[:, :, c]
        if ch.dtype != np.uint8:
            ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        equalized[:, :, c] = clahe.apply(ch)
    return equalized

def min_max_normalize(img_fused):
    img_fused = img_fused.astype(np.float32)
    for c in range(img_fused.shape[2]):
        min_val = img_fused[:, :, c].min()
        max_val = img_fused[:, :, c].max()
        if (max_val - min_val) > 0:
            img_fused[:, :, c] = (img_fused[:, :, c] - min_val) / (max_val - min_val)
        else:
            img_fused[:, :, c] = 0.0
    return img_fused

def morph_clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = (mask > 0).astype(np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

@app.post("/predict")
async def predict_microscope(
    channel1: UploadFile = File(...), channel2: UploadFile = File(...),
    channel3: UploadFile = File(...), channel4: UploadFile = File(...)
):
    if ort_session is None: return {"error": "Model runtime is offline."}
    channels = []
    for f in [channel1, channel2, channel3, channel4]:
        b = await f.read()
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        channels.append(cv2.resize(img, (256, 256)))

    fused = np.stack(channels, axis=-1)
    fused = min_max_normalize(apply_clahe_per_channel(fused))
    tensor = np.expand_dims(np.transpose(fused, (2, 0, 1)), axis=0).astype(np.float32)

    out = ort_session.run(None, {ort_session.get_inputs()[0].name: tensor})
    raw_mask = np.argmax(out[0], axis=1)[0]
    refined_mask = morph_clean_mask(raw_mask)

    return {
        "status": "success",
        "prediction_shape": list(refined_mask.shape),
        "mask_sample": refined_mask[:5, :5].tolist()
    }

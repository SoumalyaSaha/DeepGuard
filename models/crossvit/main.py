"""
CViT2 — Convolutional Vision Transformer for Video Deepfake Detection
Uses weights from: https://huggingface.co/datasets/Deressa/cvit
Paper: "Improved Deepfake Video Detection Using Convolutional Vision Transformer" (Wodajo et al.)

Weight file: weights/crossvit.pth
"""

import io, time, logging, os, math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crossvit")

app = FastAPI(title="CViT2 Video Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/crossvit.pth")
FRAMES_TO_SAMPLE = int(os.getenv("FRAMES_TO_SAMPLE", "15"))
IMG_SIZE = 224

model = None

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading CViT2 weights from {WEIGHTS_PATH}")
        try:
            # CViT2 checkpoint is a full saved model (torch.save(model, ...))
            model = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            if hasattr(model, 'eval'):
                model.eval()
                logger.info("CViT2 model loaded as full object ✓")
            else:
                # It's a state dict — build the model first
                raise ValueError("Got state dict, need model object")
        except Exception as e:
            logger.warning(f"Full model load failed ({e}), trying state_dict approach")
            try:
                from transformers import ViTForImageClassification
                model = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224",
                    num_labels=2,
                    ignore_mismatched_sizes=True,
                )
                ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
                state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                model = model.to(DEVICE).eval()
                logger.info("CViT2 loaded via ViT state_dict ✓")
            except Exception as e2:
                logger.error(f"State dict load also failed: {e2}")
                model = None
    else:
        logger.warning(f"No weights at {WEIGHTS_PATH}")
        model = None

    logger.info(f"CrossViT ready on {DEVICE}, model={'loaded' if model else 'MISSING'}")


def _sample_frames(video_bytes: bytes, n: int) -> list:
    import cv2, tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError("Could not read video")
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    os.unlink(tmp_path)
    return frames


def _predict_frames(frames: list) -> float:
    probs = []
    for img in frames:
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)
            # Handle HuggingFace output or raw tensor
            if hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            if logits.shape[-1] == 2:
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                prob = torch.sigmoid(logits).item()
            probs.append(prob)
    return float(np.mean(probs)) if probs else 0.5


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "CViT2",
        "device": str(DEVICE),
        "weights_loaded": model is not None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    if model is None:
        raise HTTPException(503, "Model weights not loaded")
    try:
        data = await file.read()
        fname = (file.filename or "upload").lower()

        if any(fname.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
            frames = [Image.open(io.BytesIO(data)).convert("RGB")]
        else:
            frames = _sample_frames(data, FRAMES_TO_SAMPLE)

        if not frames:
            raise ValueError("No frames extracted")

        prob = _predict_frames(frames)

        return {
            "model": "CViT2",
            "fake_probability": round(prob, 4),
            "verdict": "fake" if prob > 0.5 else "real",
            "frames_analyzed": len(frames),
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"CViT2 detect error: {e}")
        raise HTTPException(500, detail=str(e))

"""
NPR -- Noise Pattern Recognition image deepfake detector
CNNDetection ResNet-50 weights.
NOTE: CNNDetection outputs high score for REAL images (trained with label 0=fake, 1=real)
So we INVERT the output: fake_probability = 1 - model_output
Weight file: weights/npr.pth
"""
import io, time, logging, os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("npr")
app = FastAPI(title="NPR Image Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/npr.pth")


class NPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights=None)
        base.fc = nn.Linear(2048, 1)
        self.net = base

    def forward(self, x):
        return torch.sigmoid(self.net(x))


model = None
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.on_event("startup")
async def load_model():
    global model
    model = NPRModel().to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading NPR weights from {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("NPR weights loaded OK")
    else:
        logger.warning(f"No weights at {WEIGHTS_PATH}")
    model.eval()
    logger.info(f"NPR ready on {DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "NPR",
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = model(tensor).item()
        # CNNDetection: high score = real, low score = fake
        # Invert so that fake_probability is high for fake images
        fake_prob = 1.0 - raw
        return {
            "model": "NPR",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

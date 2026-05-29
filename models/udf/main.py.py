"""
UFD — Universal Fake Detect
Image deepfake detector using ResNet50 — runs on GPU (CUDA) if available
"""
import io, time, logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ufd")

app = FastAPI(title="UFD Image Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"UFD using device: {DEVICE}")

# ── Model definition ──────────────────────────────────────────────────────────
class UFDModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights=None)
        base.fc = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.net = base

    def forward(self, x):
        return self.net(x)

model: UFDModel = None

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

@app.on_event("startup")
async def load_model():
    global model
    model = UFDModel().to(DEVICE)
    # TODO: load real weights when available
    # model.load_state_dict(torch.load("../../weights/ufd.pth", map_location=DEVICE))
    model.eval()
    logger.info("UFD model loaded")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": "UFD", "device": str(DEVICE)}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = model(tensor).item()
        latency = int((time.time() - t0) * 1000)
        return {
            "model": "UniversalFakeDetect",
            "fake_probability": round(prob, 4),
            "verdict": "fake" if prob > 0.5 else "real",
            "latency_ms": latency,
        }
    except Exception as e:
        logger.error(f"UFD error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

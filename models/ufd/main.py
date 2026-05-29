"""
UFD — UniversalFakeDetect image deepfake detector
Uses CLIP (ViT-L/14) features + a linear classifier from:
  "Towards Universal Fake Image Detection by Training on an Arbitrary GAN"
  (Ojha et al., CVPR 2023) — https://github.com/WisconsinAIVision/UniversalFakeDetect

Weight file expected: weights/ufd.pth  (the fc_weights.pth from the repo)
CLIP backbone: loaded automatically via OpenAI CLIP (no local file needed)

Download via: python download_weights.py
"""

import io, time, logging, os
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ufd")

app = FastAPI(title="UFD Image Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/ufd.pth")

clip_model = None
preprocess = None
classifier = None


class LinearClassifier(nn.Module):
    """
    Linear probe on top of CLIP ViT-L/14 features.
    Input dim = 768 (ViT-L/14 patch embed dim).
    Matches the official UniversalFakeDetect fc_weights.pth.
    """
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


@app.on_event("startup")
async def load_model():
    global clip_model, preprocess, classifier

    # ── Load CLIP backbone ───────────────────────────────────────────────────────
    try:
        import clip  # openai-clip package
        clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE)
        clip_model.eval()
        logger.info("CLIP ViT-L/14 backbone loaded ✓")
    except ImportError:
        logger.error(
            "openai-clip not installed. Run: pip install git+https://github.com/openai/CLIP.git"
        )
        # Fallback: use torchvision ViT (lower accuracy but functional)
        from torchvision.models import vit_l_16, ViT_L_16_Weights
        import torchvision.transforms as T
        _vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        _vit.heads = nn.Identity()
        clip_model = _vit.to(DEVICE).eval()
        preprocess = T.Compose([
            T.Resize(512), T.CenterCrop(512), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.warning("Using fallback ViT-L/16 — accuracy will be reduced vs genuine CLIP weights")

    # ── Load linear classifier head ──────────────────────────────────────────────
    classifier = LinearClassifier(in_dim=768).to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading UFD weights from {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.", "").replace("fc.", ""): v for k, v in state.items()}
        # fc_weights.pth stores weight/bias directly
        if "weight" in state and "bias" in state:
            classifier.fc.weight.data = state["weight"]
            classifier.fc.bias.data = state["bias"]
            logger.info("UFD classifier weights loaded ✓")
        else:
            classifier.load_state_dict({"fc.weight": state.get("fc.weight", classifier.fc.weight),
                                        "fc.bias":   state.get("fc.bias",   classifier.fc.bias)})
            logger.info("UFD classifier weights loaded (alt format) ✓")
    else:
        logger.warning(
            f"Weight file not found at {WEIGHTS_PATH}. "
            "Run download_weights.py. Running with RANDOM classifier — results meaningless."
        )

    classifier.eval()
    logger.info(f"UFD ready on {DEVICE}")


def _extract_features(img: Image.Image) -> torch.Tensor:
    """Extract CLIP image features."""
    if hasattr(preprocess, '__call__'):
        tensor = preprocess(img)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(DEVICE)
    else:
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if hasattr(clip_model, 'encode_image'):
            feats = clip_model.encode_image(tensor).float()
        else:
            feats = clip_model(tensor).float()
    return feats


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "UniversalFakeDetect",
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        feats = _extract_features(img)

        with torch.no_grad():
            prob = classifier(feats).item()

        return {
            "model": "UniversalFakeDetect",
            "fake_probability": round(prob, 4),
            "verdict": "fake" if prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"UFD detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

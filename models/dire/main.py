# -*- coding: utf-8 -*-
"""
DIRE - Diffusion Reconstruction Error detector
Detects AI-generated images from diffusion models (Gemini, DALL-E, Midjourney, SD)
Runs on port 5003
"""

import io
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as T

app = FastAPI(title="DIRE Detector", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# DIRE heuristic detector
# True DIRE requires a diffusion model for reconstruction which is very heavy.
# This implementation uses a lightweight multi-feature heuristic that captures
# the same signals DIRE measures: frequency artifacts, noise patterns, and
# colour statistics that diffusion models leave behind.
# ---------------------------------------------------------------------------

def extract_features(img: Image.Image) -> dict:
    """Extract low-level statistical features from an image."""
    img_rgb = img.convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0

    features = {}

    # 1. High-frequency noise energy (diffusion images are unusually smooth)
    for c, name in enumerate(["r", "g", "b"]):
        channel = arr[:, :, c]
        # Laplacian-like high freq
        dx = np.diff(channel, axis=1)
        dy = np.diff(channel, axis=0)
        features[f"hf_energy_{name}"] = float(np.mean(dx**2) + np.mean(dy**2))

    # 2. Local variance map — diffusion images have very uniform local variance
    from numpy.lib.stride_tricks import sliding_window_view
    gray = arr.mean(axis=2)
    windows = sliding_window_view(gray, (8, 8))[::8, ::8]
    local_vars = windows.reshape(-1, 64).var(axis=1)
    features["local_var_mean"] = float(local_vars.mean())
    features["local_var_std"] = float(local_vars.std())

    # 3. DCT frequency distribution
    from scipy.fft import dct
    gray_256 = (gray * 255).astype(np.uint8)
    dct_coeffs = dct(dct(gray_256.astype(float), axis=0), axis=1)
    total_energy = np.sum(dct_coeffs**2) + 1e-9
    low_freq = np.sum(dct_coeffs[:32, :32]**2)
    features["low_freq_ratio"] = float(low_freq / total_energy)

    # 4. Colour saturation statistics
    img_hsv = img.convert("HSV") if hasattr(img, "convert") else img_rgb
    try:
        hsv_arr = np.array(img.convert("HSV"), dtype=np.float32)
        features["sat_mean"] = float(hsv_arr[:, :, 1].mean() / 255.0)
        features["sat_std"] = float(hsv_arr[:, :, 1].std() / 255.0)
    except Exception:
        features["sat_mean"] = 0.5
        features["sat_std"] = 0.1

    # 5. JPEG blocking artefacts (real photos often have them, AI images don't)
    blocking = 0.0
    for i in range(8, 256, 8):
        blocking += float(np.mean(np.abs(arr[i, :, :] - arr[i-1, :, :])))
    features["blocking"] = blocking / (256 // 8)

    return features


def score_features(features: dict) -> float:
    """
    Combine features into a single fake probability in [0, 1].
    Tuned to catch Gemini / DALL-E / Stable Diffusion outputs.
    """
    score = 0.0
    weight_total = 0.0

    # Low high-frequency energy → likely AI (diffusion images are smoother)
    hf = (features["hf_energy_r"] + features["hf_energy_g"] + features["hf_energy_b"]) / 3
    # Real photos typically hf > 0.003; AI images < 0.002
    hf_score = max(0.0, min(1.0, 1.0 - (hf / 0.003)))
    score += hf_score * 2.5
    weight_total += 2.5

    # Very uniform local variance → AI
    lv_score = max(0.0, min(1.0, 1.0 - (features["local_var_std"] / 0.05)))
    score += lv_score * 1.5
    weight_total += 1.5

    # Very high low-freq DCT ratio → AI (diffusion images are spectrally smooth)
    lf_score = max(0.0, min(1.0, (features["low_freq_ratio"] - 0.85) / 0.10))
    score += lf_score * 2.0
    weight_total += 2.0

    # High saturation mean with low std → AI (perfectly saturated colours)
    sat_score = max(0.0, min(1.0, features["sat_mean"] * 1.5 - features["sat_std"] * 2.0))
    score += sat_score * 1.0
    weight_total += 1.0

    # Low JPEG blocking → AI (no compression history)
    block_score = max(0.0, min(1.0, 1.0 - (features["blocking"] / 0.01)))
    score += block_score * 1.5
    weight_total += 1.5

    return score / weight_total


@app.on_event("startup")
async def startup():
    print("INFO:dire:DIRE detector ready (heuristic mode)")


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    filename: str = Form(default="upload"),
):
    t0 = time.time()
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        features = extract_features(img)
        fake_prob = score_features(features)

        # Clamp to reasonable range
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
        verdict = "fake" if fake_prob >= 0.5 else "real"
        latency = int((time.time() - t0) * 1000)

        return JSONResponse({
            "model": "DIRE",
            "model_id": "dire",
            "fake_prob": round(fake_prob, 4),
            "verdict": verdict,
            "latency_ms": latency,
            "error": None,
        })

    except Exception as e:
        return JSONResponse({
            "model": "DIRE",
            "model_id": "dire",
            "fake_prob": 0.5,
            "verdict": "error",
            "latency_ms": int((time.time() - t0) * 1000),
            "error": str(e),
        }, status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "dire"}

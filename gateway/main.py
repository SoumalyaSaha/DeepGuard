"""
FastAPI Gateway — :8000
Routes media to the correct model microservices,
then aggregates results through the meta-learner.

IMAGE DETECTION LOGIC (v4 - Clean & Reliable):
  Stage 1 → DIRE checks the image (fast local model)
            DIRE < 0.3  → REAL, stop here (confident real, no API call needed)
            DIRE >= 0.3 → suspicious → go to Stage 2

  Stage 2 → AI-or-Not API (reliable, handles compressed/webp/rotated images)
            AI-or-Not >= 0.75 → FAKE
            AI-or-Not < 0.75  → REAL
            (if AI-or-Not fails → fallback to NPR)

  Fallback → NPR only used if AI-or-Not API fails/unavailable
"""

import asyncio
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI(title="Deepfake Detection Gateway", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service registry ─────────────────────────────────────────────────────────────
MODEL_SERVICES = {
    "image": [
        {"id": "dire", "url": os.getenv("DIRE_URL", "http://localhost:5003/detect"), "name": "DIRE"},
        {"id": "npr",  "url": os.getenv("NPR_URL",  "http://localhost:5001/detect"), "name": "NPR"},
    ],
    "audio": [
        {"id": "rawnet", "url": os.getenv("RAWNET_URL", "http://localhost:5002/detect"), "name": "RawNet2"},
    ],
    "video": [
        {"id": "crossvit", "url": os.getenv("CROSSVIT_URL", "http://localhost:7001/detect"), "name": "CrossEfficientViT"},
    ],
}

# ── AI-or-Not API ────────────────────────────────────────────────────────────────
AIORNOT_API_KEY = os.getenv("AIORNOT_API_KEY", "")
AIORNOT_URL     = "https://api.aiornot.com/v2/image/sync"

# ── Thresholds ───────────────────────────────────────────────────────────────────
DIRE_SUSPICION_THRESHOLD = 0.3   # DIRE >= this → suspicious → call AI-or-Not
AIORNOT_FAKE_THRESHOLD   = 0.75  # AI-or-Not >= this → FAKE
NPR_FALLBACK_THRESHOLD   = 0.90  # NPR fallback (only if AI-or-Not fails): needs very high confidence

TIMEOUT = httpx.Timeout(60.0)


# ── Helper: call a local model ───────────────────────────────────────────────────
async def call_model(client: httpx.AsyncClient, service: dict, file_bytes: bytes, filename: str) -> dict:
    try:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        resp  = await client.post(service["url"], files=files, timeout=TIMEOUT)
        resp.raise_for_status()
        data  = resp.json()
        return {
            "model":      service["name"],
            "model_id":   service["id"],
            "fake_prob":  float(data.get("fake_probability", data.get("fake_prob", 0.5))),
            "verdict":    data.get("verdict", "unknown"),
            "latency_ms": data.get("latency_ms", None),
            "error":      None,
        }
    except Exception as e:
        logger.warning(f"Model {service['name']} failed: {e}")
        return {
            "model":      service["name"],
            "model_id":   service["id"],
            "fake_prob":  0.5,
            "verdict":    "error",
            "latency_ms": None,
            "error":      str(e),
        }


# ── Helper: call AI-or-Not API ───────────────────────────────────────────────────
async def call_aiornot(client: httpx.AsyncClient, file_bytes: bytes) -> dict:
    if not AIORNOT_API_KEY:
        return {
            "model":      "AI-or-Not",
            "model_id":   "aiornot",
            "fake_prob":  0.5,
            "verdict":    "error",
            "latency_ms": None,
            "error":      "AIORNOT_API_KEY not set",
        }
    try:
        import time
        t0      = time.time()
        headers = {"Authorization": f"Bearer {AIORNOT_API_KEY}"}
        # Send as multipart file upload (correct format for v2 API)
        files = {"image": ("image.jpg", file_bytes, "image/jpeg")}
        params = {"only": ["ai_generated"]}  # only run ai_generated to save credits
        resp = await client.post(AIORNOT_URL, headers=headers, files=files, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data        = resp.json()
        report      = data.get("report", {})
        ai_generated = report.get("ai_generated", {})
        # Response: report.ai_generated.verdict = "ai" | "human" | "unknown"
        #           report.ai_generated.ai.confidence = 0.95
        verdict_raw = ai_generated.get("verdict", "unknown")
        ai_score    = float(ai_generated.get("ai", {}).get("confidence", 0.5))
        verdict     = "fake" if verdict_raw == "ai" else "real"
        latency     = int((time.time() - t0) * 1000)
        logger.info(f"AI-or-Not: ai_score={ai_score:.3f} verdict={verdict}")
        return {
            "model":      "AI-or-Not",
            "model_id":   "aiornot",
            "fake_prob":  round(ai_score, 4),
            "verdict":    verdict,
            "latency_ms": latency,
            "error":      None,
        }
    except Exception as e:
        logger.warning(f"AI-or-Not failed: {e}")
        return {
            "model":      "AI-or-Not",
            "model_id":   "aiornot",
            "fake_prob":  0.5,
            "verdict":    "error",
            "latency_ms": None,
            "error":      str(e),
        }


# ── Helper: build result dict ────────────────────────────────────────────────────
def _result(verdict: str, fake_prob: float, path: str, raw_results: list) -> dict:
    confidence = round(fake_prob if verdict == "fake" else (1 - fake_prob), 4)
    return {
        "verdict":          verdict,
        "fake_probability": round(fake_prob, 4),
        "confidence":       confidence,
        "detection_path":   path,
        "raw_results":      raw_results,
    }


# ── Main image detection (v4) ────────────────────────────────────────────────────
async def detect_image_cascaded(file_bytes: bytes, filename: str) -> dict:
    """
    Stage 1: DIRE — fast local check
      → Clear real (< 0.3): return REAL immediately, no API call
      → Suspicious (>= 0.3): go to Stage 2

    Stage 2: AI-or-Not — reliable cloud API
      → >= 0.75: FAKE
      → < 0.75:  REAL
      → error:   fallback to NPR

    Fallback: NPR — only if AI-or-Not fails
      → needs >= 0.90 to call FAKE (very strict to avoid false positives)
      → otherwise REAL
    """
    image_services = {s["id"]: s for s in MODEL_SERVICES["image"]}
    raw_results    = []

    async with httpx.AsyncClient() as client:

        # ── Stage 1: DIRE ────────────────────────────────────────────────────────
        dire_service = image_services.get("dire")
        if not dire_service:
            raise HTTPException(500, "DIRE model not configured")

        dire_result = await call_model(client, dire_service, file_bytes, filename)
        raw_results.append(dire_result)
        logger.info(f"DIRE: {dire_result['fake_prob']:.3f}")

        # DIRE errored → skip to AI-or-Not directly
        if dire_result["verdict"] == "error":
            logger.warning("DIRE failed, going straight to AI-or-Not")
            aiornot_result = await call_aiornot(client, file_bytes)
            raw_results.append(aiornot_result)
            if aiornot_result["verdict"] == "error":
                # Last resort: NPR
                npr_service = image_services.get("npr")
                if npr_service:
                    npr_result = await call_model(client, npr_service, file_bytes, filename)
                    raw_results.append(npr_result)
                    if npr_result["verdict"] != "error":
                        verdict = "fake" if npr_result["fake_prob"] >= NPR_FALLBACK_THRESHOLD else "real"
                        return _result(verdict, npr_result["fake_prob"], "fallback_npr_only", raw_results)
                raise HTTPException(502, "All detection services failed.")
            verdict = "fake" if aiornot_result["fake_prob"] >= AIORNOT_FAKE_THRESHOLD else "real"
            return _result(verdict, aiornot_result["fake_prob"], "fallback_aiornot_dire_error", raw_results)

        # ── DIRE clear → REAL, no API call needed ────────────────────────────────
        if dire_result["fake_prob"] < DIRE_SUSPICION_THRESHOLD:
            logger.info(f"DIRE clear ({dire_result['fake_prob']:.3f}), verdict: REAL")
            return _result("real", dire_result["fake_prob"], "dire_clear_real", raw_results)

        # ── Stage 2: DIRE suspicious → call AI-or-Not ────────────────────────────
        logger.info(f"DIRE suspicious ({dire_result['fake_prob']:.3f}), calling AI-or-Not...")
        aiornot_result = await call_aiornot(client, file_bytes)
        raw_results.append(aiornot_result)

        # AI-or-Not worked → trust it
        if aiornot_result["verdict"] != "error":
            if aiornot_result["fake_prob"] >= AIORNOT_FAKE_THRESHOLD:
                avg = (dire_result["fake_prob"] + aiornot_result["fake_prob"]) / 2
                return _result("fake", avg, "dire_suspicious_aiornot_confirmed_fake", raw_results)
            else:
                avg = (dire_result["fake_prob"] + aiornot_result["fake_prob"]) / 2
                return _result("real", avg, "dire_suspicious_aiornot_says_real", raw_results)

        # ── Fallback: AI-or-Not failed → use NPR with very strict threshold ──────
        logger.warning("AI-or-Not failed, falling back to NPR with strict threshold")
        npr_service = image_services.get("npr")
        if not npr_service:
            # No NPR either → be conservative, return REAL
            return _result("real", dire_result["fake_prob"], "all_failed_default_real", raw_results)

        npr_result = await call_model(client, npr_service, file_bytes, filename)
        raw_results.append(npr_result)

        if npr_result["verdict"] == "error":
            return _result("real", dire_result["fake_prob"], "all_failed_default_real", raw_results)

        # NPR fallback: needs 90%+ AND DIRE also suspicious to call FAKE
        if npr_result["fake_prob"] >= NPR_FALLBACK_THRESHOLD and dire_result["fake_prob"] >= 0.5:
            avg = (dire_result["fake_prob"] + npr_result["fake_prob"]) / 2
            return _result("fake", avg, "fallback_dire_and_npr_both_agree_fake", raw_results)
        else:
            avg = (dire_result["fake_prob"] + npr_result["fake_prob"]) / 2
            return _result("real", avg, "fallback_npr_not_confident_enough_real", raw_results)


# ── Meta-learner (audio/video) ───────────────────────────────────────────────────
def meta_learner(predictions: list[dict], strategy: str = "average") -> dict:
    if not predictions:
        raise ValueError("No predictions to combine")
    fake_probs = [p["fake_prob"] for p in predictions]
    if strategy == "voting":
        votes_fake = sum(1 for p in predictions if p["verdict"] == "fake")
        final_prob = votes_fake / len(predictions)
    else:
        final_prob = sum(fake_probs) / len(fake_probs)
    verdict    = "fake" if final_prob >= 0.5 else "real"
    confidence = final_prob if verdict == "fake" else (1 - final_prob)
    return {
        "verdict":          verdict,
        "fake_probability": round(final_prob, 4),
        "confidence":       round(confidence, 4),
    }


# ── Routes ───────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(
    file:       UploadFile    = File(...),
    media_type: str           = Form("image"),
    models:     Optional[str] = Form(None),
    strategy:   str           = Form("average"),
):
    if media_type not in MODEL_SERVICES:
        raise HTTPException(400, f"media_type must be one of {list(MODEL_SERVICES)}")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "Empty file")

    # Images → cascaded logic
    if media_type == "image":
        result = await detect_image_cascaded(file_bytes, file.filename or "upload")
        return JSONResponse({
            "filename":         file.filename,
            "media_type":       media_type,
            "strategy":         "cascaded_v4",
            "detection_path":   result["detection_path"],
            "verdict":          result["verdict"],
            "fake_probability": result["fake_probability"],
            "confidence":       result["confidence"],
            "model_results":    result["raw_results"],
        })

    # Audio / Video → average across models
    all_services = MODEL_SERVICES[media_type]
    if models:
        requested = {m.strip() for m in models.split(",")}
        services  = [s for s in all_services if s["id"] in requested]
        if not services:
            raise HTTPException(400, f"No matching models. Available: {[s['id'] for s in all_services]}")
    else:
        services = all_services

    async with httpx.AsyncClient() as client:
        tasks       = [call_model(client, svc, file_bytes, file.filename or "upload") for svc in services]
        raw_results = await asyncio.gather(*tasks)

    valid = [r for r in raw_results if r["verdict"] != "error"]
    if not valid:
        raise HTTPException(502, "All model services failed.")

    meta = meta_learner(valid, strategy=strategy)
    return JSONResponse({
        "filename":         file.filename,
        "media_type":       media_type,
        "strategy":         strategy,
        "verdict":          meta["verdict"],
        "fake_probability": meta["fake_probability"],
        "confidence":       meta["confidence"],
        "model_results":    list(raw_results),
    })


@app.get("/api/models")
async def list_models():
    return {
        k: [{"id": s["id"], "name": s["name"], "url": s["url"]} for s in v]
        for k, v in MODEL_SERVICES.items()
    }

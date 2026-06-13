# DeepGuard — Deepfake Detection Gateway

![Version](https://img.shields.io/badge/version-0.1.0-blue) ![OAS](https://img.shields.io/badge/OAS-3.1-green) ![Status](https://img.shields.io/badge/status-active-brightgreen) ![Models](https://img.shields.io/badge/models-3%20active-orange)

> **Multi-modal deepfake detection system** that routes images, audio, and video to specialist AI models and returns ensemble verdicts via a unified REST API.

---

## What is DeepGuard?

DeepGuard is a microservice-based deepfake detection platform that combines multiple state-of-the-art detection models into a single API gateway. Upload any media file — image, audio, or video — and DeepGuard runs it through specialist models in parallel, then aggregates results using configurable ensemble strategies.

Built because no single model catches all deepfakes. Ensemble inference does.

---

## How It Works

```
Media File (image / audio / video)
        │
        ▼
┌─────────────────────────────┐
│   FastAPI Gateway (:8000)   │  ← OAS 3.1, auto-docs at /docs
│   POST /api/detect          │
└────────────┬────────────────┘
             │ routes by media type
     ┌───────┼───────┬───────┐
     ▼       ▼       ▼       ▼
  [Image]  [Image] [Image] [Video]
   NPR    UnivFD   DIRE   CrossViT
             │
     ensemble strategy
   (average / vote / stack)
             │
             ▼
    unified JSON verdict
```

---

## Active Models

| Model | Media Type | What It Detects |
|---|---|---|
| **NPR** (Noise Pattern Recognition) | Image | GAN-generated images via noise artifacts |
| **UniversalFakeDetect** | Image | Broad-spectrum AI-generated image detection |
| **DIRE** (Diffusion Reconstruction Error) | Image | Diffusion model images (Stable Diffusion, DALL-E, Midjourney) |
| **CrossEfficientViT** | Video | Face-swap and video deepfakes via Vision Transformer |
| RawNet2 *(coming soon)* | Audio | AI-cloned voice and audio deepfakes |

---

## API

### `POST /api/detect`

Upload any media file and choose an ensemble strategy.

**Request** — `multipart/form-data`
```
file       : media file (image/audio/video)
strategy   : "average" | "vote" | "stack"
```

**Response**
```json
{
  "filename": "sample.jpeg",
  "media_type": "image",
  "strategy": "average",
  "verdict": "real",
  "fake_probability": 0.2548,
  "confidence": 0.4904,
  "latency_ms": 947,
  "model_results": [
    {
      "model": "UniversalFakeDetect",
      "fake_probability": 0.0007,
      "verdict": "real",
      "latency_ms": 186
    },
    {
      "model": "CrossEfficientViT",
      "fake_probability": 0.5089,
      "verdict": "fake",
      "latency_ms": 144
    }
  ]
}
```

### Other Endpoints
```
GET /health        — service health check
GET /api/models    — list loaded models and status
```

Interactive API docs available at `http://localhost:8000/docs` when running locally.

---

## Ensemble Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `average` | Mean of all model probabilities | General use |
| `vote` | Majority verdict wins | When models disagree |
| `stack` | Weighted combination | Tuned for specific media types |

---

## Setup

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU recommended (CPU inference supported but slow)

### Run

```bash
git clone https://github.com/SoumalyaSaha/DeepGuard.git
cd DeepGuard
docker-compose up --build
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Gateway | FastAPI (Python) |
| Inference | PyTorch |
| Models | NPR, UniversalFakeDetect, CrossEfficientViT |
| API Spec | OpenAPI 3.1 (OAS 3.1) |
| Deployment | Docker Compose |

---

## Roadmap

- [x] Image deepfake detection (NPR + UniversalFakeDetect + DIRE)
- [x] Video deepfake detection (CrossEfficientViT)
- [x] Ensemble inference gateway (average, vote, stack)
- [x] FastAPI gateway with OAS 3.1 docs
- [ ] Audio deepfake detection (RawNet2)
- [ ] Frontend UI for non-technical users
- [ ] Benchmark results on standard datasets (FaceForensics++, DFDC)
- [ ] Cloud deployment

---

## Why Ensemble?

Single models have blind spots. In testing, **UniversalFakeDetect and CrossEfficientViT disagreed on edge cases** — UFD returned 0.07% fake probability while CrossViT returned 50.89% on the same image. The ensemble averaged these to 25.48% (verdict: real), but the disagreement itself is surfaced in the response so downstream systems can handle uncertainty explicitly.

---

## Author

**Soumalya Saha** — [GitHub](https://github.com/SoumalyaSaha)

Also built: [ArtifactX](https://github.com/SoumalyaSaha/ArtifactX) — AI-powered museum guide with multilingual artifact identification

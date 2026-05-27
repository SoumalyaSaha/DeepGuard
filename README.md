# 🛡️ DeepGuard — AI-Powered Deepfake Detection System

> **"Truth in every pixel."**

DeepGuard is a multi-modal deepfake detection platform that analyzes **video**, **audio**, and **images** using state-of-the-art AI models with explainable heatmap visualizations — no black-box outputs, no ML expertise required.

Built for the **IdeaTeX Heritage Hackathon | May 2026**.

---

## 🚨 The Problem

Deepfakes are eroding trust in digital media at an alarming rate:

- **500%** rise in deepfakes since 2019
- **96%** of deepfake content is non-consensual
- **$25B** projected fraud losses by 2027

From political manipulation in elections to identity fraud and fake news — the need for accessible, accurate detection has never been greater.

---

## ✅ Why DeepGuard?

| Problem with existing tools | DeepGuard's approach |
|---|---|
| Fragmented — separate tools per media type | One unified platform for video, audio & image |
| Black-box results with no explanation | Grad-CAM heatmaps highlight manipulated regions |
| Requires ML expertise | Intuitive web app + REST API for anyone |
| Batch-only, no real-time integration | Live API endpoints for platform integration |

---

## 🎯 Who Is It For?

- 📰 **Journalists & fact-checkers** — verify footage before publishing
- 🏛️ **Law enforcement** — authenticate digital evidence
- 📱 **Social media platforms** — auto-flag suspicious uploads
- 🏦 **Finance & HR** — KYC and video interview fraud prevention
- 🗳️ **Government** — election integrity monitoring
- 👤 **Everyday users** — personal media verification

---

## 🔍 How It Works

```
Upload media  →  Pre-process & extract features  →  AI model analysis  →  Confidence score + Heatmap
```

1. **Upload** — Submit a video, audio file, or image via the web app or API
2. **Analyze** — Multi-model forensic detection pipeline processes the file
3. **Score** — Receive a 0–100% authenticity confidence score
4. **Highlight** — Grad-CAM heatmap pinpoints exactly which regions were manipulated

### Supported Formats
- 🎥 Video: `.mp4`, `.mov`
- 🎵 Audio: `.wav`, `.mp3`
- 🖼️ Image: `.jpg`, `.png`, `.webp`

---

## 🤖 AI Models & Accuracy

| Modality | Model | Dataset | Accuracy |
|---|---|---|---|
| 🖼️ Image | EfficientNet-B4 | FaceForensics++ | **96.3%** |
| 🎥 Video | TimeSformer | Temporal frame analysis | **94.1%** |
| 🎵 Audio | Wav2Vec2 | Spectral + prosody anomaly | **91.7%** |

All detections include **Grad-CAM visualizations** so users see exactly what triggered the result.

---

## 🏗️ Tech Stack

```
Frontend       React.js · TailwindCSS · Grad-CAM Heatmap Viewer
Backend API    FastAPI (Python) · REST · JWT Auth · Docker
AI / ML Layer  EfficientNet · TimeSformer · Wav2Vec2
Infrastructure AWS S3 · Hugging Face Hub · CI/CD Pipeline
```

---

## 🗺️ Roadmap

### ✅ Phase 1 — Now (MVP)
- Web app with image, video, and audio detection
- Confidence score + Grad-CAM heatmap output
- REST API with JWT authentication

### 🔜 Phase 2 — 3 Months
- Browser extension for in-page detection
- API marketplace listing
- Batch processing tier

### 🔮 Phase 3 — 6 Months
- Real-time video stream analysis
- Enterprise dashboard
- Mobile SDK

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-team/deepguard.git
cd deepguard

# Install backend dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app.main:app --reload

# In a separate terminal, start the frontend
cd frontend
npm install
npm run dev
```

---

## 🔌 API Usage

```bash
# Analyze an image
curl -X POST https://api.deepguard.ai/detect \
  -H "Authorization: Bearer <your_token>" \
  -F "file=@sample.jpg" \
  -F "type=image"

# Response
{
  "authenticity_score": 23.4,
  "verdict": "LIKELY FAKE",
  "heatmap_url": "https://...",
  "model_used": "EfficientNet-B4"
}
```

---

## 📎 Links

| | |
|---|---|
| 🌐 Live Demo | file:///C:/Users/soumalyasaha/Downloads/DeepGuard_Prototype.html |
| 💻 GitHub | [github.com/your-team/deepguard](https://github.com/your-team/deepguard) |
| 📧 Contact | team@deepguard.ai |



---

<p align="center">
  Built with passion for truth · IdeaTeX Heritage Hackathon · May 2026
</p>





"""
RawNet2 — Audio deepfake / voice-spoofing detector
Architecture from:
  "Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification
   Using Raw Waveforms" (Jung et al.) — adapted for ASVspoof anti-spoofing.

Official pretrained weights: ASVspoof 2021 LA track
  https://github.com/asvspoof-challenge/2021  (or via download_weights.py)

Weight file expected: weights/rawnet2.pth
"""

import io, time, logging, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rawnet")

app = FastAPI(title="RawNet2 Audio Deepfake Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "../../weights/rawnet2.pth")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64000  # 4 seconds at 16 kHz (matches ASVspoof training)


# ── RawNet2 building blocks ─────────────────────────────────────────────────────

class SincConv(nn.Module):
    """
    Sinc-based convolution layer — learnable band-pass filters operating
    directly on the raw waveform.  Matches the original RawNet2 implementation.
    """
    def __init__(self, out_channels=128, kernel_size=1024, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate

        # Initialise cutoff frequencies from mel-scale
        low_hz = 30.0
        high_hz = sample_rate / 2 - (low_hz + 1)
        mel = np.linspace(self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1)
        hz = self._mel2hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * np.pi * torch.arange(-n, 0).view(1, -1) / sample_rate
        self.window_ = torch.hamming_window(self.kernel_size)

    @staticmethod
    def _hz2mel(hz): return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def _mel2hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)

        low  = 50 + torch.abs(self.low_hz_)
        high = torch.clamp(low + torch.abs(self.band_hz_), 50, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low  = torch.matmul(low,  self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, self.filters, stride=1, padding=self.kernel_size // 2, groups=1)


class ResBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1, bias=False)
        self.mp    = nn.MaxPool1d(3)
        self.fms   = FMS(nb_filts[1])
        if nb_filts[0] != nb_filts[1]:
            self.downsample = nn.Sequential(
                nn.Conv1d(nb_filts[0], nb_filts[1], 1, bias=False),
                nn.BatchNorm1d(nb_filts[1]),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        if not self.first:
            x = self.bn1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.3)
        x = self.conv2(x)
        x = self.fms(x)
        if self.downsample:
            identity = self.downsample(identity)
        x = x + identity
        return self.mp(x)


class FMS(nn.Module):
    """Feature Map Scaling — channel-wise attention."""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(-1)
        s = self.sig(self.fc(s)).unsqueeze(-1)
        return x * s + s


class RawNet2(nn.Module):
    """
    RawNet2 as used in ASVspoof 2021.
    Output: sigmoid probability that input audio is spoofed/fake.
    """
    def __init__(self, d_args=None):
        super().__init__()
        if d_args is None:
            d_args = {
                "nb_samp": MAX_SAMPLES,
                "first_conv": 1024,
                "in_channels": 1,
                "filts": [128, [128, 128], [128, 256], [256, 256]],
                "blocks": [2, 4],
                "nb_fc_node": 1024,
                "gru_node": 1024,
                "nb_gru_layer": 3,
                "nb_classes": 2,
            }

        self.sinc = SincConv(d_args["filts"][0], d_args["first_conv"])
        self.first_bn = nn.BatchNorm1d(d_args["filts"][0])

        self.blocks = nn.ModuleList()
        for i in range(d_args["blocks"][0]):
            self.blocks.append(ResBlock([d_args["filts"][1][0], d_args["filts"][1][1]], first=(i == 0)))
        for i in range(d_args["blocks"][1]):
            self.blocks.append(ResBlock([d_args["filts"][2 if i == 0 else 3][0],
                                         d_args["filts"][2 if i == 0 else 3][1]]))

        self.bn_before_gru = nn.BatchNorm1d(d_args["filts"][3][1])
        self.gru = nn.GRU(d_args["filts"][3][1], d_args["gru_node"],
                          d_args["nb_gru_layer"], batch_first=True)
        self.fc1 = nn.Linear(d_args["gru_node"], d_args["nb_fc_node"])
        self.fc2 = nn.Linear(d_args["nb_fc_node"], d_args["nb_classes"])

    def forward(self, x):
        x = self.sinc(x)
        x = F.leaky_relu(self.first_bn(torch.abs(x)), 0.3)
        for block in self.blocks:
            x = block(x)
        x = self.bn_before_gru(x)
        x = F.leaky_relu(x, 0.3)
        x = x.permute(0, 2, 1)
        _, x = self.gru(x)
        x = x[-1]
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.fc2(x)
        return x  # logits — [batch, 2]  (0=genuine, 1=spoof)


model: RawNet2 = None


@app.on_event("startup")
async def load_model():
    global model
    model = RawNet2().to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading RawNet2 weights from {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("RawNet2 weights loaded ✓")
    else:
        logger.warning(
            f"Weight file not found at {WEIGHTS_PATH}. "
            "Run download_weights.py. Running with RANDOM weights."
        )

    model.eval()
    logger.info(f"RawNet2 ready on {DEVICE}")


def _load_audio(data: bytes) -> torch.Tensor:
    """Load audio bytes → mono 16 kHz waveform tensor [1, 1, T]."""
    import soundfile as sf
    import librosa

    with io.BytesIO(data) as buf:
        try:
            wav, sr = sf.read(buf, dtype="float32", always_2d=False)
        except Exception:
            buf.seek(0)
            wav, sr = librosa.load(buf, sr=None, mono=True)

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)  # stereo → mono

    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad or crop to MAX_SAMPLES
    if len(wav) < MAX_SAMPLES:
        wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
    else:
        wav = wav[:MAX_SAMPLES]

    return torch.FloatTensor(wav).unsqueeze(0).unsqueeze(0)  # [1, 1, T]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "RawNet2",
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        data = await file.read()
        wav = _load_audio(data).to(DEVICE)

        with torch.no_grad():
            logits = model(wav)  # [1, 2]
            probs = torch.softmax(logits, dim=-1)
            # Class 1 = spoof/fake in ASVspoof convention
            fake_prob = probs[0, 1].item()

        return {
            "model": "RawNet2",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"RawNet2 detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#!/usr/bin/env python3
"""
download_weights.py — Fetch genuine pretrained weights for all DeepGuard models.

Run from the DeepGuard root directory:
    python download_weights.py

What this downloads
───────────────────
  weights/npr.pth       CNNDetection ResNet-50 (trained on ProGAN, generalises broadly)
                        Source: github.com/peterwang512/CNNDetection
                        ~100 MB

  weights/ufd.pth       UniversalFakeDetect linear classifier head (CLIP ViT-L/14)
                        Source: github.com/WisconsinAIVision/UniversalFakeDetect
                        ~4 KB  (CLIP backbone auto-downloaded by openai-clip)

  weights/rawnet2.pth   RawNet2 anti-spoofing (ASVspoof 2021 LA track)
                        Source: asvspoof.org / Zenodo
                        ~60 MB

  weights/crossvit.pth  CrossEfficientViT (FaceForensics++ trained)
                        Source: github.com/davide-coccomini/...
                        ~20 MB
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


# ── Download registry ────────────────────────────────────────────────────────────
# Each entry: (filename, url, expected_sha256_prefix_or_None)
#
# NOTE: Some repos require accepting a licence before downloading.
#       Where direct URLs are blocked, this script prints manual instructions.

REGISTRY = [
    {
        "name": "NPR (CNNDetection)",
        "file": "npr.pth",
        # Direct link from the CNNDetection GitHub release
        "url": "https://github.com/peterwang512/CNNDetection/releases/download/v1.0/blur_jpg_prob0.5.pth",
        "sha256": None,  # verify manually if needed
        "alt": (
            "Manual download:\n"
            "  1. Visit https://github.com/peterwang512/CNNDetection\n"
            "  2. Download blur_jpg_prob0.5.pth from the Releases page\n"
            "     or from the Google Drive link in the README\n"
            "  3. Save as weights/npr.pth"
        ),
    },
    {
        "name": "UFD (UniversalFakeDetect classifier head)",
        "file": "ufd.pth",
        # Official fc_weights.pth from Wisconsin AI Vision Lab
        "url": "https://github.com/WisconsinAIVision/UniversalFakeDetect/releases/download/v0.1/fc_weights.pth",
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Visit https://github.com/WisconsinAIVision/UniversalFakeDetect\n"
            "  2. Download fc_weights.pth from the Releases page\n"
            "  3. Save as weights/ufd.pth\n"
            "  Also install CLIP: pip install git+https://github.com/openai/CLIP.git"
        ),
    },
    {
        "name": "RawNet2 (ASVspoof 2021)",
        "file": "rawnet2.pth",
        # Zenodo deposit from ASVspoof organisers
        "url": "https://zenodo.org/record/6456915/files/RawNet2_best_model.pth",
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Visit https://zenodo.org/record/6456915\n"
            "  2. Download RawNet2_best_model.pth\n"
            "  3. Save as weights/rawnet2.pth\n"
            "  Alternatively: https://github.com/asvspoof-challenge/2021 → model zoo"
        ),
    },
    {
        "name": "CrossEfficientViT (FaceForensics++)",
        "file": "crossvit.pth",
        # GitHub release from Coccomini et al.
        "url": (
            "https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformer-"
            "for-Video-Deepfake-Detection/releases/download/v1.0/cross-efficient-vit.pth"
        ),
        "sha256": None,
        "alt": (
            "Manual download:\n"
            "  1. Visit https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformer-for-Video-Deepfake-Detection\n"
            "  2. Download cross-efficient-vit.pth from Releases (or the Google Drive link in README)\n"
            "  3. Save as weights/crossvit.pth"
        ),
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _sizeof_fmt(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _reporthook(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.0f}%  {_sizeof_fmt(downloaded)}/{_sizeof_fmt(total_size)}", end="", flush=True)
    else:
        print(f"\r  Downloaded {_sizeof_fmt(downloaded)}", end="", flush=True)


def _sha256(path: Path, prefix_len: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_len]


def download(entry: dict) -> bool:
    dest = WEIGHTS_DIR / entry["file"]

    if dest.exists():
        print(f"  ✓ {dest} already exists — skipping")
        return True

    print(f"\nDownloading {entry['name']} …")
    print(f"  URL: {entry['url']}")

    try:
        urllib.request.urlretrieve(entry["url"], dest, reporthook=_reporthook)
        print()  # newline after progress bar

        if entry["sha256"]:
            actual = _sha256(dest)
            if not actual.startswith(entry["sha256"]):
                print(f"  ⚠ SHA256 mismatch! Expected {entry['sha256']}, got {actual}")
                print("     The file may be corrupt or the URL outdated.")
            else:
                print(f"  ✓ SHA256 OK ({actual})")

        size = dest.stat().st_size
        print(f"  ✓ Saved to {dest}  ({_sizeof_fmt(size)})")
        return True

    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if dest.exists():
            dest.unlink()  # remove partial file
        print(f"\n  {entry['alt']}\n")
        return False


# ── Extras ───────────────────────────────────────────────────────────────────────

def install_clip():
    """Install openai-clip if not present (needed for UFD)."""
    try:
        import clip
        print("  ✓ openai-clip already installed")
    except ImportError:
        print("\nInstalling openai-clip (required for UFD model)…")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "git+https://github.com/openai/CLIP.git"
        ])
        print("  ✓ openai-clip installed")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DeepGuard — Weight Downloader")
    print("=" * 60)

    results = {}
    for entry in REGISTRY:
        ok = download(entry)
        results[entry["file"]] = ok

    install_clip()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_ok = True
    for fname, ok in results.items():
        status = "✓" if ok else "✗ MANUAL ACTION NEEDED"
        print(f"  {status}  weights/{fname}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All weights ready! Run ./start.sh or docker compose up --build")
    else:
        print(
            "\n  Some weights need manual download (see instructions above).\n"
            "  After placing the files in weights/, re-run this script to verify."
        )
    print()


if __name__ == "__main__":
    main()

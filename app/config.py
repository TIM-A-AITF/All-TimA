"""Konfigurasi terpusat. Semua nilai dibaca dari environment (lihat .env.example).

Bawaan path relatif terhadap root repo (folder induk `app/`), sehingga perintah yang
sama jalan di laptop maupun di container (/app). Tidak ada path absolut lokal.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default.resolve()


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# ---------- Struktur repo ----------
BASE_DIR = _path("APP_BASE_DIR", Path(__file__).resolve().parent.parent)
EXTRACTOR_DIR = _path("EXTRACTOR_DIR", BASE_DIR / "Ekstraktor-TOR-RAB")
CHECKLIST_DIR = _path("CHECKLIST_DIR", BASE_DIR / "Checklist-Rule-TOR-RAB")

# Diteruskan ke checklist_common (dibaca saat modul checklist di-import)
os.environ.setdefault("CHECKLIST_RULES_DIR", str(CHECKLIST_DIR / "rules"))
os.environ.setdefault("CHECKLIST_KATALOG_DIR", str(CHECKLIST_DIR / "katalog_sbm"))
os.environ.setdefault("CHECKLIST_WILAYAH_CSV", str(CHECKLIST_DIR / "wilayah_indonesia.csv"))

# ---------- Data run ----------
WORKSPACE = _path("TOR_RAB_WORKSPACE", BASE_DIR / "workspace")
RUNS_DIR = WORKSPACE / "runs"
os.environ.setdefault("CHECKLIST_CACHE_DIR", str(WORKSPACE / "cache"))

# ---------- Ekstraksi ----------
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or None      # None = cari di PATH (di Docker: /usr/bin/tesseract)
PAIR_MIN_SCORE = float(os.environ.get("PAIR_MIN_SCORE", "0.35"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "50"))
KEEP_WORK_DIR = _bool("KEEP_WORK_DIR", False)

# ---------- VLM (Qwen3-VL-2B, VPS terpisah) ----------
VLM_ENDPOINT_URL = os.environ.get("VLM_ENDPOINT_URL") or None  # kosong = deskripsi gambar dilewati
VLM_API_STYLE = os.environ.get("VLM_API_STYLE", "openai")     # openai (vLLM/OpenAI-compatible) | custom
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLM_TIMEOUT_SEC = float(os.environ.get("VLM_TIMEOUT_SEC", "180"))
VLM_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "400"))
VLM_MIN_AREA = float(os.environ.get("VLM_MIN_AREA", "2500"))

# ---------- SFT (Qwen3-4B-Instruct-2507 + LoRA, VPS terpisah) ----------
# Dibaca langsung oleh checklist_tor.py: SFT_ENDPOINT_URL, SFT_API_KEY, SFT_MODEL_VERSION, SFT_TIMEOUT_SEC

# ---------- API ----------
PIPELINE_WORKERS = int(os.environ.get("PIPELINE_WORKERS", "1"))   # >1 tidak disarankan: CFG ekstraktor bersifat global


_ready = False


def setup() -> None:
    """Tambahkan folder ekstraktor & checklist ke sys.path dan terapkan CFG ekstraktor. Aman dipanggil berulang."""
    global _ready
    if _ready:
        return
    for p in (EXTRACTOR_DIR, CHECKLIST_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["CHECKLIST_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    import tor_extractor as tor
    tor.CFG["tesseract_cmd"] = TESSERACT_CMD
    tor.CFG["debug"] = False
    _ready = True

"""Klien VLM jarak jauh (Qwen3-VL-2B di VPS terpisah) — PLACEHOLDER endpoint.

Pengganti `tor_extractor.FigureVLM` (yang memuat model lokal di GPU). Objek ini cukup
punya `.describe(img) -> str` dan `.model_id`, sehingga bisa langsung dipakai
`tor_extractor.describe_figures(doc, pdf_path, vlm)`.

Dua gaya API (env VLM_API_STYLE):
  openai : POST {VLM_ENDPOINT_URL}  (mis. http://<IP_VPS_VLM>:8000/v1/chat/completions, server vLLM)
           body = chat completion dengan image_url data:image/png;base64,...
  custom : POST {VLM_ENDPOINT_URL}  body = {"model", "prompt", "image_b64", "max_new_tokens"}
           respons = {"text": "..."}  (sesuaikan dengan API yang dibangun di VPS)
Keluaran yang diharapkan tetap format DESC_PROMPT ekstraktor: JENIS / TEKS / DESKRIPSI.
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

from . import config


class VLMUnavailable(RuntimeError):
    pass


class RemoteFigureVLM:
    def __init__(self, url: str | None = None, style: str | None = None, model_id: str | None = None,
                 api_key: str | None = None, timeout: float | None = None, prompt: str | None = None,
                 max_new_tokens: int | None = None, max_pixels: int = 1_000_000):
        import tor_extractor as tor
        self.url = url or config.VLM_ENDPOINT_URL
        self.style = (style or config.VLM_API_STYLE).lower()
        self.model_id = model_id or config.VLM_MODEL_ID
        self.api_key = api_key if api_key is not None else config.VLM_API_KEY
        self.timeout = timeout or config.VLM_TIMEOUT_SEC
        self.prompt = prompt or tor.DESC_PROMPT
        self.max_new_tokens = max_new_tokens or config.VLM_MAX_NEW_TOKENS
        self.max_pixels = max_pixels
        self.last: dict = {}
        self._down: str | None = None          # circuit breaker: setelah gagal koneksi, panggilan berikut langsung gagal

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _png_b64(self, img) -> str:
        w, h = img.size
        if w * h > self.max_pixels:
            k = (self.max_pixels / (w * h)) ** 0.5
            img = img.resize((max(28, int(w * k)), max(28, int(h * k))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _body(self, b64: str) -> dict:
        if self.style == "openai":
            return {"model": self.model_id, "temperature": 0, "max_tokens": self.max_new_tokens,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": self.prompt}]}]}
        return {"model": self.model_id, "prompt": self.prompt, "image_b64": b64, "max_new_tokens": self.max_new_tokens}

    @staticmethod
    def _text(data: dict, style: str) -> str:
        if style == "openai":
            return data["choices"][0]["message"]["content"]
        return data.get("text") or data.get("output") or ""

    def describe(self, img, max_new_tokens: int | None = None) -> str:
        if not self.url:
            raise VLMUnavailable("VLM_ENDPOINT_URL kosong")
        if self._down:
            raise VLMUnavailable(self._down)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = self._body(self._png_b64(img))
        if max_new_tokens:
            body["max_tokens" if self.style == "openai" else "max_new_tokens"] = max_new_tokens
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:            # server hidup tetapi menolak: jangan putus semua gambar
            raise VLMUnavailable(f"HTTP {e.code} dari endpoint VLM") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self._down = f"endpoint VLM tidak dapat dihubungi ({type(e).__name__})"
            raise VLMUnavailable(self._down) from e
        self.last = {"sec": round(time.time() - t0, 2)}
        return str(self._text(data, self.style)).strip()

    def close(self):
        pass

# -*- coding: utf-8 -*-
"""
Checklist Engine RAB
====================
Pemeriksaan RAB (rule deterministik, tanpa LLM):
  - Section III (III.A-III.L): kelengkapan key-value metadata JSON hasil ExtractorRAB
  - Section IV : IV.B (BAS KEP-331), IV.C (SBM), IV.D (batas biaya pendukung 30%)

Modul pustaka (bukan notebook): tidak ada evaluasi yang berjalan saat import.
    import checklist_rab as cr
    rows = cr.evaluate_document("X.json")       # atau dict hasil ExtractorRAB

Path referensi diambil dari checklist_common (env CHECKLIST_RULES_DIR, CHECKLIST_KATALOG_DIR,
CHECKLIST_WILAYAH_CSV, CHECKLIST_CACHE_DIR); bawaan relatif terhadap folder modul ini:
    rules/rules_config.json, rules/KEP-331-..., rules/sbm_2026.json, rules/sbm_2027.json,
    rules/sbm_override.json (opsional), katalog_sbm/, wilayah_indonesia.csv
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import checklist_common as cc
from checklist_common import ALLOWED_STATUSES, load_json_document, write_report

log = logging.getLogger("checklist_rab")

# ---------------------------------------------------------------------------
# 1. Konfigurasi
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "rab-only-checklist-v3.1.0"
PROMPT_VERSION = "not_used_rab_deterministic"
model_version = "not_used_rab_deterministic"
RULES_VERSION = cc.RULES_VERSION
REF_DIR = cc.RULES_DIR
KATALOG_DIR = cc.KATALOG_DIR

# ---------------------------------------------------------------------------
# 2. Rules (Section III + IV)
# ---------------------------------------------------------------------------
_cfg, rules_by_code, RULES_HASH = cc.load_rules()
for _sc in ("III", "IV"):
    if _sc not in rules_by_code:
        raise KeyError(f"Section {_sc} untuk RAB tidak ditemukan pada {cc.RULES_PATH}.")
ROUTES = {"RAB": {"section_code": "III", "label": "Kelengkapan RAB", "rules": rules_by_code["III"]}}


# %% [markdown] sel 8
# ## 4. Deterministic key-value matching\n\nPemeriksaan format menggunakan nama key JSON dan nilai yang terkait. Tidak menggunakan embedding atau semantic retrieval.
# %% [code] sel 9
# Semantic retrieval dihapus: struktur input bersifat key-value.

# %% [markdown] sel 10
# ## Integrasi evaluator#
# Tidak digunakan pada versi ini. Seluruh kriteria RAB diperiksa menggunakan rule engine deterministik.
# %% [code] sel 11
# Evaluasi RAB tidak memerlukan integrasi LLM eksternal.

# %% [markdown] sel 12
# ## Kontrak evaluasi RAB#
# Pemeriksaan dilakukan secara deterministik terhadap key-value JSON dan validasi tambahan RAB Section IV.
# %% [code] sel 13
def criterion_method(kind, item):
    """Seluruh kriteria RAB diperiksa menggunakan rule deterministik."""
    if kind != "RAB":
        raise ValueError(f"Jenis dokumen tidak didukung: {kind}")
    return "rule_based_key"

# %% [markdown] sel 14
# ## 5. Rule engine deterministik berbasis key-value#
# Rule engine hanya memproses dokumen RAB. Pemeriksaan membaca nama key JSON dan memvalidasi nilai/flag terkait.
# %% [code] sel 15

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().casefold()

def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").casefold())

# Alias key JSON per kriteria. Alias dibuat spesifik per template/jenis dokumen.
# Pemeriksaan dilakukan terhadap KEY dan validitas nilai, bukan isi semantik.
KEY_RULES = {
    "RAB": {
        "III.A.1": {"keys": ["kementerian"], "type": "nonempty"},
        "III.B.1": {"keys": ["unit_eselon2_satker", "unit_eselon2", "satker"], "type": "nonempty"},
        "III.C.1": {"keys": ["kegiatan"], "type": "nonempty"},
        "III.D.1": {"keys": ["sasaran_kegiatan"], "type": "nonempty"},
        "III.E.1": {"keys": ["indikator_kinerja_kegiatan"], "type": "nonempty"},
        "III.F.1": {"keys": ["kro", "kode_kro"], "type": "nonempty"},
        #"III.G.1": {"keys": ["kro", "kode_kro"], "type": "nonempty"},
        "III.G.1": {"keys": ["kode_ro_lengkap", "kode_ro", "ro"], "type": "nonempty"}, #jika key ro dan kro beda
        "III.H.1": {"keys": ["volume"], "type": "positive_number"},
        "III.I.1": {"keys": ["satuan_ukur"], "type": "nonempty"},
        "III.J.1": {"keys": ["alokasi_dana", "jumlah_total_tercantum"], "type": "positive_number"},
        # Jabatan tidak menjadi key pemeriksaan pada kriteria ini.
        "III.K.1": {
            "keys": ["nama", "jabatan", "tanda_tangan", "tanda_tangan_elektronik"],
            "type": "signature_all"
    },
        "III.L.1": {"keys": ["nip"], "type": "nip"},
    },
}

def walk_key_values(node, path="$"):
    """Yield (key, value, json_path) untuk seluruh pasangan key-value bersarang."""
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}"
            yield str(key), value, child_path
            yield from walk_key_values(value, child_path)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk_key_values(value, f"{path}[{i}]")

def _value_is_present(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (dict, list)):
        return bool(value)
    return bool(str(value).strip())

def _key_rule_passes(rule_type, value, key):
    if rule_type == "nonempty":
        return _value_is_present(value)
    if rule_type == "positive_number":
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False
    if rule_type == "nip":
        # NIP terstruktur dapat berupa string langsung atau objek hasil ekstraksi OCR.
        if isinstance(value, dict):
            return value.get("ada_label") is True or bool(re.search(r"\b\d{18}\b", str(value)))
        return bool(re.search(r"\b\d{18}\b", str(value or "")))
    if rule_type == "signature":
        # Nama pejabat harus tersedia dan metadata tanda tangan terdeteksi true.
        return _value_is_present(value)
    return False

def deterministic_key_check(kind, doc, item):
    if kind != "RAB":
        raise ValueError(f"Jenis dokumen tidak didukung: {kind}")
    code = str(item.get("kriteria_kode", "")).strip().upper()
    rule = KEY_RULES.get("RAB", {}).get(code)
    if not rule:
        return {"status": "MANUAL", "evidence": [], "note": "Key rule belum dikonfigurasi"}, []

    #entries = list(walk_key_values(doc.get("raw", {})))
    raw = doc.get("raw", {})

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, (dict, list)):
        metadata = {}

    entries = list(walk_key_values(metadata, "$.metadata"))
    wanted = {_norm_key(k) for k in rule["keys"]}
    matches = [(key, value, path) for key, value, path in entries if _norm_key(key) in wanted]

    # Kriteria tanda tangan dan NIP membutuhkan flag verifikasi nested.
    # III.K.1 mensyaratkan nama dan jabatan, serta minimal satu flag tanda tangan bernilai true.
    if rule["type"] == "signature_all":
    # Ketiga field wajib ada dalam objek induk yang sama.
        signature_groups = {}

        for k, v, p in entries:
            normalized = _norm_key(k)
            if normalized in {"nama", "jabatan", "tandatangan", "tandatanganelektronik"}:
                parent_path = p.rsplit(".", 1)[0]
                signature_groups.setdefault(parent_path, {})[normalized] = v

        valid_signature_groups = {
            parent: fields
            for parent, fields in signature_groups.items()
            if _value_is_present(fields.get("nama"))
            and _value_is_present(fields.get("jabatan"))
            and (fields.get("tandatangan") is True or fields.get("tandatanganelektronik") is True)
        }

        passed = bool(valid_signature_groups)

        matches = [
            (k, v, p)
            for k, v, p in entries
            if _norm_key(k) in {"nama", "jabatan", "tandatangan", "tandatanganelektronik"}
            and p.rsplit(".", 1)[0] in valid_signature_groups
        ]


    elif rule["type"] == "signature":
        signature_flags = [v for k, v, p in entries
                           if _norm_key(k) == "detected" and ".tanda_tangan." in p]
        signature_flag = any(v is True for v in signature_flags)
        matches = [(k, v, p) for k, v, p in matches if _value_is_present(v)]
        passed = bool(matches) and signature_flag

    elif rule["type"] == "nip":
        # Ketiga field wajib ada dalam objek induk yang sama.
        nip_groups = {}

        for k, v, p in entries:
            normalized = _norm_key(k)
            if normalized in {"nip", "nipada", "nipvalid"}:
                parent_path = p.rsplit(".", 1)[0]
                nip_groups.setdefault(parent_path, {})[normalized] = v

        valid_nip_groups = {
            parent: fields
            for parent, fields in nip_groups.items()
            if isinstance(fields.get("nip"), str)
            and bool(fields["nip"].strip())
            and fields.get("nipada") is True
            and fields.get("nipvalid") is True
        }

        passed = bool(valid_nip_groups)

        matches = [
            (k, v, p)
            for k, v, p in entries
            if _norm_key(k) in {"nip", "nipada", "nipvalid"}
            and p.rsplit(".", 1)[0] in valid_nip_groups
        ]
    else:
        valid = [(k, v, p) for k, v, p in matches
                 if _key_rule_passes(rule["type"], v, k)]
        passed = bool(valid)
        matches = valid

    # Buat/ambil chunk evidence yang mengacu pada key-value sumber.
    chunks = doc.get("chunks", [])
    evidence_chunks = []
    target_entries = matches
    if rule["type"] == "signature_all":
        target_entries += [
            (k, v, p) for k, v, p in entries
            if _norm_key(k) == "detected"
            and v is True
            and ".tanda_tangan." in p
        ]

    elif rule["type"] == "signature":
        target_entries += [
            (k, v, p) for k, v, p in entries
            if _norm_key(k) == "detected"
            and v is True
            and ".tanda_tangan." in p
        ]
    if rule["type"] == "nip":
        target_entries = list(dict.fromkeys(target_entries + matches))

    for key, value, path in target_entries:
        existing = next((c for c in chunks if c.get("json_path") == path), None)
        if existing is None:
            quote = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            existing = {
                "chunk_id": f"{Path(doc.get('path', 'document')).stem}:kv:{len(chunks)+1}",
                "text": f"{key}: {quote}", "quote": quote,
                "source_file": doc.get("path"), "json_path": path,
                "page": None, "section": None, "heading": key,
                "block_id": None, "source_id": None, "metadata": {},
                "evidence_type": "source_fragment",
            }
            chunks.append(existing)
        evidence_chunks.append(existing)

    if passed and evidence_chunks:
        unique_keys = list(dict.fromkeys(
            k for k, _, _ in target_entries
        ))

        result = {
            "status": "LOLOS",
            "evidence": [f"key:{k}" for k in unique_keys],
            "note": "Key dan nilai ditemukan"
        }
    elif rule["type"] == "signature_all" and not passed:
        result = {
            "status": "TIDAK",
            "evidence": [f"key:{k}" for k, _, _ in target_entries],
            "note": "Nama dan jabatan wajib tersedia, serta tanda_tangan atau tanda_tangan_elektronik harus true"
        }
    elif rule["type"] == "signature" and any(v is False for k, v, p in entries
                                                if _norm_key(k) == "detected" and ".tanda_tangan." in p):
        result = {"status": "TIDAK", "evidence": [f"key:{k}" for k, _, _ in matches],
                  "note": "Nama penanggung jawab ditemukan, tetapi tanda tangan tidak terdeteksi"}
    elif rule["type"] == "nip" and any(v is False for k, v, p in entries
                                       if _norm_key(k) == "adalabel" and ".nip." in p):
        result = {"status": "TIDAK", "evidence": [], "note": "Label NIP secara eksplisit tidak ditemukan"}
    elif matches or (rule["type"] in {"signature", "nip"} and target_entries):
        result = {"status": "MANUAL", "evidence": [f"key:{k}" for k, _, _ in target_entries],
                  "note": "Key ditemukan tetapi nilai/flag verifikasi belum memenuhi syarat"}
    else:
        result = {"status": "TIDAK", "evidence": [], "note": "Key wajib tidak ditemukan atau nilainya kosong"}
    return result, evidence_chunks

# %% [code] sel 16
DETECTORS_BY_TYPE = KEY_RULES
NEED_ALL = set()  # Validasi multi-key ditangani eksplisit pada deterministic_key_check().

# %% [code] sel 17
def normalize_patterns(value):
    """Kompatibilitas untuk metadata laporan lama; tidak dipakai untuk matching key-value."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [p for p in value if isinstance(p, str) and p.strip()]
    return []

def validate_all_detectors(detector_by_type):
    """Validasi mapping key rule RAB."""
    for kind, detectors in detector_by_type.items():
        if not isinstance(detectors, dict):
            raise ValueError(f"Detector harus berupa mapping: {kind}")

        for code, rule in detectors.items():
            if kind != "RAB":
                raise ValueError(f"Jenis dokumen tidak didukung: {kind}")
            if not isinstance(rule, dict) or not rule.get("keys"):
                raise ValueError(f"Key rule tidak valid: {kind}/{code}")

validate_all_detectors(DETECTORS_BY_TYPE)

# %% [code] sel 18
def check_item(body, kriteria_kode, kind):
    """Adapter kompatibilitas; evaluasi utama menggunakan deterministic_key_check()."""
    return {"status": "MANUAL", "evidence": [], "note": "Gunakan deterministic_key_check() berbasis JSON key-value"}

# %% [code] sel 19
def validate_detector_patterns(detectors):
    """Tidak ada pola regex untuk pemeriksaan key-value."""
    return True

# %% [code] sel 20
def legacy_regex_adapter(kind, doc, item):
    """Adapter lama dipertahankan agar bentuk data hasil evaluasi/report tidak berubah."""
    return deterministic_key_check(kind, doc, item)


# ---------------------------------------------------------------------------
# 6. Validator keluaran (implementasi di checklist_common)
# ---------------------------------------------------------------------------
def _versions():
    return {"pipeline": PIPELINE_VERSION, "rules_version": RULES_VERSION, "rules_sha256": RULES_HASH,
            "model_version": model_version, "sft_model_version": None,
            "prompt_version": PROMPT_VERSION, "embedding_model": "not_used_key_value"}


def validate_prediction(pred, candidates, doc_name, kind, item, method, rule_hits=None):
    return cc.validate_prediction(pred, candidates, doc_name, kind, item, method,
                                  ROUTES[kind]["section_code"], _versions(), rule_hits)


build_reasoning = cc.build_reasoning


# %% [markdown] sel 24
# ## 7. Routing metode checklist#
# Seluruh kriteria RAB Section III diperiksa secara deterministik berdasarkan key-value JSON. Pemeriksaan tambahan Section IV menjalankan validasi IV.B (BAS), IV.C (SBM), dan IV.D (batas biaya pendukung).
# %% [markdown] sel 25
# ## 8. Evaluasi setiap dokumen#
# Notebook hanya memuat file JSON dari `input/RAB/`. Setiap kriteria Section III diperiksa dengan rule deterministik; hasil validasi tambahan Section IV kemudian ditambahkan ke laporan RAB.
# %% [code] sel 26
def evaluate_item(kind, doc_name, item, doc):
    if kind != "RAB":
        raise ValueError(f"Jenis dokumen tidak didukung: {kind}")

    method = criterion_method(kind, item)
    legacy_result, legacy_matches = deterministic_key_check("RAB", doc, item)

    if legacy_result["status"] == "LOLOS" and legacy_matches:
        c = legacy_matches[0]
        pred = {
            "status": "LOLOS",
            "alasan": "Key JSON yang diwajibkan ditemukan dan nilainya memenuhi validasi deterministik.",
            "evidence_quote": c.get("quote", ""),
            "evidence_chunk_id": c.get("chunk_id"),
        }
    elif legacy_result["status"] == "TIDAK":
        pred = {
            "status": "TIDAK",
            "alasan": legacy_result["note"] + ".",
            "evidence_quote": "",
            "evidence_chunk_id": None,
        }
    else:
        c = legacy_matches[0] if legacy_matches else None
        pred = {
            "status": "MANUAL",
            "alasan": legacy_result["note"] + ". Perlu verifikasi reviewer.",
            "evidence_quote": c.get("quote", "") if c else "",
            "evidence_chunk_id": c.get("chunk_id") if c else None,
        }

    candidates = list(doc.get("chunks", []))
    for candidate in legacy_matches:
        if not any(existing.get("quote") == candidate.get("quote") for existing in candidates):
            candidates.append(candidate)

    row = validate_prediction(pred, candidates, doc_name, kind, item, method, [])
    row["alasan"], row["feedback"] = build_reasoning(kind, item, row["status"], legacy_result)
    if row["status"] == "LOLOS":
        row["catatan"] = ""
    if row.get("status") != "LOLOS" and not row.get("catatan"):
        row["catatan"] = legacy_result.get("note", "")

    row["evidence_pola"] = list(legacy_result.get("evidence", []))
    row["candidate_evidence"] = [{
        "chunk_id": c["chunk_id"], "quote": c["quote"], "text": c["text"],
        "score": None, "json_path": c["json_path"],
        "page": c.get("page"), "section": c.get("section"), "heading": c.get("heading"),
        "block_id": c.get("block_id"),
    } for c in candidates]
    row["rule_hits"] = []
    row["legacy_regex_result"] = {
        "status": legacy_result.get("status"),
        "matched_patterns": legacy_result.get("evidence", []),
        "note": legacy_result.get("note"),
    }
    row["legacy_regex_evidence_chunks"] = [
        {"chunk_id": c.get("chunk_id"), "patterns": legacy_result.get("evidence", [])}
        for c in legacy_matches
    ]
    row["evidence_quote"] = pred.get("evidence_quote", "")
    return row

# %% [markdown] sel 27
# ## 9. Validasi RAB RULE IV
# %% [code] sel 28
# ============================================================
# IV. Pemeriksaan lintas referensi RAB: IV.B (BAS), IV.C (SBM), IV.D (batas 30%)
# Taruh SETELAH cell "Evaluasi setiap dokumen", SEBELUM "## 9. Simpan laporan".
# Menambahkan hasil validasi Section IV ke all_results["RAB"].
# ============================================================
# REF_DIR dan KEP331_PATH: dari checklist_common (env CHECKLIST_RULES_DIR)
KEP331_PATH = REF_DIR / "KEP-331-2021_Lampiran_II_hal_tercetak_62-122.json"

BATAS_PENDUKUNG = 0.30
TOLERANSI_RP = 1
G_DIPERIKSA_OTOMATIS = False   # False: aâ€“f â‰¤ batas hasilnya MANUAL, bukan LOLOS


KATEGORI_NAMA = {
    "a": "Honorarium kegiatan",
    "b": "Biaya konsumsi rapat dalam kantor",
    "c": "Biaya kegiatan seremonial",
    "d": "Biaya percetakan",
    "e": "Pengadaan souvenir",
    "f": "Pengadaan ATK untuk kegiatan",
}
KATEGORI_POLA = {
    # honor, honorarium, honoraium, honorium, serta sebutan narasumber/pembicara/praktisi/pakar
    "a": r"\b(honor(a?r?i?um)?|narasumber|pembicara|praktisi|pakar)\b",
    "c": r"\b(seremonial|pembukaan|penutupan|peresmian|upacara|umbul-umbul)\b",
    "d": r"\b(cetak|penggandaan|fotokopi|foto\s?copy|banner|spanduk|sertifikat|kartu|backdrop|penjilidan|undangan|modul|pencetakan)\b",
    "e": r"\b(souvenir|seminar\s?kit|cendera\s?mata|cendera|plakat)\b",
    "f": r"\b(atk|alat tulis|kertas|tinta|hvs|pulpen|ballpoint|spidol|binder|clips|clip|klip|penjepit|staples|strapples|strappler|map|ordner)\b",
}

POLA_KONSUMSI = r"konsumsi|makan|snack"
POLA_RAPAT    = r"rapat"
POLA_EXCL_B   = r"paket|fullday|fullboard|halfday|luar kota|meeting"
G_OVERRIDE_PATHS = set()   # isi json_path item yang disepakati sebagai komponen g

# --- Referensi BAS (KEP-331): cukup kode 6 digit; uraian diabaikan karena sering rusak ---
from difflib import SequenceMatcher

kep = json.loads(KEP331_PATH.read_text(encoding="utf-8-sig"))
BAS_INFO = {}   # kode -> {"uraian": ..., "rusak": bool}
for seg in kep.get("segments", []):
    for row in seg.get("rows", []):
        k = str(row.get("kode_akun", "")).strip()
        if len(k) == 6 and k.isdigit() and k not in BAS_INFO:
            ur = str(row.get("uraian_akun", "") or "").strip()
            rusak = (not ur) or ("code_without_name" in row.get("flags", []))
            BAS_INFO[k] = {"uraian": ur, "rusak": rusak}

# Koreksi kesalahan OCR yang sudah diketahui di file KEP-331. Tambah jika ditemukan lagi.
OCR_FIX = {"honorarium": "honor"}

def norm_uraian(s):
    s = str(s or "").casefold()
    for a, b in OCR_FIX.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ===== IV.B v2: uraian harus sama persis per kata; nominal, angka, dan simbol diabaikan (anomali parsing ekstraksi) =====
def _rapikan_spasi(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def uraian_tanpa_anomali(s):
    """Buang nominal (mis. 'Rp18,661,000'), angka, dan simbol; huruf dan urutan kata tetap dipertahankan."""
    s = str(s or "")
    s = re.sub(r"(?i)\bRp\.?\s*[\d.,]+", " ", s)   # nominal berawalan Rp
    s = re.sub(r"\d[\d.,]*", " ", s)                # angka dan nominal tanpa Rp
    s = re.sub(r"[\W\d_]+", " ", s)                 # simbol, tanda baca, bullet, titik dua, strip
    return _rapikan_spasi(s)

def cek_uraian(ur_rab, ur_bas):
    """IV.B: kata dan urutannya harus sama persis dengan uraian BAS KEP-331 (huruf besar-kecil diabaikan).
    Nominal, angka, dan simbol diabaikan karena sering ikut tergabung saat ekstraksi."""
    raw_a, raw_b = _rapikan_spasi(ur_rab), _rapikan_spasi(ur_bas)
    a = uraian_tanpa_anomali(ur_rab).casefold()
    b = uraian_tanpa_anomali(ur_bas).casefold()
    if not a:
        return "MANUAL", "uraian RAB kosong setelah membuang angka dan simbol"
    catatan = "" if (raw_a == raw_b) else " (angka, simbol, atau huruf besar-kecil diabaikan)"
    if a == b:
        return "LOLOS", "uraian sama persis dengan BAS" + catatan
    if b.startswith(a + " "):
        sisa = b[len(a):].strip()[:60]
        return "MANUAL", (f"uraian BAS memuat teks tambahan '{sisa}...' setelah uraian RAB; "
                          f"kemungkinan kesalahan parser KEP-331, perlu verifikasi")
    return "TIDAK", f"uraian RAB '{ur_rab}' tidak sama dengan BAS '{ur_bas}'"

# %% [markdown] sel 29
# ## Katalog SBM
# %% [code] sel 30
# ===== IV.C v3: SBM terstruktur, pemecahan uraian (wilayah / eselon / kategori), alasan per item =====
from collections import Counter

SBM_CONFIG = {
    2026: {"path": REF_DIR / "sbm_2026.json", "dasar": "PMK 32/2025"},
    2027: {"path": REF_DIR / "sbm_2027.json", "dasar": "KMK 296/2026"},
}
SBM_OVERRIDE_PATH = REF_DIR / "sbm_override.json"
SBM_OVERRIDE = {norm_uraian(k): str(v) for k, v in (
    json.loads(SBM_OVERRIDE_PATH.read_text(encoding="utf-8")).items()
    if SBM_OVERRIDE_PATH.exists() else {})}

SBM_STOP = {"kudapan", "honor", "honorarium", "belanja", "biaya", "uang", "harian", "dan", "per", "untuk",
            "pengadaan", "orang", "org", "keg", "kegiatan", "hari", "jam", "kali", "paket",
            "satuan", "yang", "dengan", "atau", "pada", "dalam", "di", "lain", "lainnya",
            "disetarakan", "setara", "setingkat", "sejenis", "kgt"}

GENERIK = {"jasa", "barang", "biaya", "sewa", "penyediaan", "pembelian", "pengadaan", "pejabat", "tenaga"}
BULLET_RE = re.compile(r"^\s*(?:[-•*]+\s*|\d+[.)]?\s+|\(?[a-z][.)]\s+)", re.I)
REG_KEY_RE = re.compile(r"^(provinsi|ibu kota provinsi|kabupaten/kota tujuan|kabupaten/kota|kabupaten|kota|negara|asal|tujuan)$", re.I)
UR_REG_KEYS = {"ASAL", "TUJUAN", "KOTA", "PROVINSI", "IBU KOTA PROVINSI", "KABUPATEN/KOTA TUJUAN", "KABUPATEN/KOTA", "NEGARA"}
KEY_TIPE = {"provinsi": "prov", "ibu kota provinsi": "kota", "kota": "kota", "asal": "kota", "tujuan": "kota",
            "kabupaten/kota tujuan": "kabkota", "kabupaten/kota": "kabkota", "kabupaten": "kab", "negara": "negara"}
MATA_UANG_OK = {"", "IDR", "RP", "RUPIAH"}

def tipe_dari_kunci(k):
    return KEY_TIPE.get(str(k).strip().casefold(), "")

def tipe_kompatibel(a, b):
    if not a or not b or a == b:
        return True
    return {a, b} <= {"kabkota", "kab", "kota"}

def _flatten_uraian_dict(d, out_regs, out_txt):
    """Uraian leaf berbentuk objek, mis. {'KOTA': {'ASAL': 'JAKARTA', 'TUJUAN': 'AMBON'}}: ambil wilayahnya."""
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str):
                out_txt.append(v)
                if str(k).strip().upper() in UR_REG_KEYS:
                    core, _ = region_parts(v)
                    if core:
                        out_regs.add((core, tipe_dari_kunci(k)))
            else:
                _flatten_uraian_dict(v, out_regs, out_txt)
    return out_regs, out_txt
REGION_STOP = {"prov", "provinsi", "kab", "kabupaten", "kota", "dki", "daerah", "khusus", "ibukota", "ibu"}
TYPE_WORDS = {"kab": "kab", "kabupaten": "kab", "kota": "kota", "prov": "prov", "provinsi": "prov"}
ROMAWI = {"1": "i", "2": "ii", "3": "iii", "4": "iv"}
ESEL_RE = re.compile(r"\b(?:eselon|es)\.?\s*(iv|iii|ii|i|[1-4])\b")
GOL_RE = re.compile(r"\bgol(?:ongan)?\.?\s*(iv|iii|ii|i|[1-4]|[a-d])\b")

_TEMPLATE = {
    "HARGA_SESUAI": "Harga Rp{hs} tidak melebihi batas Rp{ref} ({lampiran}) untuk {padanan}.",
    "HARGA_MELEBIHI_BATAS": "Harga Rp{hs} melebihi batas tertinggi Rp{ref} untuk {padanan}{wilayah}.",
    "DAPAT_DILAMPAUI": "Harga Rp{hs} melebihi Rp{ref}, tetapi padanan {padanan} berada di lampiran dapat dilampaui.", 
    "WILAYAH_TIDAK_TERDETEKSI": "Tarif {padanan} bergantung wilayah, tetapi uraian tidak menyebut wilayah.",
    "WILAYAH_AMBIGU": "Nama wilayah di uraian cocok lebih dari satu entri SBM tanpa pembeda kab/kota.",
    "WILAYAH_BANYAK": "Uraian menyebut lebih dari satu wilayah, sehingga tarif {padanan} tidak dapat ditentukan.",
    "WILAYAH_TIDAK_DIATUR": "Wilayah di uraian{wilayah} tidak tercantum pada tarif {padanan}.",
    "ESELON_TIDAK_TERDETEKSI": "Tarif {padanan} bergantung eselon atau golongan, tetapi uraian tidak menyebut eselon.",
    "TIER_TIDAK_COCOK": "Eselon atau golongan di uraian tidak cocok dengan tarif {padanan}.",
    "PADANAN_TIDAK_ADA": "Tidak ada padanan SBM sebagai batas tertinggi untuk uraian ini.",
    "UANG_FULLBOARD_TANPA_PADANAN": "Uang harian kegiatan rapat/pertemuan di luar kantor (fullboard) tidak memiliki padanan tarif di SBM; perlu verifikasi manual.",
    "UANG_JENIS_TIDAK_JELAS": "Uraian uang harian atau uang representasi belum menyebut luar kota, dalam kota, atau diklat, sehingga tarif tidak dapat dipilih.",
    "KANDIDAT_BANYAK": "Beberapa tarif SBM cocok dengan besaran berbeda ({padanan}); uraian belum cukup spesifik untuk memilih satu tarif, sehingga perlu verifikasi manual.",
    "HARGA_DI_BAWAH_SEMUA_TARIF": "Harga Rp{hs} berada di bawah seluruh kemungkinan tarif ({padanan}), sehingga pasti tidak melebihi batas.",
    "HARGA_DI_ATAS_SEMUA_TARIF": "Harga Rp{hs} melebihi seluruh kemungkinan tarif Rp{ref} ({padanan}).",
    "HARGA_KOSONG": "Harga satuan tidak tersedia.",
    "KONTEKS_BANYAK": "Kata utama uraian cocok dengan lebih dari satu kelompok tarif SBM ({padanan}); uraian tidak menyebut kelompok yang dimaksud.",
    "PROVINSI_TIDAK_DISEBUT": "Tarif {padanan} dihitung per provinsi, tetapi uraian{wilayah} tidak menyebut nama provinsi. Tulis nama provinsi (kota atau kabupaten saja tidak cukup).",
    "PROVINSI_AMBIGU": "Status wilayah ambigu / lebih dari satu opsi provinsi: {wilayah}.",
    "JENIS_UANG_HARIAN_TIDAK_JELAS": "Uraian uang harian belum bisa dibedakan: tidak ada wilayah (perjalanan dinas dalam negeri) dan tidak ada kata paket rapat (fullboard/fullday/halfday/residence). Tulis wilayah atau jenis paket kegiatannya.",
    "JENIS_UANG_HARIAN_KONFLIK": "Uraian uang harian memuat wilayah, tetapi akun RAB adalah paket meeting luar kota (524119). Wilayah menandakan perjalanan dinas dan akun menandakan paket, sehingga perlu diperiksa manual.",
    "AKUN_TIDAK_SESUAI": "Tarif SBM dengan judul yang cocok ({padanan}) tidak termasuk rincian akun {akun} menurut KEP-331 ({rincian_akun}).",
    "DAPAT_DILAMPAUI": "Tarif {padanan} berada di lampiran II (dapat dilampaui). Harga boleh melebihi tarif SBM jika harga pasar naik atau ada kondisi khusus; perlu verifikasi manual.",
    "MATA_UANG_ASING": "Uraian memuat perjalanan dinas luar negeri. SBM memakai USD sedangkan RAB memakai rupiah, sehingga perlu konversi kurs secara manual.",
    "RUTE_TIDAK_COCOK": "Rute perjalanan ({rute}) tidak sesuai dengan tarif {padanan}.",
    "WILAYAH_TIPE_BEDA": "Nama wilayah ({wilayah}) ada pada tarif {padanan}, tetapi jenis wilayahnya (provinsi/kota/kabupaten) berbeda.",
    "KATEGORI_TIDAK_TERDETEKSI": "Tarif {padanan} dibedakan menurut jenis (mis. makan atau kudapan), tetapi uraian tidak menyebut jenis tersebut.",
    "TARIF_TIDAK_DIBACA": "Struktur tarif {padanan} tidak dapat dibaca otomatis.",
    "TAG_TIDAK_COCOK": "Uraian menyebut jenis (mis. halfday/fullday, makan/kudapan, atau moda kendaraan) yang berbeda dengan tarif {padanan}.",
}

class _SafeDict(dict):
    def __missing__(self, key):
        return "-"

def _alasan(kode, **kw):
    return _TEMPLATE[kode].format_map(_SafeDict(**kw))

# Sinonim: semua bentuk di kiri dipetakan ke bentuk baku di kanan sebelum dibandingkan (RAB dan SBM)
SINONIM = {
    "narsumber": "narasumber", "nara sumber": "narasumber",
    "meeting": "rapat", "pertemuan": "rapat",
    "transportasi": "transport",
    "honorarium": "honor", "gaji": "honor",
    "hotel": "penginapan",
    "pelatihan": "diklat",
    "snack": "kudapan", 
    "makan": "nasi box",
    "fulboard": "fullboard", "full board": "fullboard", 
    "fulday": "fullday",
    "uang saku harian": "uang harian", "uang saku": "uang harian",
    "biaya representasi": "uang representasi",
    "i": "1", "ii": "2", "iii": "3", "iv": "4"
}
_SINONIM_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SINONIM), key=len, reverse=True)) + r")\b")

def norm_text(s):
    s = BULLET_RE.sub("", str(s or ""), count=1)
    s = re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()
    s = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", s)   # 8jam -> 8 jam
    return _SINONIM_RE.sub(lambda m: SINONIM[m.group(1)], s)

def stem(t):
    """Pencocokan bentuk kata sederhana: 6 huruf pertama untuk kata >= 7 huruf (transport/transportasi)."""
    return t[:6] if len(t) >= 7 else t

def _first_sig(text):
    """Kata pertama yang bermakna pada judul (setelah kata stop dan generik), dalam bentuk stem."""
    return next((stem(t) for t in norm_text(text).split()
                 if len(t) > 2 and t not in SBM_STOP and t not in GENERIK and not t.isdigit()), "")

def tokens_sbm(s):
    return {stem(t) for t in norm_text(s).split()
            if len(t) > 2 and t not in SBM_STOP and not t.isdigit()}

def _leaf_alts(text):
    """Uraian leaf dipecah pada '/' jadi alternatif token, mis. 'Ketua/Wakil ketua' -> [{ketua}, {wakil, ketua}]."""
    return [t for t in (tokens_sbm(p) for p in str(text or "").split("/")) if t]

def _contains(seq, sub):
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))

CORE_ALIAS = {"sumatra": "sumatera", "sumatra utara": "sumatera utara", "sumatra barat": "sumatera barat", "sumatra selatan": "sumatera selatan",
               "papuaoh": "papua"}   # ejaan tidak seragam di SBM (sumber resmi tetap tidak diubah)

def region_parts(text):
    """(inti_wilayah, tipe) dari teks, mis. 'Kab. Blitar' -> ('blitar', 'kab'); 'D.K.I. JAKARTA' -> ('jakarta', '')."""
    toks = norm_text(text).split()
    types = {TYPE_WORDS[t] for t in toks if t in TYPE_WORDS}
    core = " ".join(t for t in toks if t not in REGION_STOP and t not in TYPE_WORDS and len(t) > 2)
    core = CORE_ALIAS.get(core, core)
    return core, (types.pop() if len(types) == 1 else "")

def build_region_vocab(data):
    """Kosakata wilayah diambil dari isi SBM itu sendiri (bukan daftar tulisan tangan)."""
    vocab = {}
    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
        elif isinstance(n, dict):
            for k, v in n.items():
                if REG_KEY_RE.match(str(k).strip()) and isinstance(v, str) and v.strip():
                    core, _ = region_parts(v)
                    if core:
                        vocab.setdefault(core, set()).add(tipe_dari_kunci(k))
                if isinstance(v, (list, dict)):
                    walk(v)
    walk(data.get("lampiran", []))
    return vocab

def parse_wilayah(text, vocab):
    toks = norm_text(text).split()
    itype_set = {TYPE_WORDS[t] for t in toks if t in TYPE_WORDS}
    itype = itype_set.pop() if len(itype_set) == 1 else ""
    hits = [c for c in vocab if _contains(toks, c.split())]
    keep = [c for c in hits if not any(c != o and _contains(o.split(), c.split()) for o in hits)]
    ids, ambigu = [], False
    for c in keep:
        # tipe wilayah item hanya dari teks item (kab/kota); tanpa kata tipe, nama yang punya >1 tipe di SBM = ambigu
        if not itype and len(vocab[c] - {""}) > 1:
            ambigu = True
        ids.append((c, itype))
    return {"ids": ids, "ambigu": ambigu, "banyak": len({c for c, _ in ids}) > 1}

def provinsi_set(vocab):
    """Daftar nama provinsi yang diambil dari SBM (kunci 'provinsi'); kota/kabupaten tidak masuk."""
    return {c for c, tipes in vocab.items() if "prov" in tipes and not re.search(r"pejabat|menteri", c)}

def region_cocok(ids, rec_regions):
    for c, t in ids:
        for rc, rt in rec_regions:
            if c == rc and tipe_kompatibel(t, rt):
                return True
    return False

def region_tipe_beda(ids, rec_regions):
    """Nama wilayah sama, tetapi tipe (provinsi/kota/kabupaten) di RAB dan SBM berbeda."""
    for c, t in ids:
        for rc, rt in rec_regions:
            if c == rc and not tipe_kompatibel(t, rt):
                return True
    return False

def canon_tier(text):
    t = str(text or "").casefold()
    out = {"eselon " + ROMAWI.get(m, m) for m in ESEL_RE.findall(t)}
    for m in re.finditer(r"\beselon\s+((?:(?:iv|iii|ii|i|[1-4])\s*(?:dan|/|,|&)\s*)+(?:iv|iii|ii|i|[1-4])\b)", t):
        out |= {"eselon " + ROMAWI.get(x, x) for x in re.findall(r"\b(iv|iii|ii|i|[1-4])\b", m.group(1))}
    out |= {"gol " + ROMAWI.get(m, m) for m in GOL_RE.findall(t)}
    if re.search(r"\bmenteri\b", t):
        out.add("menteri")
    return out

def expand_besaran(bes):
    """Angka -> [('', nilai)]; dict kategori/tier -> satu entri per kunci; daftar objek -> per kunci; tidak terbaca -> [None]."""
    if isinstance(bes, (int, float)):
        return [("", float(bes))]
    if isinstance(bes, dict):
        out = []
        for k, v in bes.items():
            if isinstance(v, (int, float)):
                out.append((str(k), float(v)))
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        out.append((f"{k} {k2}", float(v2)))
        return out or [None]
    return [None]

EXCLUDED_CURRENCY = []

def _build_records(lamp, vocab):
    jenis = lamp.get("jenis_standar", "")
    recs = []

    def walk(node, anc, parent_no, top, regs_anc, mu_inh=""):
        # anc: tuple (key, judul_tokens, judul_teks) dari akar ke induk terdekat
        if isinstance(node, list):
            for x in node:
                walk(x, anc, parent_no, top, regs_anc, mu_inh)
            return
        if not isinstance(node, dict):
            return
        regs = set(regs_anc)
        for k, v in node.items():
            if REG_KEY_RE.match(str(k).strip()) and isinstance(v, str) and v.strip():
                core, _ = region_parts(v)
                if core:
                    regs.add((core, tipe_dari_kunci(k)))
        no = str(node.get("no") or "").strip() or parent_no

        mu_efektif = str(node.get("mata_uang") or mu_inh or "").strip().upper()   # mata uang diwarisi dari tabel induk
        if "besaran" in node and mu_efektif not in MATA_UANG_OK:
            EXCLUDED_CURRENCY.append(no)                       # tabel non-rupiah dilewati
            return
        if "besaran" in node and (isinstance(node["besaran"], list) or "perwakilan" in node):
            EXCLUDED_CURRENCY.append(no)                       # tarif perwakilan/daftar: mata uang tidak ditandai, dilewati
            return
        if "besaran" in node:                                   # leaf: tarif
            judul_node = str(node.get("judul") or "").strip()
            ur_raw = node.get("uraian")
            if isinstance(ur_raw, dict):
                ur_regs, ur_txt = _flatten_uraian_dict(ur_raw, set(), [])
                regs |= ur_regs
                label = judul_node
                leaf_is_region = False
            else:
                label = str(ur_raw or judul_node or "")
                core_label, _ = region_parts(label)
                leaf_is_region = bool(core_label) and core_label in vocab   # baris tabel wilayah
                if leaf_is_region:
                    regs.add(region_parts(label))
            base_alts = [] if leaf_is_region else _leaf_alts(label)
            tier0 = canon_tier(label)
            for _, _, jt in anc:
                tier0 |= canon_tier(jt)
            chain = [jt for _, _, jt in anc if jt] + ([label] if label else [])
            judul = " > ".join(chain)
            g_key, g_tok, g_judul = (anc[-1] if anc else (no, set(), ""))
            for ex in expand_besaran(node["besaran"]):
                rec = {"lamp": lamp.get("id"), "jenis": jenis, "no": no, "top": top,
                       "grp": g_key, "grp_judul": g_judul, "grp_tokens": g_tok,
                       "judul": judul, "alts": list(base_alts), "regions": set(regs),
                       "tier": set(tier0), "anc": list(anc), "kunci_tok": set()}
                if ex is None:
                    rec.update(besaran=None, kunci="", tidak_dapat_dibaca=True)
                else:
                    kunci, val = ex
                    rec.update(besaran=val, kunci=kunci, tidak_dapat_dibaca=False)
                    if kunci:
                        kt = canon_tier(kunci)
                        if kt:
                            rec["tier"] |= kt
                        else:
                            rec["alts"] += _leaf_alts(kunci)
                            rec["kunci_tok"] = tokens_sbm(kunci)   # jenis tarif, mis. makan / kudapan(snack)
                recs.append(rec)
            return

        judul_txt = str(node.get("judul") or "").strip()
        if not judul_txt and isinstance(node.get("uraian"), str) and "besaran" not in node:
            judul_txt = str(node["uraian"]).strip()          # judul kelompok yang ditulis di 'uraian', mis. 'Pejabat Eselon I dan II'
        new_anc, new_top = anc, top
        if judul_txt:
            new_anc = anc + ((no or judul_txt, tokens_sbm(judul_txt), judul_txt),)
        if re.fullmatch(r"\d+", no) and judul_txt:
            new_top = no
        for v in node.values():
            if isinstance(v, (list, dict)):
                walk(v, new_anc, no, new_top, regs, mu_efektif)

    walk(lamp, (), "", "", set())
    return recs

_SBM_CACHE = {}
KATALOG_VERSI = 6   # naikkan bila struktur katalog berubah (v5: mata uang diwarisi dari tabel induk; v6: alias inti wilayah)
KATALOG_DIR = cc.KATALOG_DIR   # hasil build katalog; dibuat ulang otomatis jika SBM berubah (env CHECKLIST_KATALOG_DIR)

def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _katalog_to_json(recs, vocab):
    """Rekaman SBM -> JSON. Set jadi daftar terurut; dimensi dibuat eksplisit agar mudah diperiksa."""
    out_recs = []
    for r in recs:
        d = dict(r)
        d["grp_tokens"] = sorted(r["grp_tokens"])
        d["alts"] = [sorted(x) for x in r["alts"]]
        d["regions"] = sorted([list(x) for x in r["regions"]])
        d["tier"] = sorted(r["tier"])
        d["kunci_tok"] = sorted(r["kunci_tok"])
        d["anc"] = [[k, sorted(t), jt] for k, t, jt in r["anc"]]
        d["jalur"] = " > ".join(t for _, _, t in r["anc"] if t) + ((" > " + r["kunci"]) if r.get("kunci") else "")
        d["dimensi"] = {"provinsi": sorted({c for c, t in r["regions"] if t == "prov"}),
                        "tier": sorted(r["tier"]), "paket_atau_kunci": r.get("kunci", "")}
        out_recs.append(d)
    return {"vocab": {c: sorted(t) for c, t in vocab.items()}, "recs": out_recs}

def _katalog_from_json(obj):
    vocab = {c: set(t) for c, t in obj["vocab"].items()}
    recs = []
    for d in obj["recs"]:
        r = dict(d)
        r["grp_tokens"] = set(d["grp_tokens"])
        r["alts"] = [set(x) for x in d["alts"]]
        r["regions"] = {(c, t) for c, t in d["regions"]}
        r["tier"] = set(d["tier"])
        r["kunci_tok"] = set(d["kunci_tok"])
        r["anc"] = [(k, set(t), jt) for k, t, jt in d["anc"]]
        for kk in ("jalur", "dimensi"):
            r.pop(kk, None)
        recs.append(r)
    return recs, vocab

def build_katalog(tahun):
    """Buat katalog JSON dari file SBM sumber (dipanggil sekali, atau otomatis jika katalog kedaluwarsa).
    Bila folder katalog tidak bisa ditulis (mis. image read-only), katalog tetap dipakai dari memori."""
    cfg = SBM_CONFIG[tahun]
    data = json.loads(cfg["path"].read_text(encoding="utf-8-sig"))
    vocab = build_region_vocab(data)
    recs = []
    for lamp in data.get("lampiran", []):
        recs += _build_records(lamp, vocab)
    obj = {"versi": KATALOG_VERSI, "tahun": tahun, "sumber": cfg["path"].name, "sha256_sumber": _sha(cfg["path"]),
           **_katalog_to_json(recs, vocab)}
    try:
        KATALOG_DIR.mkdir(parents=True, exist_ok=True)
        (KATALOG_DIR / f"katalog_sbm_{tahun}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        log.warning("katalog SBM %s tidak dapat ditulis ke %s: %r (dipakai dari memori)", tahun, KATALOG_DIR, e)
    return recs, vocab

def get_sbm_bundle(tahun):
    """Return (recs, vocab) untuk TA SBM tertentu, dari katalog JSON bila masih sesuai dengan file sumber."""
    cfg = SBM_CONFIG.get(tahun)
    if cfg is None or not cfg["path"].is_file():
        return None
    key = str(cfg["path"])
    if key not in _SBM_CACHE:
        kat = KATALOG_DIR / f"katalog_sbm_{tahun}.json"
        if kat.is_file():
            obj = json.loads(kat.read_text(encoding="utf-8"))
            if obj.get("versi") == KATALOG_VERSI and obj.get("sha256_sumber") == _sha(cfg["path"]):
                _SBM_CACHE[key] = _katalog_from_json(obj)
                return _SBM_CACHE[key]
        _SBM_CACHE[key] = build_katalog(tahun)
    return _SBM_CACHE[key]

def resolve_tahun_sbm(tahun):
    """TA <= 2026 memakai SBM 2026; TA 2027 ke atas memakai SBM 2027 (atau yang terbaru <= TA)."""
    kunci = sorted(SBM_CONFIG)
    if tahun <= kunci[0]:
        return kunci[0]
    return max(k for k in kunci if k <= tahun)

def _pilih_jenis(cand):
    bt = [r for r in cand if r["jenis"] == "batas_tertinggi"]
    return bt or cand

# Rincian akun (KEP-331 Lampiran II): hanya judul SBM yang sesuai rincian akun yang boleh dipadankan.
# Kunci = kode akun; nilai = pola judul SBM (huruf besar) yang diizinkan. Akun tanpa pola: tidak disaring.
AKUN_RINCIAN_SBM = {   # kode akun: (judul SBM yang boleh, judul SBM yang ditolak, keterangan) - sumber: cab09df3 daftar_padanan_sbm_kep331.txt
    "521115": (r"HONORARIUM.*(PENGELOLA KEUANGAN|PENGADAAN|PEJABAT PENGADAAN|POKJA|PPHP|KPA|PPK|BENDAHARA)", None, "honor operasional satker rutin (PJ pengelola keuangan, pengadaan barang/jasa)"),
    "521213": (r"HONORARIUM.*(TIM PELAKSANA|SEKRETARIAT|PENGARAH|PANITIA|PENANGGUNG JAWAB|KETUA|SEKRETARIS|ANGGOTA|STAF|PEMBUAT ARTIKEL|PENGADAAN)", None, "honor output kegiatan (tim, panitia, pembuat artikel)"),
    "522151": (r"HONORARIUM.*(NARASUMBER|MODERATOR|PEMBAWA ACARA|ROHANIWAN|PENGAJAR|INSTRUKTUR|PENCERAMAH)", None, "jasa profesi: narasumber/moderator/pembawa acara/rohaniwan, pengajar eksternal"),
    "521111": (r"KONSUMSI|SATPAM|PENGEMUDI|KEBERSIHAN|PRAMUBAKTI|ALIH DAYA", None, "konsumsi rapat rutin internal; tenaga alih daya rutin"),
    "521211": (r"KONSUMSI|BAHAN MAKANAN|PENGGANDAAN", None, "bahan: konsumsi kegiatan spesifik, bahan makanan, penggandaan"),
    "521119": (r"PAKAIAN DINAS", None, "pakaian dinas rutin pegawai"),
    "521219": (r"PAKAIAN DINAS|SERAGAM", None, "seragam petugas kegiatan khusus"),
    "512211": (r"LEMBUR", None, "uang lembur ASN"),
    "511129": (r"MAKAN", None, "uang makan pegawai ASN"),
    "522141": (r"SEWA", None, "sewa kendaraan, gedung, peralatan pendukung kegiatan"),
    "523111": (r"PEMELIHARAAN GEDUNG|KEBERSIHAN GEDUNG|PERAWATAN GEDUNG", None, "pemeliharaan gedung"),
    "523121": (r"PEMELIHARAAN KENDARAAN|SERVIS|BAHAN BAKAR", None, "pemeliharaan kendaraan dinas"),
    "524111": (r"UANG HARIAN|REPRESENTASI|PENGINAPAN|TIKET|TAKSI", r"LUAR NEGERI", "perjalanan dinas biasa dalam negeri: uang harian, representasi, penginapan, tiket, taksi"),
    "524113": (r"TRANSPOR", r"LUAR NEGERI", "transpor lokal / perjalanan dinas dalam kota"),
    "524114": (r"RAPAT/PERTEMUAN|UANG HARIAN|PENGINAPAN", r"LUAR NEGERI", "paket meeting (halfday, fullday, fullboard, residence); dalam/luar kota mengikuti SBM"),
    "524119": (r"RAPAT/PERTEMUAN|UANG HARIAN|PENGINAPAN", r"LUAR NEGERI", "paket meeting luar kota (halfday, fullday, fullboard, residence)"),
}
ROLE_RE = re.compile(r"\b(penanggung jawab|ketua|wakil ketua|sekretaris|anggota|pengarah|koordinator)\b")
FRASA_UTAMA = ("uang harian", "uang representasi")
PANITIA_RE = re.compile(r"narasumber|moderator|panitia|pembawa acara|peserta|pembicara|penyuluh|ujian|diklat|jurnal|penelitian|komite|pengadaan|seleksi|butir soal|penulis")

# Daftar akun resmi: 524111 = perjalanan dinas biasa dalam negeri (luar kota hanya bila uraian menyebut provinsi);
# 524114 dan 524119 = paket meeting (tarif mengikuti SBM, dalam/luar kota tidak dipaksa oleh akun);
# 524113 = perjalanan dinas dalam kota; 524211 = luar negeri (MATA_UANG_ASING).
AKUN_RUTE = {"524119": "luar kota", "524113": "dalam kota"}


def nilai_item(uraian, hs, recs, vocab, induk="", rute="", akun="", wil_override=None):
    # luar/dalam kota: 524119 luar kota, 524113 dalam kota; 524111 luar kota bila ada provinsi (lihat _nilai_item_core)
    if str(akun) in AKUN_RUTE:
        rute = AKUN_RUTE[str(akun)]
    r = _nilai_item_core(uraian, hs, recs, vocab, induk=induk, rute=rute, akun=akun, wil_override=wil_override)
    if str(akun).startswith("5242") or re.search(r"\bluar negeri\b", norm_text(uraian)):
        r.update(status="MANUAL", kode="MATA_UANG_ASING", padanan="-", ref="-",
                 alasan=_alasan("MATA_UANG_ASING"))
    return r

def route_kota(kel):
    """Dari judul kelompok 'ASAL - TUJUAN, PROVINSI' (mis. 'JAKARTA - KOTA TANGERANG, BANTEN'):
    asal dan tujuan sama -> 'dalam kota'; berbeda -> 'luar kota'; tidak ada pola -> ''."""
    m = re.match(r"\s*(.+?)\s+-\s+(.+?)\s*$", str(kel or ""))
    if not m:
        return ""
    asal = region_parts(m.group(1).split(",")[0])[0]
    tujuan = region_parts(m.group(2).split(",")[0])[0]
    if not asal or not tujuan:
        return ""
    return "dalam kota" if asal == tujuan else "luar kota"

TAG_GROUPS = {   # multi-label tagging: kelompok -> {tag: pola pada teks ter-normalisasi}
    "KONSUMSI": {"halfday": r"\bhalf ?day\b", "fullday": r"\bfull ?day\b", "fullboard": r"\bfull ?board\b", "residence": r"\bresidence\b"},
    "MAKAN": {"makan": r"\bmakan\b|\blunch\b|\bdinner\b", "kudapan": r"\bkudapan\b|\bsnack\b"},
    "MODA": {"roda4": r"\broda (?:4|empat)\b", "roda2": r"\broda (?:2|dua)\b", "pjr": r"\bpjr\b", "minibus": r"\bminibus\b",
             "bus": r"\bbus\b", "pickup": r"\bpick up\b", "sedan": r"\bsedan\b", "gardan": r"\bdouble gardan\b"},
}

def _tag_teks(teks):
    """Himpunan (kelompok, tag) yang terdeteksi pada teks uraian atau kunci SBM."""
    t = norm_text(teks)
    return {(g, tag) for g, d in TAG_GROUPS.items() for tag, pola in d.items() if re.search(pola, t)}

def _tag_konflik(kandidat, un):
    """True bila uraian menyebut tag dalam sebuah kelompok yang tidak dimiliki kandidat SBM (dan sebaliknya)."""
    ct, ut = _tag_teks(kandidat.get("kunci", "")), _tag_teks(un)
    for g in TAG_GROUPS:
        cg = {t for gg, t in ct if gg == g}
        ug = {t for gg, t in ut if gg == g}
        if cg and ug and not (cg & ug):
            return True
    return False

def _kurang_rentang(pilih, uraian, wil, canon):
    """Alasan: uraian belum menyebut apa yang membedakan kandidat tarif (jenis rapat, snack/makan, provinsi, eselon)."""
    un = norm_text(uraian)
    kurang = []
    judul_all = " ".join((r.get("judul") or "") + " " + " ".join(t for _, _, t in r["anc"]) for r in pilih).upper()
    if "KONSUMSI" in judul_all and not re.search(r"diklat|pelatihan|tingkat menteri|menteri|eselon i\b|eselon 1|setingkat", un):
        kurang.append("jenis rapat (rapat koordinasi tingkat menteri/eselon I, rapat biasa, atau diklat)")
    kunci = {norm_text(r.get("kunci", "")) for r in pilih} - {""}
    if "KONSUMSI" in judul_all and len(kunci) > 1 and not re.search(r"snack|kudapan|makan", un):
        kurang.append("snack atau makan")
    provs = {c for r in pilih for c, _ in r["regions"]}
    if len(provs) > 1 and not wil["ids"]:
        kurang.append("provinsi")
    tiers = {tuple(sorted(r["tier"])) for r in pilih if r["tier"]}
    if len(tiers) > 1 and not canon:
        kurang.append("eselon/golongan")
    if not kurang:
        return ""
    return "Uraian belum menyebut " + ", ".join(kurang) + ", sehingga satu tarif tidak dapat dipilih."


UANG_KAT_JUDUL = {"HARIAN": "UANG HARIAN PERJALANAN DINAS DALAM NEGERI", "REPRESENTASI": "UANG REPRESENTASI PERJALANAN DINAS DALAM NEGERI"}
UANG_KAT_KUNCI = {"DIKLAT": r"diklat", "DALAM": r"dalam kota", "LUAR": r"luar kota"}

def _kategori_uang(uraian, akun, vocab, wil, recs):
    """Uang harian / uang representasi dibedakan menurut SBM:
    LUAR KOTA, DALAM KOTA (> 8 jam), atau DIKLAT. Akun 524119 = luar kota, 524113 = dalam kota,
    524111 = luar kota bila ada provinsi. Uang harian fullboard = paket rapat di luar kantor tanpa padanan.
    Mengembalikan (tipe, kategori, kandidat SBM dengan kategori tersebut)."""
    un = norm_text(uraian)
    un2 = re.sub(r"^[^a-z]+", "", un)
    if not (un2.startswith("uang") or re.search(r"\buang\b", un)):   # "uang" boleh di mana saja, mis. "Pemberian uang saku fulboard"
        return None, None, []
    if re.search(r"\bfullboard\b", un):      # satu-satunya kebijakan: Uang Harian Kegiatan Rapat/Pertemuan di Luar Kantor
        pool = [r for r in recs if "UANG HARIAN KEGIATAN RAPAT" in (r.get("judul") or "").upper()]
        fb = [r for r in pool if re.search(r"fullboard", norm_text(r.get("kunci", "")))]
        return "PAKET", "FULLBOARD", fb or pool
    if not un2.startswith(("uang harian", "uang representasi")):
        return None, None, []
    tipe = "REPRESENTASI" if un2.startswith("uang representasi") else "HARIAN"
    prov_stated = any("prov" in vocab.get(c, set()) for c, _ in wil["ids"])
    if re.search(r"\bdiklat\b", un):
        kat = "DIKLAT"
    elif re.search(r"\bdalam kota\b", un):
        kat = "DALAM"
    elif re.search(r"\bluar kota\b", un):
        kat = "LUAR"
    elif str(akun) == "524119":
        kat = "LUAR"
    elif str(akun) in ("524113", "524114"):   # 524114 termasuk dalam kota / diklat
        kat = "DALAM"
    elif str(akun) == "524111" and prov_stated:
        kat = "LUAR"
    else:
        kat = None
    if kat is None:
        return tipe, None, []
    pool = [r for r in recs if UANG_KAT_JUDUL[tipe] in (r.get("judul") or "").upper()
            and re.search(UANG_KAT_KUNCI[kat], norm_text(r.get("kunci", "")))]
    return tipe, kat, pool


def _filter_uang_harian(cands, recs, uraian_norm, has_wil, akun=""):
    """Uang harian dibedakan: perjalanan dinas dalam negeri (ada wilayah di uraian/kelompok) vs
    paket kegiatan rapat/pertemuan di luar kantor (tanpa wilayah; biasanya ada kata fullboard/fullday/halfday/residence)."""
    if not re.sub(r"^[^a-z]+", "", uraian_norm).startswith("uang harian"):   # hanya bila uraian ITEM diawali 'uang harian'
        return cands
    paket_kata = re.search(r"fullboard|fullday|halfday|residence|paket|rapat|pertemuan|meeting", uraian_norm)
    if str(akun) in ("524114", "524119"):             # akun paket meeting (rapat/pertemuan di luar kantor)
        need = "PAKET"
    elif str(akun) in ("524111", "524113"):           # akun perjalanan dinas
        need = "PERJALANAN"
    elif has_wil:
        need = "PERJALANAN"
    elif paket_kata:
        need = "PAKET"
    else:
        return cands
    allow_a = AKUN_RINCIAN_SBM.get(str(akun), (None, None, ""))[0]
    pool = [r for r in recs if "UANG HARIAN" in (r.get("judul") or "").upper()
            and (not allow_a or re.search(allow_a, (r.get("judul") or "").upper() + " " + " > ".join(t for _, _, t in r["anc"] if t).upper()))]
    def tag(r):
        j = (r.get("judul") or "").upper()
        return "PAKET" if "UANG HARIAN KEGIATAN RAPAT" in j else "PERJALANAN"
    kept = [r for r in pool if tag(r) == need]
    kept_bt = [r for r in kept if r["jenis"] == "batas_tertinggi"]
    return kept_bt or kept or cands


def _filter_jenis_konsumsi(cands, recs, uraian_norm, akun=""):
    """Konsumsi rapat dibedakan per jenis (SBM 38.1 koordinasi tingkat menteri/eselon I, 38.2 rapat biasa, diklat).
    Kandidat diambil dari seluruh SBM konsumsi (sesuai akun RAB), bukan hanya dari kandidat awal."""
    if not re.search(r"konsumsi|snack|makan|kudapan", uraian_norm):
        return cands
    allow_a = AKUN_RINCIAN_SBM.get(str(akun), (None, None, ""))[0]
    pool = [r for r in recs if "SATUAN BIAYA KONSUMSI" in (r.get("judul") or "").upper()
            and (not allow_a or re.search(allow_a, " > ".join(t for _, _, t in r["anc"] if t).upper()) or re.search(allow_a, (r.get("judul") or "").upper()))]
    if not pool:
        return cands
    if re.search(r"diklat|pelatihan|pendidikan", uraian_norm):
        need = "DIKLAT"
    elif re.search(r"koordinasi|menteri|eselon i\b|setingkat", uraian_norm):
        need = "KOORDINASI"
    else:
        need = "BIASA"
    def tag(r):
        j = (r.get("judul") or "").upper()
        if "DIKLAT" in j:
            return "DIKLAT"
        if "RAPAT KOORDINASI TINGKAT" in j:
            return "KOORDINASI"
        return "BIASA"
    kept = [r for r in pool if tag(r) == need]
    kept_bt = [r for r in kept if r["jenis"] == "batas_tertinggi"]   # Lampiran I diutamakan bila ada
    return kept_bt or kept or cands


def _nilai_item_core(uraian, hs, recs, vocab, induk="", rute="", akun="", wil_override=None):
    """Satu item RAB -> dict alasan lengkap. Tidak ada item yang dilewati."""
    REGION_TOK = {stem(w) for c in vocab for w in c.split()}
    ut = tokens_sbm(uraian) | tokens_sbm(rute)   # rute (luar/dalam kota) dari judul kelompok ikut dibaca
    wil = parse_wilayah(uraian, vocab)
    if wil_override is not None:   # provinsi sudah ditentukan di luar (uraian item, judul kelompok, atau label lokasi)
        wil = {"ids": [(c, "prov") for c in wil_override], "ambigu": False, "banyak": len(wil_override) > 1}
    elif "," in uraian:
        _ptoks = norm_text(uraian.rsplit(",", 1)[-1]).split()
        _pv = [c for c in provinsi_set(vocab) if _contains(_ptoks, c.split())]
        _pv = [c for c in _pv if not any(c != o and _contains(o.split(), c.split()) for o in _pv)]
        if _pv:   # provinsi ditulis setelah koma: itu yang dipakai, nama asal/tujuan kota diabaikan
            wil = {"ids": [(c, "prov") for c in _pv], "ambigu": False, "banyak": len(_pv) > 1}
    # wilayah = provinsi (kebijakan SBM): bila ada kecocokan provinsi, kecocokan kab/kota diabaikan
    _provs = [(c, t) for c, t in wil["ids"] if "prov" in vocab.get(c, set())]
    if _provs and len(_provs) < len(wil["ids"]) + 1 and any("prov" in vocab.get(c, set()) for c, _ in wil["ids"]):
        wil = {"ids": _provs, "ambigu": False, "banyak": len(_provs) > 1}
    canon = canon_tier(uraian)
    wil_tokens = {stem(w) for c, _ in wil["ids"] for w in c.split()}
    tier_tokens = {stem(w) for c in canon for w in c.split()}

    _un = norm_text(uraian)

    _aturan = AKUN_RINCIAN_SBM.get(str(akun), (None, None, ""))
    _allow_c = re.compile(_aturan[0]) if _aturan[0] else None
    _deny_c = re.compile(_aturan[1]) if _aturan[1] else None

    def _akun_ok(r):
        if not (_allow_c or _deny_c):
            return True
        _cache_r = r.setdefault("_akun_ok", {})      # cache disimpan di rekaman itu sendiri (aman antar katalog)
        if str(akun) in _cache_r:
            return _cache_r[str(akun)]
        jr = r.get("_jr")                       # teks judul per rekaman di-cache (dihitung sekali)
        if jr is None:
            jr = " > ".join([t for _, _, t in r["anc"] if t] + [r.get("kunci", "")]).upper()
            r["_jr"] = jr
        ok = True
        if _allow_c and not _allow_c.search(jr):
            ok = False
        elif _deny_c and _deny_c.search(jr):
            ok = False
        _cache_r[str(akun)] = ok
        return ok
        if str(akun) == "521213" and ROLE_RE.search(_un) and not PANITIA_RE.search(_un) and "TIM PELAKSANA" not in jr:
            return False
        if str(akun) == "521213" and re.search(r"narasumber|moderator|pembawa acara|rohaniwan|pengajar|instruktur|penceramah", _un):
            return False          # narasumber/pengajar eksternal masuk 522151 (daftar_padanan_sbm_kep331.txt)
        return True
    cara, cands, matched = "", [], set()
    ctx_banyak, ctx_titles = False, []
    # frasa utama yang kata-katanya ada di SBM_STOP (mis. 'uang harian'): dipakai sebagai kata utama
    frasa_u = next((f for f in FRASA_UTAMA if f in norm_text(uraian)), "")
    head = stem(frasa_u.split()[-1]) if frasa_u else next((stem(t) for t in norm_text(uraian).split()
                 if len(t) > 2 and t not in SBM_STOP and t not in GENERIK and not re.fullmatch(r"\d+\w*", t)), "")
    # keputusan reviewer (sbm_override.json): uraian langsung dipetakan ke nomor SBM, pemeriksaan wilayah/eselon dilewati
    key = norm_uraian(uraian)
    paksa = key in SBM_OVERRIDE
    if paksa:
        no_o = SBM_OVERRIDE[key]
        cands = [r for r in recs if r["no"] == no_o or r["grp"] == no_o]
        cara = "override"
    scored = []
    for r in ([] if paksa else recs):
        if not _akun_ok(r):
            continue
        # alternatif yang hanya berisi kata generik (mis. 'jasa') atau nama wilayah tidak boleh berdiri sendiri sebagai padanan
        okalt = [a for a in r["alts"] if a and a <= ut and not a <= GENERIK and not a <= REGION_TOK]
        best = max((len(a) for a in okalt), default=0)
        if best:
            scored.append((best, r))
    kat_ok = False
    if scored:
        top = max(s for s, _ in scored)
        cands = [r for s, r in scored if s == top]
        cara = "kategori"
        matched = max((a for a in cands[0]["alts"] if a and a <= ut and not a <= GENERIK and not a <= REGION_TOK), key=len)
        # kata utama uraian harus ikut cocok; jika tidak, kecocokan kategori dianggap tidak kuat
        kat_ok = (not head) or head in matched
        if not kat_ok:
            cands, cara, matched = [], "", set()
    if not cands and not paksa and head:
        # konteks: judul induk yang memuat kata utama uraian (mis. 'konsumsi'); ambil kedalaman terdalam
        # kelompok = satu judul induk tertentu (prefiks rantai judul); ambil kedalaman terdalam yang memuat kata utama
        # kelompok valid: kata utama uraian = kata pertama yang bermakna pada judul (bukan judul daftar)
        groups = {}
        # aturan: wilayah domestik memilih kelompok dalam negeri (kelompok yang judulnya memuat 'luar' dibuang)
        domestik = any(c in vocab and (vocab[c] & {"prov", "kab", "kabkota"}) for c, _ in wil["ids"])
        _sig = [t for t in norm_text(uraian).split() if len(t) > 2 and t not in SBM_STOP and t not in GENERIK and not re.fullmatch(r"\d+\w*", t)]
        frasa = " ".join(_sig[:2]) if len(_sig) >= 2 else ""   # dua kata utama berurutan, mis. 'uang harian'
        for r in recs:
            if not _akun_ok(r):
                continue
            for depth, (_, toks, jt) in enumerate(r["anc"]):
                if domestik and "luar" in toks and "negeri" in toks:   # hanya "luar negeri"; "luar kantor" tetap
                    continue
                _frasa_ok = bool(frasa_u) and (" " + frasa_u + " ") in (" " + norm_text(jt) + " ")
                if (head in toks and _first_sig(jt) == head) or _frasa_ok:
                    prefix = tuple(k for k, _, _ in r["anc"][:depth + 1])
                    groups.setdefault((depth, prefix), []).append((r, depth))
        if groups:
            # kelompok paling spesifik (terdalam) yang memuat kata utama
            mind = max(d for d, _ in groups)
            picked = [(k, v) for k, v in groups.items() if k[0] == mind]
            # kata lain di uraian yang juga ada di judul kelompok dipakai untuk memilih (mis. 'rapat')
            ut_induk = ut | tokens_sbm(induk)   # konteks induk (akun RAB) ikut membantu memilih kelompok
            # aturan: wilayah domestik memilih kelompok dalam negeri -> buang kelompok yang judulnya memuat 'luar'
            ut_raw = {t for t in norm_text(uraian).split() if len(t) > 2 and not t.isdigit()}
            def _skor(item):
                (d, _), v = item
                jt = v[0][0]["anc"][d][2]
                raw = {t for t in norm_text(jt).split() if len(t) > 2}
                return len(raw & ut_raw) + len(v[0][0]["anc"][d][1] & ut_induk)
            best = max(_skor(p) for p in picked)
            if best > 1:
                picked = [p for p in picked if _skor(p) == best]
            cands = [r for _, v in picked for r, _ in v]
            ctx_titles = sorted({" > ".join(t for _, _, t in v[0][0]["anc"][mind:]) for _, v in picked})
            cara = "konteks"
            matched = {head}
            if len(picked) > 1:
                ctx_banyak = True
    if not cands and not paksa:
        lead = next((t for t in norm_text(uraian).split() if len(t) > 2), "")
        grup = {}
        for r in recs:
            if r["grp_tokens"] and r["grp_tokens"] <= ut and lead in r["grp_tokens"]:
                grup.setdefault(r["grp"], []).append(r)
        if grup:
            best_g = max(grup, key=lambda k: len(grup[k][0]["grp_tokens"]))
            cands = grup[best_g]
            cara = "grup"
            matched = set(cands[0]["grp_tokens"]) & ut

    tipe_uang, jenis_uang, pool_uang = (None, None, [])
    if not paksa:
        cands = _filter_jenis_konsumsi(cands, recs, norm_text(uraian), akun)
        tipe_uang, jenis_uang, pool_uang = _kategori_uang(uraian, akun, vocab, wil, recs)
        if pool_uang and canon:       # eselon/golongan di uraian: pilih tarif yang bertingkat sama bila ada
            tepat_tier = [r for r in pool_uang if r["tier"] and (canon & r["tier"])]
            if tepat_tier:
                pool_uang = tepat_tier
        if pool_uang:
            cands = pool_uang          # kandidat uang harian/representasi sesuai kategori luar kota/dalam kota/diklat
            ctx_banyak = False         # kategori uang menggantikan pencarian konteks
    uang_kat = bool(pool_uang) and not paksa
    kunci_pakai = {k for r in cands for k in r["kunci_tok"]} | {stem(k) for r in cands for k in r["kunci_tok"]}
    _asli = {}
    for _t in norm_text(uraian).split():
        _asli.setdefault(stem(_t), _t)
    _penunjuk = {"provinsi", "kota", "kabupaten", "kab"} | tokens_sbm(rute)
    tidak_dikenali = sorted({_asli.get(x, x) for x in (ut - matched - wil_tokens - tier_tokens - kunci_pakai - {head} - _penunjuk - ({stem("fullboard")} if jenis_uang == "FULLBOARD" else set()))})
    base = {"cara": cara or "-", "wilayah": [c for c, _ in wil["ids"]], "eselon": sorted(canon),
            "tidak_dikenali": tidak_dikenali, "kandidat": [], "no_sbm": "-", "lampiran": "-",
            "ref": "-", "padanan": "-", "wilayah_txt": "", "besaran_sbm": "-"}
    if wil["ids"]:
        base["wilayah_txt"] = " / ".join(sorted({c for c, _ in wil["ids"]}))

    def done(status, kode, **extra):
        base.update(extra)
        base["status"], base["kode"] = status, kode
        base["alasan"] = _alasan(kode, hs=f"{float(hs):,.0f}" if hs is not None else "-",
                                 ref=base["ref"], padanan=base["padanan"], lampiran=base["lampiran"],
                                 wilayah=(f" (wilayah {base['wilayah_txt']})" if base["wilayah_txt"] else ""),
                                 rute=rute or "-", akun=str(akun), rincian_akun=AKUN_RINCIAN_SBM.get(str(akun), (None, None, "-"))[2])
        return base

    _uh = re.sub(r"^[^a-z]+", "", norm_text(uraian))
    if tipe_uang and jenis_uang is None and not paksa:
        return done("MANUAL", "UANG_JENIS_TIDAK_JELAS", padanan="-")
    if tipe_uang and jenis_uang and not pool_uang and not paksa:
        return done("MANUAL", "PADANAN_TIDAK_ADA", padanan="-")

    if not cands:
        return done("MANUAL", "PADANAN_TIDAK_ADA", padanan=(f"override SBM {SBM_OVERRIDE[key]} tidak ada di file" if paksa else "-"))

    if cara == "konteks":
        padanan_txt = " / ".join(ctx_titles[:3])[:200]
    else:
        padanan_txt = " / ".join(sorted({(r["judul"] or r["grp_judul"])[:80] for r in cands[:3]}))
    base["padanan"] = padanan_txt
    base["no_sbm"] = ", ".join(sorted({r["no"] for r in cands}))

    # Kandidat yang MUNGKIN berlaku: dimensi yang disebut di uraian menyaring; dimensi yang tidak disebut tetap terbuka.
    possible, kendala, belum = [], Counter(), []
    # kata kunci konsumsi (snack/kudapan, makan) ada di SBM_STOP sehingga tidak ada di ut; dibaca langsung dari uraian
    _nu = set(norm_text(uraian).split())
    _kunci_u = {"kudapan", "makan"} & _nu
    def _kw(r):
        return set(norm_text(r.get("kunci", "")).split())
    mention_k = any(r["kunci_tok"] & ut for r in cands)
    if ctx_banyak:
        return done("MANUAL", "KONTEKS_BANYAK", padanan=padanan_txt)
    if str(akun) == "524111" and not rute and any("prov" in vocab.get(c, set()) for c, _ in wil["ids"]):   # provinsi tertulis di uraian = luar kota
        rute = "luar kota"
    prov_set = provinsi_set(vocab)
    for r in cands:
        if paksa:
            possible.append(r)
            continue
        need_w, need_e = bool(r["regions"]), bool(r["tier"])
        w_stated = bool(wil["ids"])
        e_stated = bool(canon)
        need_prov = bool(r["regions"]) and all(t == "prov" for _, t in r["regions"])
        prov_stated = any(c in prov_set for c, _ in wil["ids"])
        # SBM yang mengatur wilayah hanya cocok bila uraian/lokasi menyebut PROVINSI (kab/kota saja tidak cukup)
        w_ok = (not need_w) or (prov_stated and not wil["ambigu"] and not wil["banyak"]
                                and region_cocok(wil["ids"], r["regions"]))
        w_ok = w_ok and (not need_prov or prov_stated)
        e_ok = (not need_e) or (not e_stated) or bool(canon & r["tier"])
        if _kunci_u and (_kw(r) & {"kudapan", "makan"}):   # konsumsi: snack atau makan harus disebut di uraian
            k_ok = bool(_kw(r) & _kunci_u)
        else:
            k_ok = (not r["kunci_tok"]) or bool(r["kunci_tok"] & ut) or (not mention_k)
        allow_akun, deny_akun, _ket = AKUN_RINCIAN_SBM.get(str(akun), (None, None, ""))
        if (allow_akun or deny_akun) and not uang_kat:
            jalur_r = " > ".join([t for _, _, t in r["anc"]] + [r.get("kunci", "")]).upper()
            tolak = (allow_akun and not re.search(allow_akun, jalur_r)) or (deny_akun and re.search(deny_akun, jalur_r))
            if str(akun) == "521213" and ROLE_RE.search(_un) and not PANITIA_RE.search(_un) and "TIM PELAKSANA" not in jalur_r:
                tolak = True   # peran (ketua/anggota/...) pada akun honor output: tim pelaksana, bukan panitia/narasumber
            if tolak:
                kendala["AKUN_TIDAK_SESUAI"] += 1
                continue
        kata_kunci = set(norm_text(r.get("kunci", "")).split())   # kata mentah (termasuk 'dalam', yang ada di SBM_STOP)
        rute_ok = not ((rute == "luar kota" and "dalam" in kata_kunci) or (rute == "dalam kota" and "luar" in kata_kunci))
        if not rute_ok:
            kendala["RUTE_TIDAK_COCOK"] += 1
            continue
        if k_ok and _tag_konflik(r, _un):
            kendala["TAG_TIDAK_COCOK"] += 1
            continue
        if w_ok and e_ok and k_ok:
            possible.append(r)
            if need_w and not w_stated:
                belum.append("WILAYAH_TIDAK_TERDETEKSI")
            if need_e and not e_stated:
                belum.append("ESELON_TIDAK_TERDETEKSI")
            if r["kunci_tok"] and not (r["kunci_tok"] & ut):
                belum.append("KATEGORI_TIDAK_TERDETEKSI")
            continue
        if not w_ok:
            if need_prov and not prov_stated:
                kode = "PROVINSI_TIDAK_DISEBUT"
            elif not wil["ids"]:
                kode = "WILAYAH_TIDAK_TERDETEKSI"
            elif wil["ambigu"]:
                kode = "WILAYAH_AMBIGU"
            elif wil["banyak"]:
                kode = "WILAYAH_BANYAK"
            elif region_tipe_beda(wil["ids"], r["regions"]):
                kode = "WILAYAH_TIPE_BEDA"
            else:
                kode = "WILAYAH_TIDAK_DIATUR"
        elif not e_ok:
            kode = "TIER_TIDAK_COCOK"
        else:
            kode = "KATEGORI_TIDAK_TERDETEKSI"
        kendala[kode] += 1

    base["kandidat"] = [{"no": r["no"], "judul": r["judul"][:80], "kunci": r.get("kunci", ""),
                         "jalur": " > ".join([t for _, _, t in r["anc"] if t] + ([r["kunci"]] if r.get("kunci") else [])),
                         "besaran": r.get("besaran"), "lampiran": r["jenis"]} for r in cands[:5]]
    if cara == "konteks" and possible:
        # padanan yang ditampilkan hanya kelompok dari kandidat yang masih mungkin berlaku
        padanan_txt = " / ".join(sorted({" > ".join(t for _, _, t in r["anc"]) for r in possible})[:3])[:200]
        base["padanan"] = padanan_txt
    if not possible:
        kode = kendala.most_common(1)[0][0] if kendala else "PADANAN_TIDAK_ADA"
        return done("MANUAL", kode)

    if hs is None:
        return done("MANUAL", "HARGA_KOSONG", padanan=padanan_txt)
    if any(r["tidak_dapat_dibaca"] for r in possible):
        return done("MANUAL", "TARIF_TIDAK_DIBACA", padanan=padanan_txt)

    if any(r["jenis"] == "dapat_dilampaui" for r in possible):
        return done("MANUAL", "DAPAT_DILAMPAUI", padanan=padanan_txt)
    pilih = _pilih_jenis(possible)
    jenis = pilih[0]["jenis"]
    base["lampiran"] = "Batas tertinggi" if jenis == "batas_tertinggi" else "Dapat dilampaui"
    base["no_sbm"] = ", ".join(sorted({r["no"] for r in pilih}))
    refs = sorted({r["besaran"] for r in pilih})
    hs_f = float(hs)
    if len(refs) == 1:
        ref = refs[0]
        base["ref"] = f"{ref:,.0f}"
        base["besaran_sbm"] = f"Rp{ref:,.0f}"
        base["padanan"] = padanan_txt
        if hs_f <= ref:
            return done("LOLOS", "HARGA_SESUAI", padanan=padanan_txt)
        if jenis == "batas_tertinggi":
            return done("TIDAK", "HARGA_MELEBIHI_BATAS", padanan=padanan_txt)
        return done("LOLOS", "DAPAT_DILAMPAUI", padanan=padanan_txt)

    # beberapa kemungkinan tarif: pasti lolos di bawah semua tarif, pasti TIDAK di atas semua (hanya batas tertinggi)
    base["ref"] = f"{refs[0]:,.0f} s.d. {refs[-1]:,.0f}"
    base["besaran_sbm"] = f"Rentang Rp{refs[0]:,.0f} s.d. Rp{refs[-1]:,.0f} ({len(refs)} nilai)"
    base["padanan"] = padanan_txt
    kurang_txt = _kurang_rentang(pilih, uraian, wil, canon)
    # rentang tarif (lebih dari satu besaran) selalu MANUAL: uraian belum cukup spesifik untuk memilih satu tarif
    res = done("MANUAL", "KANDIDAT_BANYAK", padanan=padanan_txt)
    if kurang_txt:
        res["alasan"] = res["alasan"] + " " + kurang_txt
    return res

# %% [code] sel 31
def iter_items(node, akun=None, akun_ur="", path="$"):
    """Yield (kode_akun, uraian_akun, item, json_path) untuk semua item di struktur RAB."""
    if isinstance(node, list):
        for i, n in enumerate(node):
            yield from iter_items(n, akun, akun_ur, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return
    if node.get("tipe") == "akun":
        akun, akun_ur = str(node.get("kode", "")), node.get("uraian", "")
    for i, it in enumerate(node.get("items", []) or []):
        yield akun, akun_ur, it, f"{path}.items[{i}]"
    if "children" in node:
        yield from iter_items(node["children"], akun, akun_ur, f"{path}.children")

iv_items = rules_by_code.get("IV", {}).get("items", [])
def find_item(prefix):
    return next((i for i in iv_items if str(i.get("kriteria_kode", "")).upper().startswith(prefix)),
                {"kriteria_kode": prefix + ".1", "pertanyaan": "", "feedback_revisi": ""})

def make_iv_row(doc_name, item, status, alasan, feedback, metode, quote="", json_path=""):
    loc = {"source_file": doc_name, "json_path": json_path} if json_path else None
    return {
        "jenis_dokumen": "RAB", "file_sumber": doc_name, "section_code": "IV",
        "kriteria_kode": item.get("kriteria_kode"), "pertanyaan": item.get("pertanyaan", ""),
        "catatan": "" if status == "LOLOS" else (item.get("catatan_kriteria", item.get("catatan", "")) or ""),
        "feedback": "" if status == "LOLOS" else feedback,
        "metode_checklist": {"kode": metode, "deskripsi": metode,
                             "rule_hit_count": 0, "semantic_retrieval": False},
        "status": status, "alasan": alasan,
        "evidence": {"sentence": quote or None, "quote": quote or None,
                     "evidence_type": "source_fragment" if quote else None,
                     "location": loc, "retrieval_score": None},
        "versions": {"pipeline": PIPELINE_VERSION, "rules_version": RULES_VERSION,
                     "rules_sha256": RULES_HASH, "model_version": model_version,
                     "prompt_version": PROMPT_VERSION, "embedding_model": "not_used_key_value"},
        "review_required": status != "LOLOS",
        "evidence_pola": [], "candidate_evidence": [], "rule_hits": [],
        "legacy_regex_result": {}, "legacy_regex_evidence_chunks": [],
        "evidence_quote": quote or "",
        "detail_headers": [], "detail_rows": [],
    }

def with_detail(row, headers, detail_rows):
    row["detail_headers"] = headers
    row["detail_rows"] = detail_rows
    return row

# %% [markdown] sel 32
# ## Evaluasi IV.B
# %% [code] sel 33
def eval_ivb(doc_name, items, akun_locs, itB):
    hdr = ["No", "Kode akun", "Uraian RAB", "Uraian BAS (KEP-331)", "Hasil", "Keterangan", "Lokasi"]
    akun = {}
    for a, aur, it, p in items:
        if a and a not in akun:
            akun[a] = aur
    if not akun:
        return make_iv_row(doc_name, itB, "MANUAL",
                           "Tidak ada kode akun pada struktur RAB.", "", "IV.B-BAS")

    hasil, detail = [], []
    for no, (kode, ur_rab) in enumerate(akun.items(), 1):
        locs = akun_locs.get(kode, [])
        if kode not in BAS_INFO:
            st, ket, ur_bas = "TIDAK", "kode tidak ada di BAS KEP-331", "-"
        elif BAS_INFO[kode]["rusak"]:
            st, ket, ur_bas = "MANUAL", "uraian BAS rusak atau kosong", BAS_INFO[kode]["uraian"] or "-"
        else:
            ur_bas = BAS_INFO[kode]["uraian"]
            st, ket = cek_uraian(ur_rab, ur_bas)
        hasil.append((st, ket, kode, ur_rab, locs[0] if locs else ""))
        detail.append([no, kode, ur_rab, ur_bas, st, ket, fmt_lokasi(locs)])

    tidak = [h for h in hasil if h[0] == "TIDAK"]
    manual = [h for h in hasil if h[0] == "MANUAL"]
    if tidak:
        st, ket, kode, ur, p = tidak[0]
        row = make_iv_row(doc_name, itB, "TIDAK",
                          f"{len(tidak)} akun tidak sesuai BAS KEP-331/PB/2021 (berlaku untuk semua TA). Contoh {kode}: {ket}.",
                          itB.get("feedback_revisi", ""), "IV.B-BAS", ur, p)
    elif manual:
        st, ket, kode, ur, p = manual[0]
        row = make_iv_row(doc_name, itB, "MANUAL",
                          f"{len(manual)} akun perlu verifikasi terhadap BAS KEP-331/PB/2021. Contoh {kode}: {ket}.",
                          "", "IV.B-BAS", ur, p)
    else:
        st, ket, kode, ur, p = hasil[0]
        row = make_iv_row(doc_name, itB, "LOLOS",
                          f"Seluruh {len(akun)} kode dan uraian akun sesuai BAS KEP-331/PB/2021 (berlaku untuk semua TA).",
                          "", "IV.B-BAS", ur, p)
    return with_detail(row, hdr, detail)

# %% [markdown] sel 34
# ## Evaluasi IV.C
# %% [code] sel 35
# Pemetaan provinsi memakai pustaka id-location-ner (pip install id-location-ner) dan gazetteer
# wilayah_indonesia.csv. Bila pustaka/CSV tidak tersedia, dipakai logika lama (_provinsi_dari_teks_lama).
def _siapkan_csv_wilayah(path: Path) -> Path | None:
    """CSV ber-delimiter ';' dikonversi ke ',' di CACHE_DIR (file sumber tidak ditimpa)."""
    if not path.is_file():
        log.warning("CSV wilayah tidak ditemukan: %s", path)
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = next(csv.reader(f), [])
    if len(first) == 1 and ";" in first[0]:
        out = cc.CACHE_DIR / f"{path.stem}_koma.csv"
        if not out.is_file() or out.stat().st_mtime < path.stat().st_mtime:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(path, encoding="utf-8-sig", newline="") as fi, open(out, "w", encoding="utf-8", newline="") as fo:
                csv.writer(fo).writerows(csv.reader(fi, delimiter=";"))
        return out
    return path


LOC_EXT = None
try:
    from location_ner.extractor import LocationExtractor
    _csv = _siapkan_csv_wilayah(cc.WILAYAH_CSV)
    if _csv is not None:
        LOC_EXT = LocationExtractor.from_csv(str(_csv), language="id", max_level="city")
except ImportError:
    log.warning("id-location-ner tidak terpasang; deteksi provinsi memakai logika lama")
except Exception as e:                      # gazetteer rusak tidak boleh menghentikan service
    log.warning("LocationExtractor gagal dimuat (%r); deteksi provinsi memakai logika lama", e)

# %% [code] sel 36
def parse_tahun(nilai):
    """Ambil tahun dari nilai apa pun, mis. 2026, "2026", "TA 2026", "2026/2027" -> 2026."""
    m = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(nilai or ""))
    return int(m.group(1)) if m else 0

def ivc_items_with_context(node, akun=None, akun_ur="", kel_ur="", path="$", judul_induk="", lokasi="", region_context=None):
    """Yield item beserta konteks wilayah seluruh induk, terdekat lebih dahulu."""
    region_context = list(region_context or [])
    if isinstance(node, list):
        heading, ctx_list = judul_induk, region_context
        for i, n in enumerate(node):
            # lokasi hanya berlaku untuk node itu sendiri (dan turunannya); tidak bocor ke saudara
            if (isinstance(n, dict) and n.get("tipe") == "kelompok" and not n.get("items")
                    and not any(isinstance(v, (list, dict)) for k, v in n.items() if k not in ("uraian", "tipe", "lokasi"))):
                heading = (heading + " > " if heading else "") + str(n.get("uraian", "") or "")
                if n.get("uraian"):
                    ctx_list = [(str(n["uraian"]), "uraian heading")] + ctx_list
                continue
            yield from ivc_items_with_context(n, akun, akun_ur, kel_ur, f"{path}[{i}]", heading, lokasi, ctx_list)
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("lokasi"), str):
        lokasi = node["lokasi"]
    local_context = []
    if lokasi:
        local_context.append((str(lokasi), "lokasi"))
    if node.get("uraian"):
        local_context.append((str(node["uraian"]), "uraian induk"))
    region_context = local_context + region_context
    if node.get("tipe") == "akun":
        akun, akun_ur, kel_ur, judul_induk = str(node.get("kode", "")), node.get("uraian", ""), "", ""
    elif node.get("tipe") == "kelompok":
        kel_ur = str(node.get("uraian", "") or "")
    for i, it in enumerate(node.get("items", []) or []):
        yield akun, akun_ur, it, f"{path}.items[{i}]", kel_ur, judul_induk, lokasi, region_context
    for k, v in node.items():
        if k != "items" and isinstance(v, (list, dict)):
            yield from ivc_items_with_context(v, akun, akun_ur, kel_ur, f"{path}.{k}", judul_induk, lokasi, region_context)

_KOTA_CACHE = {}

def kota_ke_provinsi(recs, provset):
    """Peta kab/kota -> provinsi dari struktur SBM (judul provinsi sebagai induk). Kota yang muncul di dua provinsi dibuang."""
    key = (id(recs), len(provset))
    if key in _KOTA_CACHE:
        return _KOTA_CACHE[key]
    peta, ambigu = {}, set()
    for r in recs:
        provs = {region_parts(txt)[0] for _no, _tok, txt in r.get("anc", ()) if txt} & provset
        if len(provs) != 1:
            continue
        prov = provs.pop()
        for c, _t in r.get("regions", ()):
            if not c or c in provset:
                continue
            if c in peta and peta[c] != prov:
                ambigu.add(c)
            peta.setdefault(c, prov)
    for c in ambigu:
        peta.pop(c, None)
    _KOTA_CACHE[key] = peta
    return peta

PROVINSI_ALIAS = {
    "jkt": ("jakarta",), "jkt pusat": ("jakarta",), "jkt selatan": ("jakarta",),
    "dki": ("jakarta",), "d k i": ("jakarta",), "dki jakarta": ("jakarta",),
    "d k i jakarta": ("jakarta",), "daerah khusus ibukota jakarta": ("jakarta",),
    "daerah khusus ibu kota jakarta": ("jakarta",), "prov jakarta": ("jakarta",),
    "provinsi jakarta": ("jakarta",), "diy": ("yogyakarta",), "d i y": ("yogyakarta",),
    "di yogyakarta": ("yogyakarta",), "d i yogyakarta": ("yogyakarta",),
    "daerah istimewa yogyakarta": ("yogyakarta",), "prov yogyakarta": ("yogyakarta",),
    "provinsi yogyakarta": ("yogyakarta",), "jabar": ("jawa barat",), "jateng": ("jawa tengah",),
    "jatim": ("jawa timur",), "sumut": ("sumatera utara",), "sumbar": ("sumatera barat",),
    "sumsel": ("sumatera selatan",), "babel": ("bangka belitung", "kepulauan bangka belitung"),
    "kepri": ("kepulauan riau",), "kep riau": ("kepulauan riau",),
    "kepaulauan riau": ("kepulauan riau",), "ntb": ("nusa tenggara barat",), "ntt": ("nusa tenggara timur",),
    "kalbar": ("kalimantan barat",), "kalteng": ("kalimantan tengah",), "kalsel": ("kalimantan selatan",),
    "kaltim": ("kalimantan timur",), "kaltara": ("kalimantan utara",), "sulut": ("sulawesi utara",),
    "sulteng": ("sulawesi tengah",), "sulsel": ("sulawesi selatan",), "sultra": ("sulawesi tenggara",),
    "sulbar": ("sulawesi barat",), "malut": ("maluku utara",), "papbar": ("papua barat",),
    "sumatra utara": ("sumatera utara",), "sumatra barat": ("sumatera barat",), "sumatra selatan": ("sumatera selatan",),
}

_PROVINSI_CORE_AMBIGU = set()   # Jakarta dan Yogyakarta dikenali sebagai provinsi (tidak diabaikan)

def _norm_wilayah(teks):
    return re.sub(r"[^a-z0-9]+", " ", str(teks or "").casefold()).strip()

def _provinsi_hits(teks, provset):
    clean = re.sub(r"\[[^\]]*\]", " ", str(teks or ""))
    clean = re.sub(r"[-–—→]+", " rute ", clean).replace("/", " slash ")
    nt = _norm_wilayah(clean)
    patterns = {}
    for prov in provset:
        if prov not in _PROVINSI_CORE_AMBIGU:
            patterns.setdefault(prov, set()).add(prov)
    for alias, targets in PROVINSI_ALIAS.items():
        for prov in provset:
            if any(prov == target or prov.endswith(" " + target) for target in targets):
                patterns.setdefault(prov, set()).add(alias)
    hits = []
    for prov, names in patterns.items():
        for name in names:
            match = re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", nt)
            if match:
                hits.append((match.start(), match.end(), prov))
    return [h for h in hits if not any(h[0] >= o[0] and h[1] <= o[1] and h[1] - h[0] < o[1] - o[0] for o in hits)]

# ---- Pencarian provinsi dengan kemiripan nama (fuzzy) pada uraian ----
# Daftar nama provinsi resmi; dicocokkan per frasa 1-4 kata dari uraian. Kecocokan tepat/alias tetap diprioritaskan.
FUZZY_PROVINSI = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau", "Jambi", "Sumatera Selatan",
    "Kepulauan Bangka Belitung", "Bengkulu", "Lampung", "DKI Jakarta", "Banten", "Jawa Barat", "Jawa Tengah",
    "DI Yogyakarta", "Jawa Timur", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Kalimantan Barat",
    "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara",
    "Gorontalo", "Sulawesi Tengah", "Sulawesi Barat", "Sulawesi Selatan", "Sulawesi Tenggara", "Maluku",
    "Maluku Utara", "Papua", "Papua Barat", "Papua Selatan", "Papua Tengah", "Papua Pegunungan", "Papua Barat Daya",
]
FUZZY_MIN = 85   # ambang kemiripan per frasa (skala 0-100); di bawah ini dianggap bukan nama provinsi
try:
    from thefuzz import fuzz as _fuzz
    def _rasio(a, b):
        return _fuzz.ratio(a, b)
except ImportError:                      # thefuzz tidak terpasang: pakai difflib (rumus sama, 0-100)
    import difflib
    def _rasio(a, b):
        return round(100 * difflib.SequenceMatcher(None, a, b).ratio())

def _nt_uraian(teks):
    """Teks uraian yang sudah dinormalisasi; posisi huruf di sini dipakai bersama _provinsi_hits."""
    t = re.sub(r"\[[^\]]*\]", " ", str(teks or "")).strip()
    return _norm_wilayah(re.sub(r"[-–—→]+", " rute ", t).replace("/", " slash "))

def _fuzzy_hits(teks, provset, sudah):
    """Frasa 1-4 kata yang mirip nama provinsi (>= FUZZY_MIN), tidak tumpang tindih dengan kecocokan tepat."""
    nt = _nt_uraian(teks)
    spans = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[a-z0-9]+", nt)]
    kanon = {}
    for nama in FUZZY_PROVINSI:
        core = region_parts(nama)[0]
        if core in provset:
            kanon[_norm_wilayah(nama)] = core
    if not kanon:                        # SBM tanpa daftar provinsi: tidak ada yang bisa dicocokkan
        return []
    kandidat = []
    for n in range(1, 5):
        for i in range(len(spans) - n + 1):
            frasa = " ".join(s[2] for s in spans[i:i + n])
            if frasa in ("rute", "slash", "dan", "atau"):
                continue
            nama_terbaik, skor = max(((k, _rasio(frasa, k)) for k in kanon),
                                     key=lambda kv: (kv[1], len(kv[0])))
            if skor >= FUZZY_MIN:
                kandidat.append((skor, spans[i][0], spans[i + n - 1][1], kanon[nama_terbaik]))
    diterima = []
    for skor, s, e, prov in sorted(kandidat, key=lambda k: (-k[0], -(k[2] - k[1]))):
        if any(s < e2 and s2 < e for s2, e2, _p in sudah + diterima):
            continue
        diterima.append((s, e, prov))
    return diterima

def _gabung_hits(teks, provset, vocab=None):
    """Kecocokan tepat/alias, ditambah kecocokan fuzzy. Nama provinsi yang sebenarnya bagian dari nama
    kab/kota di SBM (mis. 'Maluku' dalam 'Maluku Tengah') dibuang bila vocab dikirim."""
    tepat = _provinsi_hits(teks, provset)
    hits = tepat + _fuzzy_hits(teks, provset, tepat)
    if not vocab:
        return hits
    nt = _nt_uraian(teks)
    spans = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[a-z0-9]+", nt)]
    keluar = []
    for s, e, prov in hits:
        idx = [i for i, sp in enumerate(spans) if sp[0] == s]
        gabung = False
        if idx:
            i = idx[0]
            for k in (1, 2):   # frasa provinsi + 1-2 kata berikutnya yang membentuk nama kab/kota
                if i + k < len(spans):
                    nama = " ".join(sp[2] for sp in spans[i:i + k + 1])
                    if vocab.get(region_parts(nama)[0], set()) & {"kab", "kabkota", "kota"} and region_parts(nama)[0] != prov:
                        gabung = True
        if not gabung:
            keluar.append((s, e, prov))
    return keluar

def deteksi_provinsi(teks_input, provset=None):
    """Diagnosis cepat: nama provinsi yang terdeteksi pada satu teks uraian (untuk pengecekan manual)."""
    provset = provset or {region_parts(n)[0] for n in FUZZY_PROVINSI}
    hits = sorted(_gabung_hits(teks_input, provset))
    if not hits:
        return "Provinsi tidak dikenali"
    return ", ".join(h[2] for h in hits)

def _provinsi_dari_teks_lama(teks, vocab, provset, kota_map):
    """Ambil provinsi eksplisit; kota/kabupaten dan kawasan gabungan bukan provinsi."""
    t = re.sub(r"\[[^\]]*\]", " ", str(teks or "")).strip()
    if not t:
        return [], []
    nt = _norm_wilayah(re.sub(r"[-–—→]+", " rute ", t).replace("/", " slash "))
    hits = sorted(_gabung_hits(t, provset, vocab))
    by_name = {}
    for start, end, prov in hits:
        by_name.setdefault(prov, (start, end))
    if not by_name:
        return [], []
    ordered = sorted((start, end, prov) for prov, (start, end) in by_name.items())
    # tujuan berupa kota/kabupaten tanpa provinsi (mis. "Jakarta - Bogor"): provinsi asal bukan provinsi tujuan
    m_rute = re.search(r"\brute\b", nt)
    if m_rute and all(end <= m_rute.start() for _, end, _p in ordered):
        kanan = [w for w in nt[m_rute.end():].split() if len(w) > 2 and w not in ("prov", "provinsi", "kota", "kab", "kabupaten")]
        if kanan:
            return [], ["TUJUAN_TANPA_PROVINSI: " + ", ".join(p.title() for _, _, p in ordered)]
    if len(ordered) == 1:
        return [ordered[0][2]], []
    between = [nt[ordered[i][1]:ordered[i + 1][0]] for i in range(len(ordered) - 1)]
    if any(re.search(r"\b(slash|atau)\b", s) for s in between):
        return [], ["AMBIGU: " + ", ".join(p for _, _, p in ordered)]
    if any(re.search(r"\b(rute|dan|ke|menuju)\b", s) for s in between):
        return [ordered[-1][2]], []
    if re.search(r"\b(jabodetabek|bandung raya|nasional)\b", norm_text(t)):
        return [], []
    return [], ["AMBIGU: " + ", ".join(p for _, _, p in ordered)]

# ---------------------------------------------------------------------------
# Pemetaan provinsi berbasis id-location-ner (LOC_EXT dibuat di cell sebelum cell ini).
# Aturan: jika ada beberapa lokasi, ambil yang paling akhir muncul di teks.
# Jika library tidak menemukan lokasi atau namanya tidak cocok dengan SBM, pakai logika lama.
# ---------------------------------------------------------------------------
USE_LIB_PROVINSI = True

_PREFIX_PROV = ("daerah khusus ibukota", "daerah istimewa", "dki", "di", "provinsi", "prov")

def _cocokkan_provset(nama, provset):
    """Cocokkan nama provinsi dari library ke nama provinsi di SBM (provset)."""
    n = _norm_wilayah(nama)
    kandidat = [n]
    for pre in _PREFIX_PROV:
        if n.startswith(pre + " "):
            kandidat.append(n[len(pre) + 1:].strip())
    peta = {_norm_wilayah(c): c for c in provset}
    for k in kandidat:
        if k in peta:
            return peta[k]
    return None

# Singkatan umum nama kota/kabupaten (bukan nama provinsi). Diperluas sebelum pencocokan library.
SINGKATAN_KOTA = {
    "tangsel": "kota tangerang selatan",
    "jaksel": "kota jakarta selatan", "jakut": "kota jakarta utara", "jakbar": "kota jakarta barat",
    "jaktim": "kota jakarta timur", "jakpus": "kota jakarta pusat",
    "bekasi kota": "kota bekasi", "bekasi kab": "kabupaten bekasi",
}
_SINGKATAN_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(SINGKATAN_KOTA, key=len, reverse=True)) + r")\b", re.IGNORECASE)

def _perluas_singkatan(teks):
    return _SINGKATAN_RE.sub(lambda m: SINGKATAN_KOTA[m.group(1).casefold()], teks)

def _provinsi_dari_lib(teks, provset):
    """Kembalikan (provs, catatan) dari library, atau None bila library tidak menemukan lokasi yang cocok."""
    t = re.sub(r"\[[^\]]*\]", " ", str(teks or "")).strip()
    if not t or "LOC_EXT" not in globals() or LOC_EXT is None:
        return None
    # hanya provinsi dan kota/kabupaten; nama kecamatan/kelurahan sering sama di banyak provinsi
    ms = [x for x in LOC_EXT.extract(_perluas_singkatan(t), fuzzy=True, fuzzy_threshold=90) if x.level in ("province", "city")]
    if not ms:
        return None
    m = ms[-1]
    provs_lib = sorted({c["province_name"] for c in m.candidates})
    provs = sorted({_cocokkan_provset(p, provset) for p in provs_lib} - {None})
    if not provs:
        return None
    if len(provs) == 1:
        return provs, []
    return [], ["AMBIGU: " + ", ".join(provs)]

def provinsi_dari_teks(teks, vocab, provset, kota_map):
    """Pilih sumber provinsi: library id-location-ner dulu, lalu logika lama sebagai cadangan."""
    if USE_LIB_PROVINSI:
        hasil = _provinsi_dari_lib(teks, provset)
        if hasil is not None:
            return hasil
    return _provinsi_dari_teks_lama(teks, vocab, provset, kota_map)


def resolve_wilayah(uraian, kel, lok, vocab, provset, kota_map, region_context=None, item_lok=""):
    """Cari dari item, lalu induk terdekat sampai root; provinsi terdekat menjadi konteks tarif."""
    candidates = [(uraian, "uraian item"), (item_lok, "lokasi item")]
    candidates.extend(region_context or [])
    if not region_context:
        candidates.extend(((kel, "judul kelompok"), (lok, "label lokasi")))
    for teks, sumber in candidates:
        provs, catatan = provinsi_dari_teks(teks, vocab, provset, kota_map)
        if catatan and catatan[0].startswith(("AMBIGU:", "TUJUAN_TANPA_PROVINSI:")):
            return [], sumber, catatan
        if provs:
            return provs, sumber, catatan
    return [], "", []

def rute_asal_tujuan(kel):
    """Dari judul kelompok 'ASAL - TUJUAN, PROVINSI': kembalikan teks asal dan tujuan (kosong bila tidak ada pola)."""
    m = re.match(r"\s*(.+?)\s+-\s+(.+?)\s*$", str(kel or ""))
    if not m:
        return ""
    asal = m.group(1).strip()
    tujuan = m.group(2).strip()
    return f"Rute: asal {asal}, tujuan {tujuan} (tarif mengikuti provinsi tujuan)."


def fmt_jabatan(daftar):
    """'eselon ii' -> 'Eselon II', 'gol iv' -> 'Golongan IV', 'menteri' -> 'Menteri' (untuk kolom Jabatan)."""
    out = []
    for j in sorted(daftar or []):
        if j.startswith("eselon "):
            out.append("Eselon " + j[7:].upper())
        elif j.startswith("gol "):
            out.append("Golongan " + j[4:].upper())
        else:
            out.append(j.title())
    return ", ".join(out)

def fmt_nilai(v):
    return f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)

def keterangan_item(r):
    """Reasoning per item: alasan utama + cara pencocokan + kata yang tidak dikenali + kandidat yang dipertimbangkan."""
    bagian = [r["alasan"]]
    if r.get("cara") and r["cara"] != "-":
        bagian.append(f"Cara pencocokan: {r['cara']}.")
    if r.get("wilayah"):
        bagian.append(f"Wilayah uraian: {', '.join(r['wilayah'])}.")
    if r.get("tidak_dikenali") and r["status"] != "LOLOS":
        bagian.append(f"Kata tidak dikenali: {', '.join(r['tidak_dikenali'][:8])}.")
    if r.get("kandidat") and r["status"] in ("MANUAL", "DILEWATI"):
        ks = "; ".join(f"{k['no']} ({k.get('jalur', k['judul'])[:150]}; nilai {fmt_nilai(k['besaran'])})" for k in r["kandidat"][:3])
        bagian.append(f"Kandidat SBM: {ks}.")
    return " ".join(bagian)

def eval_ivc(doc_name, items, meta, itC):
    hdr = ["No", "Kode akun", "Uraian item", "Harga satuan RAB", "Kode alasan", "Wilayah (uraian)",
           "Jabatan (uraian)", "Padanan SBM", "No SBM", "Hasil", "Keterangan (reasoning)", "Besaran SBM (dibandingkan)"]
    raw_tahun = (meta or {}).get("tahun_anggaran")
    tahun = parse_tahun(raw_tahun)
    if tahun == 0:
        return make_iv_row(doc_name, itC, "MANUAL",
                           f"Tahun anggaran tidak terbaca dari metadata (nilai: {raw_tahun!r}), sehingga SBM tidak dapat ditentukan.",
                           "", "IV.C-SBM")
    if not items:
        return make_iv_row(doc_name, itC, "MANUAL",
                           "Tidak ada item RAB yang terbaca dari struktur JSON (cek kunci item dan children).",
                           "", "IV.C-SBM")
    tahun_sbm = resolve_tahun_sbm(tahun)
    bundle = get_sbm_bundle(tahun_sbm)
    if bundle is None:
        tersedia = ", ".join(str(t) for t in SBM_CONFIG)
        return make_iv_row(doc_name, itC, "MANUAL",
                           f"File SBM untuk TA {tahun_sbm} belum tersedia atau tidak terbaca "
                           f"(tersedia untuk TA: {tersedia}).", "", "IV.C-SBM")
    recs, vocab = bundle
    dasar = SBM_CONFIG[tahun_sbm]["dasar"]
    sbm_info = f"SBM yang digunakan: TA {tahun_sbm} ({dasar})."
    catatan = (" Pengecekan dilakukan terhadap SBM Lampiran I sebagai batas tertinggi."
               + ("" if tahun == tahun_sbm else f" (TA {tahun} diperiksa dengan SBM TA {tahun_sbm})"))

    reasons, det = [], []
    for no, t in enumerate(items, 1):
        a, aur, it, p = t[:4]
        kel = t[4] if len(t) > 4 else ""
        judul_induk = t[5] if len(t) > 5 else ""
        uraian = it.get("uraian", "")
        hs = it.get("harga_satuan")
        lok = t[6] if len(t) > 6 else ""                   # label lokasi (kelompok/akun induk), opsional
        provset = provinsi_set(vocab)
        kota_map = kota_ke_provinsi(recs, provset)
        region_context = t[7] if len(t) > 7 else []
        provs, sumber_w, catatan_w = resolve_wilayah(uraian, kel, lok, vocab, provset, kota_map,
                                                       region_context, it.get("lokasi", ""))
        wil_arg = provs                              # tanpa provinsi: wilayah kosong -> SBM yang mengatur wilayah tidak lolos
        uraian_eff = f"{uraian} {kel}".strip()   # kata utama tetap dari uraian item
        induk_ctx = (f"{aur} {judul_induk}").strip()   # akun + heading kelompok tanpa item
        r = nilai_item(uraian_eff, hs, recs, vocab, induk=induk_ctx, rute=route_kota(kel), akun=a, wil_override=wil_arg)   # akun RAB sebagai konteks induk
        if provs:                                              # sumber wilayah disebut di alasan
            r["alasan"] = f"{r['alasan']} Wilayah dibaca dari {sumber_w}: {', '.join(c.title() for c in provs)}."
            if catatan_w:
                r["alasan"] = f"{r['alasan']} ({'; '.join(catatan_w)}.)"
            _rt = rute_asal_tujuan(kel)
            if _rt:
                r["alasan"] = f"{r['alasan']} {_rt}"
        r.update(no=no, kode_akun=a, uraian=uraian, harga=hs, json_path=p, kelompok=kel)
        r["bukti"] = {"json_path": p, "substring": uraian}
        if catatan_w and catatan_w[0].startswith("TUJUAN_TANPA_PROVINSI:"):
            asal = catatan_w[0].partition(":")[2].strip()
            r.update(status="MANUAL", kode="PROVINSI_TIDAK_DISEBUT", wilayah=[], wilayah_txt="-", padanan="-",
                     alasan=(f"Tujuan perjalanan berupa kota/kabupaten tanpa provinsi (provinsi di uraian hanya asal: {asal}). "
                             "Tulis nama provinsi tujuan agar tarif per provinsi dapat dicocokkan."))
        if catatan_w and catatan_w[0].startswith("AMBIGU:"):
            kandidat_wilayah = catatan_w[0].partition(":")[2].strip()
            r.update(status="TIDAK", kode="PROVINSI_AMBIGU", wilayah=kandidat_wilayah.split(", "),
                     wilayah_txt=kandidat_wilayah, alasan=_alasan("PROVINSI_AMBIGU", wilayah=kandidat_wilayah))
        reasons.append(r)
        # semua item masuk tabel detail (termasuk LOLOS dan DILEWATI) agar bisa ditelusuri reviewer
        det.append([no, a, uraian, f"{float(hs):,.0f}" if hs is not None else "-", r["kode"],
                    r["wilayah_txt"] or "-", fmt_jabatan(r["eselon"]) or "-", r["padanan"],
                    r["no_sbm"], r["status"], keterangan_item(r), r.get("besaran_sbm", "-")])

    cnt = Counter(r["status"] for r in reasons)
    dicek = [r for r in reasons if r["status"] != "DILEWATI"]
    kode_cnt = Counter(r["kode"] for r in dicek if r["status"] != "LOLOS")
    ringkas = ", ".join(f"{k} {v}" for k, v in kode_cnt.most_common()) or "-"
    tidak = [r for r in reasons if r["status"] == "TIDAK"]
    dilewati = cnt["DILEWATI"]   # tidak lagi dipakai: uraian tanpa padanan menjadi MANUAL
    if not reasons:
        row = make_iv_row(doc_name, itC, "MANUAL", "Tidak ada item RAB.", sbm_info, "IV.C-SBM")
    elif not dicek:
        row = make_iv_row(doc_name, itC, "MANUAL",
                          f"Tidak ada item RAB yang dapat diperiksa.{catatan}", sbm_info, "IV.C-SBM")
    elif tidak:
        c = tidak[0]
        row = make_iv_row(doc_name, itC, "TIDAK",
                          f"{len(tidak)} item melebihi batas tertinggi {dasar}. Contoh baris {c['no']}: {c['alasan']} "
                          f"Rincian kode: {ringkas}.{catatan}",
                          (itC.get("feedback_revisi", "") + " " + sbm_info).strip(), "IV.C-SBM", c["uraian"], c["json_path"])
    elif cnt["MANUAL"]:
        row = make_iv_row(doc_name, itC, "MANUAL",
                          f"{cnt['LOLOS']} dari {len(dicek)} item yang diperiksa LOLOS; {cnt['MANUAL']} item MANUAL. "
                          f"Rincian kode: {ringkas}.{catatan}",
                          sbm_info, "IV.C-SBM")
    else:
        row = make_iv_row(doc_name, itC, "LOLOS",
                          f"Seluruh {len(dicek)} item yang diperiksa tidak melebihi batas {dasar}.{catatan}",
                          sbm_info, "IV.C-SBM")
    row = with_detail(row, hdr, det)
    row["iv_c_items"] = reasons
    row["iv_c_ringkas"] = dict(kode_cnt)
    return row


def ringkas_backlog(all_reasons):
    """Ringkasan alasan dan token yang paling sering tidak dikenali (dasar pengembangan checklist)."""
    kode = Counter(r["kode"] for r in all_reasons if r["status"] != "LOLOS")
    token = Counter(t for r in all_reasons if r["status"] != "LOLOS" for t in r["tidak_dikenali"])
    return kode.most_common(), token.most_common(30)

# %% [markdown] sel 37
# ## Evaluasi IV.D
# %% [code] sel 38
# ===== IV.D v6: honor (a) dan belanja bahan 521211 (b–f; yang tidak cocok dicatat sebagai catatan, tidak dihitung) =====
AKUN_HONORARIUM = {"521213", "522151"}   # seluruh item akun ini = a (honorarium)
AKUN_BELANJA_BAHAN = {"521211"}          # dirinci per uraian ke b–f; yang tidak cocok hanya dicatat (tidak dihitung)
# Label a–f dikunci di cell ini, sehingga tidak tergantung urutan eksekusi cell 28
KATEGORI_LABEL = {
    "a": "Honorarium kegiatan",
    "b": "Biaya konsumsi rapat dalam kantor",
    "c": "Biaya kegiatan seremonial",
    "d": "Biaya percetakan",
    "e": "Pengadaan souvenir",
    "f": "Pengadaan ATK untuk kegiatan",
}

def eval_ivd(doc_name, items, meta, itD):
    """IV.D (substansi): komponen a–f dijumlahkan terhadap total RAB.
    Total RAB = alokasi dana pada metadata.
    LOLOS  : a–f <= 30% total RAB (termasuk tepat 30%); sisa dihitung dan ditampilkan.
    TIDAK  : a–f > 30% total RAB (tidak ada sisa)."""
    hdr_detail = ["No", "Kode akun", "Uraian item", "Jumlah (Rp)", "Kategori", "Kata kunci", "Lokasi"]
    hdr_sum = ["Komponen", "Nilai (Rp)", "% dari total RAB"]

    total = float(meta.get("alokasi_dana") or 0)      # total RAB = alokasi dana pada metadata

    def pct(v):
        return f"{v / total:.2%}" if total > 0 else "-"

    per_kat = {k: 0.0 for k in KATEGORI_LABEL}
    detail, no = [], 0
    catatan_n, catatan_rp = 0, 0.0
    for a, aur, it, p in items:
        kat = klasifikasi_pendukung(a, it.get("uraian", ""))
        if not kat:
            if str(a) in AKUN_BELANJA_BAHAN and float(it.get("jumlah") or 0) > 0:
                catatan_n += 1
                catatan_rp += float(it.get("jumlah") or 0)
                no += 1
                detail.append([no, a, it.get("uraian", ""), f"{float(it.get('jumlah') or 0):,.0f}",
                               "Catatan (tidak dihitung)", "uraian tidak cocok b–f", p])
            continue
        k, kata = kat
        jml = float(it.get("jumlah") or 0)
        per_kat[k] += jml
        no += 1
        detail.append([no, a, it.get("uraian", ""), f"{jml:,.0f}",
                       f"{k}. {KATEGORI_LABEL[k]}", kata, p])

    lower = sum(per_kat.values())                 # Σ a–f (biaya pendukung yang dihitung)
    batas = BATAS_PENDUKUNG * total
    sisa = batas - lower                          # sisa kuota untuk substansi

    summary = [[f"{k}. {KATEGORI_LABEL[k]}", f"{per_kat[k]:,.0f}", pct(per_kat[k])]
               for k in KATEGORI_LABEL]
    summary += [
        ["Σ a–f (biaya pendukung)", f"{lower:,.0f}", pct(lower)],
        ["Total RAB (alokasi dana metadata)", f"{total:,.0f}", "100,00%" if total > 0 else "-"],
    ]
    if catatan_n:
        summary.append([f"Catatan: belanja bahan 521211 tanpa pola ({catatan_n} item, tidak dihitung)",
                        f"{catatan_rp:,.0f}", ""])
    summary += [
        ["Batas maksimal", f"{batas:,.0f}", f"{BATAS_PENDUKUNG:.0%}"],
        ["Sisa untuk komponen yang substansinya tidak berpengaruh secara langsung terhadap keluaran", f"{sisa:,.0f}", pct(max(sisa, 0))],
    ]

    if total <= 0:
        st = "MANUAL"
        alasan = "Alokasi dana pada metadata (total RAB) kosong atau tidak terbaca."
        fb = ""
    elif lower > batas + TOLERANSI_RP:
        st = "TIDAK"
        alasan = (f"Komponen a–f Rp{lower:,.0f} ({pct(lower)} dari total RAB) melebihi batas {BATAS_PENDUKUNG:.0%} "
                  f"(Rp{batas:,.0f}); tidak ada sisa untuk komponen yang substansinya tidak berpengaruh secara langsung terhadap keluaran.")
        fb = itD.get("feedback_revisi", "")
    elif abs(sisa) <= TOLERANSI_RP:
        st = "LOLOS"
        alasan = (f"Komponen a–f Rp{lower:,.0f} ({pct(lower)} dari total RAB) tepat {BATAS_PENDUKUNG:.0%}; "
                  f"batas terpenuhi, sisa untuk komponen yang substansinya tidak berpengaruh secara langsung terhadap keluaran Rp0.")
        fb = ""
    else:
        st = "LOLOS"
        alasan = (f"Komponen a–f Rp{lower:,.0f} ({pct(lower)} dari total RAB) di bawah batas {BATAS_PENDUKUNG:.0%} "
                  f"(Rp{batas:,.0f}); masih tersisa Rp{sisa:,.0f} untuk komponen yang substansinya tidak berpengaruh secara langsung terhadap Keluaran.")
        fb = ""

    if catatan_n:
        alasan = (f"{alasan} Catatan: {catatan_n} item belanja bahan 521211 (Rp{catatan_rp:,.0f}) "
                  f"tidak cocok dengan pola b–f dan tidak dihitung; perlu dicek manual.")
    row = make_iv_row(doc_name, itD, st, alasan, fb, "IV.D-HITUNG", "", "$.struktur")
    row = with_detail(row, hdr_detail, detail)
    row["summary_headers"] = hdr_sum
    row["summary_rows"] = summary
    return row

def klasifikasi_pendukung(akun, uraian):
    """Return (kode_kategori, kata_kunci_yang_cocok) atau None.
    Urutan: akun honorarium -> a; pola uraian a..f; lainnya -> None."""
    u = str(uraian or "").casefold()
    akun = str(akun)
    if akun in AKUN_HONORARIUM:
        return "a", f"akun {akun}"
    if re.search(KATEGORI_POLA["a"], u):
        return "a", re.search(KATEGORI_POLA["a"], u).group(0)
    m_k = re.search(POLA_KONSUMSI, u)
    if m_k and re.search(POLA_RAPAT, u) and not re.search(POLA_EXCL_B, u):
        return "b", f"{m_k.group(0)} + rapat"
    for k in ["c", "d", "e", "f"]:
        m = re.search(KATEGORI_POLA[k], u)
        if m:
            return k, m.group(0)
    return None

def iter_akun_nodes(node, path="$"):
    """Yield (kode_akun, uraian_akun, json_path) untuk setiap node tipe 'akun'."""
    if isinstance(node, list):
        for i, n in enumerate(node):
            yield from iter_akun_nodes(n, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return
    if node.get("tipe") == "akun":
        yield str(node.get("kode", "")), node.get("uraian", ""), path
    if "children" in node:
        yield from iter_akun_nodes(node["children"], f"{path}.children")

def fmt_lokasi(paths):
    if not paths:
        return "-"
    s = "; ".join(paths[:3])
    return s + (f" (+{len(paths) - 3} lagi)" if len(paths) > 3 else "")

# %% [markdown] sel 39
# ## Evaluasi seluruh rule IV
# %% [code] sel 40
def eval_iv(doc_name, doc):
    raw = doc.get("raw", {})
    meta = raw.get("metadata", {}) or {}
    struktur = raw.get("struktur", [])
    items = list(iter_items(struktur, path="$.struktur"))          # IV.B dan IV.D: tanpa konteks wilayah
    items_c = list(ivc_items_with_context(struktur, path="$.struktur"))   # IV.C: membawa lokasi dan konteks wilayah induk
    akun_locs = {}
    for kode, _, p in iter_akun_nodes(struktur, "$.struktur"):
        akun_locs.setdefault(kode, []).append(p)

    return [
        eval_ivb(doc_name, items, akun_locs, find_item("IV.B")),
        eval_ivc(doc_name, items_c, meta, find_item("IV.C")),
        eval_ivd(doc_name, items, meta, find_item("IV.D")),
    ]

# %% [markdown] sel 41
# ## Diagnostik IV.C
# %% [code] sel 42
# ===== Diagnostik IV.C untuk semua RAB: tahun, jumlah item, status, kode alasan =====
def _hitung_harga_rekursif(node):
    """Hitung semua dict yang punya 'harga_satuan', tanpa bergantung pada kunci struktur."""
    if isinstance(node, list):
        return sum(_hitung_harga_rekursif(x) for x in node)
    if isinstance(node, dict):
        n = 1 if "harga_satuan" in node else 0
        return n + sum(_hitung_harga_rekursif(v) for v in node.values() if isinstance(v, (list, dict)))
    return 0

def diagnose_ivc(docs: dict) -> list[dict]:
    """Diagnostik IV.C: docs = {nama: doc dari load_json_document}. Return list ringkasan per file."""
    out = []
    for doc_name, doc in docs.items():
        raw = doc.get("raw", {})
        meta = raw.get("metadata", {}) or {}
        items = list(ivc_items_with_context(raw.get("struktur", []), path="$.struktur"))
        row = eval_ivc(doc_name, items, meta, find_item("IV.C"))
        reasons = row.get("iv_c_items", [])
        st = Counter(x["status"] for x in reasons)
        kd = Counter(x["kode"] for x in reasons if x["status"] != "LOLOS")
        out.append({"file": doc_name, "tahun_raw": meta.get("tahun_anggaran"),
                    "tahun": parse_tahun(meta.get("tahun_anggaran")),
                    "item_iter": len(items), "item_rekursif": _hitung_harga_rekursif(raw.get("struktur", [])),
                    "status": row["status"], **{k: st.get(k, 0) for k in ("LOLOS", "TIDAK", "MANUAL", "DILEWATI")},
                    "kode_teratas": ", ".join(f"{k} {v}" for k, v in kd.most_common(3)) or row["alasan"][:80]})
    return out

import re

def cek_konsistensi_ivc(struktur, tahun=2027):
    items = list(ivc_items_with_context(struktur, path="$.struktur"))
    row = eval_ivc("cek", items, {"tahun_anggaran": tahun}, {})
    grup = {}
    for d in row["detail_rows"]:
        kunci = (str(d[1]).strip(), norm_text(str(d[2])), str(d[3]).strip())
        grup.setdefault(kunci, []).append(d)
    beda = {k: v for k, v in grup.items() if len({(x[9], x[4]) for x in v}) > 1}
    for k, v in beda.items():
        print(f"URAIAN: {k[1]} | akun {k[0]} | harga {k[2]}")
        for x in v:
            print(f"   No {x[0]} | {x[9]} | {x[4]} | wilayah: {x[5]} | {str(x[10])[:90]}")
    print(f"{len(beda)} kelompok uraian sama dengan hasil berbeda dari {len(grup)} kelompok")
    return beda


# ---------------------------------------------------------------------------
# API publik
# ---------------------------------------------------------------------------
def evaluate_document(src, doc_name: str | None = None) -> list[dict]:
    """src = path JSON hasil ExtractorRAB (satu RAB) atau dict-nya. Return baris Section III + IV."""
    doc = load_json_document(src, doc_name)
    if not isinstance(doc["raw"], dict) or "metadata" not in doc["raw"]:
        raise ValueError(f"{doc['name']}: JSON RAB harus memiliki kunci 'metadata' (keluaran ExtractorRAB).")
    rows = [evaluate_item("RAB", doc["name"], item, doc) for item in ROUTES["RAB"]["rules"]["items"]]
    rows.extend(eval_iv(doc["name"], doc))
    return rows


def save_report(doc_name, rows, out_dir, source_path=""):
    return write_report("RAB", doc_name, source_path or doc_name, rows, out_dir, _versions(), rules_by_code)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Checklist RAB (Section III + IV)")
    ap.add_argument("inputs", nargs="+", help="file JSON RAB atau folder berisi *.json")
    ap.add_argument("-o", "--out", default="hasil_cek", help="folder laporan")
    ap.add_argument("--konsistensi", action="store_true", help="cetak uraian sama yang hasil IV.C-nya berbeda")
    a = ap.parse_args(argv)
    files = []
    for p in map(Path, a.inputs):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        rows = evaluate_document(f)
        save_report(f.name, rows, a.out, str(f))
        print(f.name, {s: sum(r["status"] == s for r in rows) for s in ALLOWED_STATUSES})
        if a.konsistensi:
            raw = json.loads(f.read_text(encoding="utf-8-sig"))
            cek_konsistensi_ivc(raw.get("struktur", []), tahun=parse_tahun((raw.get("metadata") or {}).get("tahun_anggaran")) or 2027)
    return 0


if __name__ == "__main__":
    sys.exit(main())

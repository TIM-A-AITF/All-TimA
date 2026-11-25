# -*- coding: utf-8 -*-
"""
Checklist Engine TOR (Section I, I.A-I.T)
=========================================
Modul pustaka (bukan lagi notebook berjalan saat import). Input: JSON checklist hasil
`tor_extractor.checklist_fields()` -> {"fields": {"I.A": {...}, ...}}.

    import checklist_tor as ct
    rows = ct.evaluate_document("checklist_rule_based/X.json")   # atau dict {"fields": ...}

CLI (opsional, untuk uji lokal):
    python checklist_tor.py input/TOR -o hasil_cek

Kriteria I.M-I.R dinilai model Qwen3-4B-Instruct-2507 + LoRA yang di-deploy di VPS lain.
Endpoint diatur lewat env (lihat .env.example):
    SFT_ENDPOINT_URL, SFT_API_KEY, SFT_MODEL_VERSION, SFT_TIMEOUT_SEC, SFT_MAX_EVIDENCE_CHUNKS
Bila SFT_ENDPOINT_URL kosong, kriteria tersebut berstatus MANUAL.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import checklist_common as cc
from checklist_common import (ALLOWED_STATUSES, build_reasoning, load_json_document, write_report)

# ---------------------------------------------------------------------------
# 1. Konfigurasi (semua dari environment)
# ---------------------------------------------------------------------------
SFT_ENDPOINT_URL = os.environ.get("SFT_ENDPOINT_URL") or None   # mis. http://<IP_VPS_SFT>:8000/evaluate
SFT_API_KEY = os.environ.get("SFT_API_KEY", "")
SFT_BASE_MODEL_ID = os.environ.get("SFT_BASE_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
SFT_MODEL_VERSION = os.environ.get("SFT_MODEL_VERSION", "qwen3-4b-instruct-2507-lora")
SFT_TIMEOUT_SEC = float(os.environ.get("SFT_TIMEOUT_SEC", "120"))
SFT_MAX_EVIDENCE_CHUNKS = int(os.environ.get("SFT_MAX_EVIDENCE_CHUNKS", "50"))

PIPELINE_VERSION = "key-value-checklist-v2.2.0"
PROMPT_VERSION = "external-llm-integration-v1"
RULES_VERSION = cc.RULES_VERSION
SECTION_CODE = "I"

# ---------------------------------------------------------------------------
# 2. Rules (dimuat sekali saat import)
# ---------------------------------------------------------------------------
_cfg, rules_by_code, RULES_HASH = cc.load_rules()
if SECTION_CODE not in rules_by_code:
    raise KeyError(f"Section {SECTION_CODE} untuk TOR tidak ditemukan pada {cc.RULES_PATH}.")
TOR_RULES = {**rules_by_code[SECTION_CODE],
             "items": [it for it in rules_by_code[SECTION_CODE]["items"]
                       if str(it.get("kriteria_kode", "")).strip().startswith("I.")]}
ROUTES = {"TOR": {"section_code": SECTION_CODE, "label": "Kelengkapan TOR", "rules": TOR_RULES}}

# ---------------------------------------------------------------------------
# 3. Evaluator LLM eksternal (placeholder endpoint SFT)
# ---------------------------------------------------------------------------
OUTPUT_SCHEMA = {
    "status": "LOLOS | TIDAK | MANUAL",
    "alasan": "alasan spesifik terhadap kriteria dan evidence; jangan mengarang",
    "evidence_quote": "kutipan verbatim dari salah satu kandidat, atau string kosong",
    "evidence_chunk_id": "chunk_id kandidat, atau null",
}

LLM_EVALUATOR = None          # boleh diganti fungsi lain dengan kontrak yang sama (lihat run_external_llm_evaluator)
model_version = f"sft-endpoint ({SFT_MODEL_VERSION})" if SFT_ENDPOINT_URL else "sft-endpoint-not-configured"


def _manual_result(alasan):
    return {"status": "MANUAL", "alasan": alasan, "evidence_quote": "", "evidence_chunk_id": None}


def _build_sft_request(kind, doc_name, item, document):
    """Payload untuk endpoint SFT. SESUAIKAN dengan kontrak API di VPS model."""
    code = str(item.get("kriteria_kode", "")).strip()
    fields = (document.get("raw") or {}).get("fields", {})
    konteks = {fc: (fields.get(fc) or {}).get("value", "") for fc in KEY_RULES["TOR"].get(code, [])}
    return {
        "task": "tor_checklist",
        "model": SFT_BASE_MODEL_ID,
        "jenis_dokumen": kind,
        "nama_file": doc_name,
        "kriteria_kode": code,
        "pertanyaan": item.get("pertanyaan", ""),
        "konteks": konteks,                                   # isi bagian yang dinilai (dari field I.M-I.R)
        "candidates": [{"chunk_id": c["chunk_id"], "quote": c["quote"]}
                       for c in document.get("chunks", [])[:SFT_MAX_EVIDENCE_CHUNKS]],
        "output_schema": OUTPUT_SCHEMA,
    }


def sft_endpoint_evaluator(kind, doc_name, item, document):
    """POST ke SFT_ENDPOINT_URL (PLACEHOLDER).
    Kontrak yang diasumsikan:
      Request : JSON dari _build_sft_request(); header Authorization: Bearer <SFT_API_KEY> bila diisi
      Response: JSON {"status", "alasan", "evidence_quote", "evidence_chunk_id"}
    Kutipan tetap divalidasi terhadap chunk sumber di validate_prediction()."""
    if not SFT_ENDPOINT_URL:
        return _manual_result("Endpoint model SFT belum dikonfigurasi (SFT_ENDPOINT_URL kosong). Perlu review manual.")
    body = json.dumps(_build_sft_request(kind, doc_name, item, document), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if SFT_API_KEY:
        headers["Authorization"] = f"Bearer {SFT_API_KEY}"
    req = urllib.request.Request(SFT_ENDPOINT_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=SFT_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return _manual_result(f"Panggilan endpoint SFT gagal ({type(e).__name__}). Perlu review manual.")
    if not isinstance(data, dict):
        return _manual_result("Respons endpoint SFT bukan objek JSON. Perlu review manual.")
    return {"status": data.get("status", "MANUAL"), "alasan": data.get("alasan", ""),
            "evidence_quote": data.get("evidence_quote", "") or "",
            "evidence_chunk_id": data.get("evidence_chunk_id")}


def run_external_llm_evaluator(kind, doc_name, item, document):
    fn = LLM_EVALUATOR or sft_endpoint_evaluator
    return fn(kind=kind, doc_name=doc_name, item=item, document=document)


def criterion_method(kind, item):
    """I.M-I.R -> LLM; I.A-I.L dan I.S-I.T -> rule deterministik."""
    code = str(item.get("kriteria_kode", "")).strip().upper()
    match = re.match(r"^I\.([A-T])(?:[.\-]|$)", code)
    if kind == "TOR" and match:
        letter = match.group(1)
        if "M" <= letter <= "R":
            return "llm_integration"
        return "rule_based_key"
    return "manual"

# ---------------------------------------------------------------------------
# 5. Rule engine deterministik berbasis key-value
# ---------------------------------------------------------------------------
import re


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().casefold()


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").casefold())


# Alias key JSON per kriteria. Alias dibuat spesifik per template/jenis dokumen.
# Pemeriksaan dilakukan terhadap KEY dan validitas nilai, bukan isi semantik.
KEY_RULES = {
    "TOR": {
        "I.A.1": ["I.A"],
        "I.B.1": ["I.B"],
        "I.C.1": ["I.C"],
        "I.D.1": ["I.D"],
        "I.E.1": ["I.E"],
        "I.F.1": ["I.F"],
        "I.G.1": ["I.G"],
        "I.H.1": ["I.H"],
        "I.I.1": ["I.I"],
        "I.J.1": ["I.J"],
        "I.K.1": ["I.K"],
        "I.L.1": ["I.L"],
        "I.M.1": ["I.M.1", "I.M.2", "I.M.3"],
        "I.M.2": ["I.M.2"],
        "I.M.3": ["I.M.3"],
        "I.N.1": ["I.N"],
        "I.O.1": ["I.O.1", "I.O.2", "I.O.3"],
        "I.O.2": ["I.O.2"],
        "I.O.3": ["I.O.3"],
        "I.P.1": ["I.P"],
        "I.Q.1": ["I.Q"],
        "I.R.1": ["I.R"],
        "I.S.1": ["I.S.1", "I.S.2"],
        "I.T.1": ["I.T"],
    }
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


def check_tor_fields(doc, criterion_code):
    raw = doc.get("raw", {})
    fields = raw.get("fields", {})

    field_codes = KEY_RULES.get("TOR", {}).get(criterion_code, [])

    if not field_codes:
        return {
            "status": "TIDAK",
            "evidence": [],
            "quote": "",
            "field_codes": field_codes,
            "note": f"Mapping field TOR tidak ditemukan untuk {criterion_code}",
            "missing_detail": [],
        }

    evidence = []
    quotes = []
    missing = []
    missing_detail = []
    is_llm_tor_criterion = bool(re.match(r"^I\.[M-R](?:[.\-]|$)", criterion_code))
    # I.S.1: kedua field (I.S.1 & I.S.2) harus present=true dengan value tidak kosong;
    # label_found tidak menjadi syarat pada format baru.
    presence_value_only = is_llm_tor_criterion or criterion_code.strip().upper() == "I.S.1"

    for field_code in field_codes:
        field = fields.get(field_code)

        if not isinstance(field, dict):
            missing.append(field_code)
            missing_detail.append({"field": field_code, "sebab": "tidak_ada"})
            continue

        label = str(field.get("label", "")).strip()
        raw_value = field.get("value")
        value = str(raw_value).strip() if raw_value is not None else ""

        present = field.get("present") is True
        label_found = field.get("label_found") is True

        if not present:
            missing.append(field_code)
            missing_detail.append({"field": field_code, "sebab": "tidak_terdeteksi"})
            continue
        if not value:
            missing.append(field_code)
            missing_detail.append({"field": field_code, "sebab": "nilai_kosong"})
            continue
        if not presence_value_only and not label_found:
            missing.append(field_code)
            missing_detail.append({"field": field_code, "sebab": "label_tidak_ditemukan"})
            continue

        if label:
            evidence.append(f"key:{label}")

        quotes.append(value)

    if missing:
        return {
            "status": "TIDAK",
            "evidence": evidence,
            "quote": "\n".join(quotes),
            "field_codes": field_codes,
            "note": (
                "Field TOR tidak memenuhi syarat "
                + ("present=true dan value tidak kosong: " if presence_value_only
                   else "present=true, label_found=true, dan value tidak kosong: ")
                + ", ".join(missing)
            ),
            "missing_detail": missing_detail,
        }

    return {
        "status": "LOLOS",
        "evidence": list(dict.fromkeys(evidence)),
        "quote": "\n".join(quotes),
        "field_codes": field_codes,
        "note": "Field TOR ditemukan dan memenuhi pemeriksaan keberadaan.",
        "missing_detail": [],
    }


def deterministic_key_check(kind, doc, item):
    code = str(item.get("kriteria_kode", "")).strip().upper()
    rule = KEY_RULES.get(kind, {}).get(code)

    if kind == "TOR":
        criterion_code = item.get("kriteria_kode", "")
        result = check_tor_fields(doc, criterion_code)

        matches = []
        if result["quote"]:
            matches.append({
                "quote": result["quote"],
                "chunk_id": f"{Path(doc.get('path', 'document')).stem}:field:{criterion_code}",
                "text": result["quote"],
                "json_path": f"$.fields.{criterion_code}",
            })

        return {
            "status": result["status"],
            "evidence": result["evidence"],
            "note": result["note"],
            "missing_detail": result.get("missing_detail", []),
        }, matches

    if not rule:
        return {"status": "MANUAL", "evidence": [], "note": "Key rule belum dikonfigurasi"}, []

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
        unique_keys = list(dict.fromkeys(k for k, _, _ in target_entries))
        result = {
            "status": "LOLOS",
            "evidence": [f"key:{k}" for k in unique_keys],
            "note": "Key dan nilai ditemukan",
        }
    elif rule["type"] == "signature_all" and not passed:
        result = {
            "status": "TIDAK",
            "evidence": [f"key:{k}" for k, _, _ in target_entries],
            "note": "Nama dan jabatan wajib tersedia, serta tanda_tangan atau tanda_tangan_elektronik harus true",
        }
    elif rule["type"] == "signature" and any(
        v is False for k, v, p in entries if _norm_key(k) == "detected" and ".tanda_tangan." in p
    ):
        result = {"status": "TIDAK", "evidence": [f"key:{k}" for k, _, _ in matches],
                  "note": "Nama penanggung jawab ditemukan, tetapi tanda tangan tidak terdeteksi"}
    elif rule["type"] == "nip" and any(
        v is False for k, v, p in entries if _norm_key(k) == "adalabel" and ".nip." in p
    ):
        result = {"status": "TIDAK", "evidence": [], "note": "Label NIP secara eksplisit tidak ditemukan"}
    elif matches or (rule["type"] in {"signature", "nip"} and target_entries):
        result = {"status": "MANUAL", "evidence": [f"key:{k}" for k, _, _ in target_entries],
                  "note": "Key ditemukan tetapi nilai/flag verifikasi belum memenuhi syarat"}
    else:
        result = {"status": "TIDAK", "evidence": [], "note": "Key wajib tidak ditemukan atau nilainya kosong"}
    return result, evidence_chunks

# %%
DETECTORS_BY_TYPE = KEY_RULES
NEED_ALL = set()  # Validasi multi-key ditangani eksplisit pada deterministic_key_check().


def normalize_patterns(value):
    """Kompatibilitas untuk metadata laporan lama; tidak dipakai untuk matching key-value."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [p for p in value if isinstance(p, str) and p.strip()]
    return []


def validate_all_detectors(detector_by_type):
    """Validasi mapping field TOR dan objek key rule RAB."""
    for kind, detectors in detector_by_type.items():
        if not isinstance(detectors, dict):
            raise ValueError(f"Detector harus berupa mapping: {kind}")

        for code, rule in detectors.items():
            if kind == "TOR":
                # Aturan TOR berisi daftar kode field, misalnya ["I.A"].
                if (
                    not isinstance(rule, (list, tuple))
                    or not rule
                    or not all(isinstance(x, str) and x.strip() for x in rule)
                ):
                    raise ValueError(f"Mapping field TOR tidak valid: {kind}/{code}")
            elif not isinstance(rule, dict) or not rule.get("keys"):
                raise ValueError(f"Key rule tidak valid: {kind}/{code}")


validate_all_detectors(DETECTORS_BY_TYPE)


def check_item(body, kriteria_kode, kind):
    """Adapter kompatibilitas; evaluasi utama menggunakan deterministic_key_check()."""
    return {"status": "MANUAL", "evidence": [], "note": "Gunakan deterministic_key_check() berbasis JSON key-value"}


def validate_detector_patterns(detectors):
    """Tidak ada pola regex untuk pemeriksaan key-value."""
    return True


def legacy_regex_adapter(kind, doc, item):
    """Adapter lama dipertahankan agar bentuk data hasil evaluasi/report tidak berubah."""
    return deterministic_key_check(kind, doc, item)

# ---------------------------------------------------------------------------
# 6. Validator keluaran (implementasi di checklist_common)
# ---------------------------------------------------------------------------
def _versions():
    return {"pipeline": PIPELINE_VERSION, "rules_version": RULES_VERSION, "rules_sha256": RULES_HASH,
            "model_version": model_version, "sft_model_version": SFT_MODEL_VERSION if SFT_ENDPOINT_URL else None,
            "prompt_version": PROMPT_VERSION, "embedding_model": "not_used_key_value"}


def validate_prediction(pred, candidates, doc_name, kind, item, method, rule_hits=None):
    return cc.validate_prediction(pred, candidates, doc_name, kind, item, method,
                                  ROUTES[kind]["section_code"], _versions(), rule_hits)

# ---------------------------------------------------------------------------
# 7. Evaluasi satu dokumen
# ---------------------------------------------------------------------------
def evaluate_item(kind, doc_name, item, doc):
    method = criterion_method(kind, item)
    pred = None
    rule_hits = []
    legacy_result = {"status": "MANUAL", "evidence": [], "note": "Belum dievaluasi"}
    legacy_matches = []

    if method == "rule_based_key":
        legacy_result, legacy_matches = deterministic_key_check(kind, doc, item)
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

    elif method == "llm_integration":
        # Tidak menjalankan model di notebook ini. Seluruh data dan kriteria diteruskan
        # melalui hook LLM_EVALUATOR yang berasal dari notebook LLM terpisah.
        candidates = list(doc.get("chunks", []))
        code = str(item.get("kriteria_kode", "")).strip().upper()
        is_tor_llm_criterion = kind == "TOR" and re.match(r"^I\.[M-R](?:[.\-]|$)", code)
        field_result = None
        if is_tor_llm_criterion:
            field_result = check_tor_fields(doc, code)

        if field_result is not None and field_result["status"] != "LOLOS":
            legacy_result = field_result
            pred = {
                "status": "TIDAK",
                "alasan": field_result["note"],
                "evidence_quote": "",
                "evidence_chunk_id": None,
            }
        elif field_result is not None:
            legacy_result = field_result
            try:
                pred = run_external_llm_evaluator(kind, doc_name, item, doc)
            except Exception as e:
                pred = {
                    "status": "MANUAL",
                    "alasan": f"Pipeline LLM belum tersedia/gagal dihubungkan ({type(e).__name__}).",
                    "evidence_quote": "",
                    "evidence_chunk_id": None,
                }
            if str(pred.get("status", "")).upper().strip() == "MANUAL":
                quote = field_result["quote"]
                evidence_chunk = next(
                    (candidate for candidate in candidates if candidate.get("quote") == quote),
                    None,
                )
                if evidence_chunk is None:
                    evidence_chunk = {
                        "chunk_id": f"{Path(doc.get('path', doc_name)).stem}:field:{code}",
                        "text": quote,
                        "quote": quote,
                        "source_file": doc.get("path", doc_name),
                        "json_path": "$.fields." + ", $.fields.".join(field_result["field_codes"]),
                        "page": None,
                        "section": None,
                        "heading": code,
                        "block_id": None,
                        "source_id": None,
                        "metadata": {},
                        "evidence_type": "source_fragment",
                    }
                    candidates.append(evidence_chunk)
                pred["evidence_quote"] = quote
                pred["evidence_chunk_id"] = evidence_chunk["chunk_id"]
        else:
            try:
                pred = run_external_llm_evaluator(kind, doc_name, item, doc)
            except Exception as e:
                pred = {
                    "status": "MANUAL",
                    "alasan": f"Pipeline LLM belum tersedia/gagal dihubungkan ({type(e).__name__}).",
                    "evidence_quote": "",
                    "evidence_chunk_id": None,
                }

    else:
        # Untuk metode manual pada dokumen tunggal TOR atau RAB.
        candidates = list(doc.get("chunks", []))
        pred = {
            "status": "MANUAL",
            "alasan": "Kriteria memerlukan pemeriksaan dan validasi reviewer.",
            "evidence_quote": "",
            "evidence_chunk_id": None,
        }

    if method == "rule_based_key":
        candidates = list(doc.get("chunks", []))
        if kind == "TOR":
            for candidate in legacy_matches:
                if not any(existing.get("quote") == candidate.get("quote") for existing in candidates):
                    candidates.append(candidate)

    row = validate_prediction(pred, candidates, doc_name, kind, item, method, rule_hits)
    alasan_validator = row["alasan"]
    pred_status = str(pred.get("status", "")).upper().strip()
    # model benar-benar dipanggil bila kriteria LLM lolos pemeriksaan keberadaan field
    llm_called = method == "llm_integration" and legacy_result.get("status") == "LOLOS"

    if method in ("rule_based_key", "llm_integration"):
        row["alasan"], row["feedback"] = build_reasoning(kind, item, row["status"], legacy_result)
        if row["status"] == "LOLOS":
            row["catatan"] = ""

    # Kriteria LLM: alasan dari model dipakai untuk status apa pun (bukan catatan rule keberadaan field).
    # Bila validator menurunkan status model (mis. kutipan tidak identik), alasan validator didahulukan.
    if llm_called:
        if row["status"] != pred_status:
            row["alasan"] = alasan_validator + (f" Alasan model ({pred_status}): {pred['alasan']}" if pred.get("alasan") else "")
        elif pred.get("alasan"):
            row["alasan"] = pred["alasan"]

    if row.get("status") != "LOLOS" and not row.get("catatan"):
        row["catatan"] = legacy_result.get("note", "")

    if kind == "TOR" and method == "llm_integration" and row.get("status") == "MANUAL":
        row["feedback"] = ""

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


# ---------------------------------------------------------------------------
# API publik
# ---------------------------------------------------------------------------
def evaluate_document(src, doc_name: str | None = None) -> list[dict]:
    """src = path JSON {"fields": ...} hasil tor_extractor, atau dict-nya. Return baris hasil per kriteria."""
    doc = load_json_document(src, doc_name)
    if not isinstance(doc["raw"], dict) or "fields" not in doc["raw"]:
        raise ValueError(f"{doc['name']}: JSON TOR harus memiliki kunci 'fields' (keluaran checklist_fields).")
    return [evaluate_item("TOR", doc["name"], item, doc) for item in ROUTES["TOR"]["rules"]["items"]]


def save_report(doc_name, rows, out_dir, source_path=""):
    return write_report("TOR", doc_name, source_path or doc_name, rows, out_dir, _versions(), rules_by_code)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Checklist TOR (Section I)")
    ap.add_argument("inputs", nargs="+", help="file JSON {fields} atau folder berisi *.json")
    ap.add_argument("-o", "--out", default="hasil_cek", help="folder laporan")
    a = ap.parse_args(argv)
    files = []
    for p in map(Path, a.inputs):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        rows = evaluate_document(f)
        save_report(f.name, rows, a.out, str(f))
        print(f.name, {s: sum(r["status"] == s for r in rows) for s in ALLOWED_STATUSES})
    return 0


if __name__ == "__main__":
    sys.exit(main())

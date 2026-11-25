# -*- coding: utf-8 -*-
"""Komponen bersama checklist TOR dan RAB.

Semua path dibaca dari environment variable dengan bawaan relatif terhadap folder
modul ini, sehingga sama-sama jalan di laptop (Windows) maupun container Docker:

    CHECKLIST_RULES_DIR   folder rules/            (bawaan: <folder ini>/rules)
    CHECKLIST_KATALOG_DIR folder katalog_sbm/      (bawaan: <folder ini>/katalog_sbm)
    CHECKLIST_WILAYAH_CSV wilayah_indonesia.csv    (bawaan: <folder ini>/wilayah_indonesia.csv)
    CHECKLIST_CACHE_DIR   cache yang boleh ditulis (bawaan: <folder katalog>)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

CHECKLIST_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else Path(default).resolve()


RULES_DIR = env_path("CHECKLIST_RULES_DIR", CHECKLIST_DIR / "rules")
RULES_PATH = RULES_DIR / "rules_config.json"
KATALOG_DIR = env_path("CHECKLIST_KATALOG_DIR", CHECKLIST_DIR / "katalog_sbm")
WILAYAH_CSV = env_path("CHECKLIST_WILAYAH_CSV", CHECKLIST_DIR / "wilayah_indonesia.csv")
CACHE_DIR = env_path("CHECKLIST_CACHE_DIR", KATALOG_DIR)

RULES_VERSION = "checklist-rules-v1"   # ubah saat rules direvisi; hash rules juga dicatat
ALLOWED_STATUSES = {"LOLOS", "TIDAK", "MANUAL"}


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def load_rules(path: Path = RULES_PATH):
    """Return (cfg, rules_by_code, rules_hash)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Rules tidak ditemukan: {path}. Letakkan rules_config.json pada folder rules/ "
            "atau set CHECKLIST_RULES_DIR.")
    raw = path.read_bytes()
    cfg = json.loads(raw.decode("utf-8-sig"))
    rules_hash = hashlib.sha256(raw).hexdigest()[:16]
    sections = cfg.get("screening_satu_format_dan_substansi")
    if not isinstance(sections, list):
        raise ValueError("rules_config.json harus memiliki list 'screening_satu_format_dan_substansi'.")
    by_code = {}
    for section in sections:
        code = section.get("section_code")
        if not code:
            raise ValueError(f"Section tanpa section_code: {section}")
        if code in by_code:
            raise ValueError(f"section_code duplikat: {code}")
        if not isinstance(section.get("items"), list):
            raise ValueError(f"Section {code} tidak memiliki list 'items'.")
        by_code[code] = section
    return cfg, by_code, rules_hash


# ---------------------------------------------------------------------------
# JSON -> chunk evidence (mempertahankan lokasi sumber)
# ---------------------------------------------------------------------------
TEXT_KEYS = ("raw_text", "chunk_text", "markdown", "text", "content", "paragraph", "body", "quote")
META_KEYS = {
    "page", "page_number", "page_no", "section", "section_title", "heading",
    "block_id", "chunk_id", "source_id", "document_id", "file_name", "table", "row",
}
ENTITY_KEYS = ("entity", "label", "field", "name", "key")
VALUE_KEYS = ("value", "normalized_value", "text_value")


def first_present(d, keys):
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def split_source_sentences(text):
    """Memecah teks menjadi kalimat (heuristik) tanpa mengubah kutipan verbatim."""
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<!\d\.)(?<=[.!?])\s+(?=[A-Z0-9\"“‘(])", text)
    return [p.strip() for p in parts if p.strip()] or [text]


def extract_chunks_from_json(obj, source_file):
    chunks = []

    def add_chunk(text, quote, path, meta, context=""):
        raw_quote = str(quote).strip()
        if not raw_quote:
            return
        units = split_source_sentences(raw_quote) if len(raw_quote) > 180 else [raw_quote]
        for unit in units:
            display = (context + " | " if context else "") + (unit if len(units) > 1 else str(text).strip())
            chunks.append({
                "chunk_id": "",
                "text": display,
                "quote": unit,
                "source_file": str(source_file),
                "json_path": path,
                "page": meta.get("page_number", meta.get("page_no", meta.get("page"))),
                "section": meta.get("section_title", meta.get("section")),
                "heading": meta.get("heading"),
                "block_id": meta.get("block_id"),
                "source_id": meta.get("source_id"),
                "metadata": dict(meta),
                "evidence_type": "sentence" if re.search(r"[.!?]$", unit) else "source_fragment",
            })

    def walk(node, path="$", inherited=None):
        inherited = dict(inherited or {})
        if isinstance(node, dict):
            for k in META_KEYS:
                if node.get(k) not in (None, ""):
                    inherited[k] = node[k]
            payload_key = next((k for k in TEXT_KEYS if isinstance(node.get(k), str) and node[k].strip()), None)
            entity = first_present(node, ENTITY_KEYS)
            context = " | ".join(
                f"{k}: {node[k]}" for k in ("entity", "label", "field", "name", "key", "type")
                if isinstance(node.get(k), (str, int, float)) and str(node[k]).strip())
            if payload_key:
                add_chunk(node[payload_key], node[payload_key], f"{path}.{payload_key}", inherited, context)
                for k, v in node.items():
                    if k != payload_key and isinstance(v, (dict, list)):
                        walk(v, f"{path}.{k}", inherited)
                return
            value = first_present(node, VALUE_KEYS)
            if entity is not None and value is not None and not isinstance(value, (dict, list)):
                add_chunk(f"{entity}: {value}", value, path, inherited, str(entity))
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, f"{path}.{k}", inherited)
                elif k not in TEXT_KEYS and v is not None and isinstance(v, (str, int, float)):
                    val = str(v).strip()
                    if k not in META_KEYS and (isinstance(v, (bool, int, float)) or len(val) >= 8):
                        add_chunk(f"{k}: {val}", val, f"{path}.{k}", inherited, str(k))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]", inherited)
        elif isinstance(node, str) and node.strip():
            add_chunk(node, node, path, inherited)

    walk(obj)
    seen, unique = set(), []
    for c in chunks:
        key = (c["source_file"], c["json_path"], c["quote"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    stem = Path(str(source_file)).stem
    for i, c in enumerate(unique, 1):
        c["chunk_id"] = f"{stem}:{i}"
    return unique


def load_json_document(src, name: str | None = None) -> dict:
    """src = path JSON atau dict yang sudah dimuat. Return {"path", "name", "raw", "chunks"}."""
    if isinstance(src, (str, Path)):
        path = Path(src)
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        name = name or path.name
        path_s = str(path)
    else:
        raw = src
        name = name or "document.json"
        path_s = name
    return {"path": path_s, "name": name, "raw": raw, "chunks": extract_chunks_from_json(raw, path_s)}


# ---------------------------------------------------------------------------
# Validator keluaran & alasan
# ---------------------------------------------------------------------------
def sentence_from_quote(quote):
    if not quote:
        return None
    units = split_source_sentences(quote)
    return units[0] if units else quote


METHOD_DESCRIPTION = {
    "rule_based_key": "Deterministik: pencocokan key-value JSON",
    "llm_integration": "Integrasi LLM eksternal (endpoint SFT)",
    "cross_document_manual": "Pemeriksaan lintas dokumen; wajib review manusia",
    "manual": "Pemeriksaan manual",
}


def validate_prediction(pred, candidates, doc_name, kind, item, method, section_code, versions, rule_hits=None):
    status = str(pred.get("status", "MANUAL")).upper().strip()
    alasan = str(pred.get("alasan", "")).strip()
    quote = str(pred.get("evidence_quote", "") or "")
    chunk_id = pred.get("evidence_chunk_id")
    rule_hits = rule_hits or []

    if status not in ALLOWED_STATUSES:
        status, alasan = "MANUAL", "Status model tidak valid; perlu pemeriksaan manual."
    chosen = next((c for c in candidates if chunk_id is not None and str(c["chunk_id"]) == str(chunk_id)), None)
    if quote:
        matching = next((c for c in candidates if c.get("quote") == quote), None)
        if chosen is None and matching is not None:
            chosen = matching
        if chosen is None or quote != chosen.get("quote"):
            status = "MANUAL"
            alasan = "Kutipan tidak identik dengan kandidat sumber; perlu pemeriksaan manual."
            chosen = None
    elif status == "LOLOS":
        status, alasan, chosen = "MANUAL", "Status LOLOS tidak memiliki kutipan evidence yang dapat diverifikasi.", None
    if status == "LOLOS" and chosen is None:
        status, alasan = "MANUAL", "Status LOLOS tidak dapat diverifikasi dengan evidence yang valid."

    loc = None
    if chosen is not None:
        start = chosen.get("quote", "").find(quote) if quote else -1
        loc = {
            "source_file": chosen.get("source_file"), "json_path": chosen.get("json_path"),
            "page": chosen.get("page"), "section": chosen.get("section"),
            "heading": chosen.get("heading"), "block_id": chosen.get("block_id"),
            "source_id": chosen.get("source_id"), "chunk_id": chosen.get("chunk_id"),
            "char_start_in_quote": start if start >= 0 else None,
            "char_end_in_quote": start + len(quote) if start >= 0 else None,
        }
    lolos = status == "LOLOS"
    return {
        "jenis_dokumen": kind,
        "file_sumber": doc_name,
        "section_code": section_code,
        "kriteria_kode": item.get("kriteria_kode"),
        "pertanyaan": item.get("pertanyaan", ""),
        "catatan": "" if lolos else (item.get("catatan_kriteria", item.get("catatan", "")) or ""),
        "feedback": "" if lolos else (item.get("feedback_revisi", item.get("feedback", "")) or ""),
        "metode_checklist": {"kode": method, "deskripsi": METHOD_DESCRIPTION.get(method, method),
                             "rule_hit_count": len(rule_hits), "semantic_retrieval": False},
        "status": status,
        "alasan": alasan or "Belum ada alasan yang memadai; perlu pemeriksaan manual.",
        "evidence": {
            "sentence": sentence_from_quote(quote) if chosen is not None else None,
            "quote": quote if chosen is not None else None,
            "evidence_type": chosen.get("evidence_type") if chosen else None,
            "location": loc,
            "retrieval_score": chosen.get("retrieval_score") if chosen else None,
        },
        "versions": dict(versions),
        "review_required": status != "LOLOS" or loc is None,
    }


SEBAB_TEXT = {
    "tidak_ada": "belum tercantum dalam dokumen",
    "tidak_terdeteksi": "tidak terdeteksi pada dokumen",
    "nilai_kosong": "label ada tetapi isinya kosong",
    "label_tidak_ditemukan": "isi ada tetapi label/judulnya tidak ditemukan",
}


def build_reasoning(kind, item, status, legacy_result):
    kode = item.get("kriteria_kode", "")
    base_fb = item.get("feedback_revisi") or item.get("feedback") or ""
    note = legacy_result.get("note", "")
    missing = legacy_result.get("missing_detail", [])
    if status == "LOLOS":
        return (f"Kriteria {kode} terpenuhi. {note}", "")
    if status == "TIDAK":
        if missing:
            rincian = "; ".join(f"{m['field']} {SEBAB_TEXT.get(m['sebab'], '')}" for m in missing)
            n = len(missing)
            alasan = f"Kriteria {kode} belum terpenuhi ({n} bagian kurang): {rincian}."
            feedback = ((base_fb + f" Fokus pada: {rincian}.").strip() if n == 1
                        else (base_fb or f"Lengkapi {n} bagian yang kurang."))
            return (alasan, feedback)
        return (f"Kriteria {kode} belum terpenuhi: {note}.", base_fb)
    return (f"Perlu pemeriksaan manual untuk {kode}: {note}.", "")


# ---------------------------------------------------------------------------
# Laporan per dokumen (JSON + Markdown)
# ---------------------------------------------------------------------------
def perbaiki_teks(s):
    """Perbaiki mojibake (UTF-8 terbaca cp1252) dan ganti tanda strip panjang dengan '-'."""
    if not isinstance(s, str):
        return s
    if re.search(r"[ÃÂâ]", s):
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return s.replace("—", "-").replace("–", "-")


def md_escape(value):
    s = perbaiki_teks(str(value if value is not None else ""))
    return (s.replace("\\", "\\\\").replace("$", "\\$").replace("|", "\\|")
             .replace("\n", " ").replace("\r", " "))


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(md_escape(c) for c in r) + " |")
    return out


def loc_str(ev):
    loc = (ev or {}).get("location") or {}
    return "; ".join(f"{k}={v}" for k, v in loc.items() if v not in (None, ""))


def safe_file_stem(name, max_len=120):
    """Nama file aman di Linux maupun Windows."""
    stem = re.sub(r'[\\/:*?"<>|#\x00-\x1f]', "_", Path(str(name)).stem).strip(" .") or "dokumen"
    if len(stem) <= max_len:
        return stem
    digest = hashlib.md5(str(name).encode("utf-8")).hexdigest()[:6]
    return f"{stem[:max_len - 7]}_{digest}"


def write_report(kind, doc_name, source_path, rows, out_dir, versions, rules_by_code) -> dict:
    """Tulis <out_dir>/<kind>/<kind>__<stem>__checklist_report.{json,md}. Return path keduanya."""
    base = f"{kind}__{safe_file_stem(doc_name)}__checklist_report"
    d = Path(out_dir) / kind
    d.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(r["status"] == s for r in rows) for s in ALLOWED_STATUSES}
    section_codes = list(dict.fromkeys(r.get("section_code") for r in rows if r.get("section_code")))
    payload = {"jenis_dokumen": kind, "file_sumber": str(source_path), "nama_file": doc_name,
               "section_codes": section_codes, "summary": {**counts, "total": len(rows)},
               "versions": dict(versions), "hasil": rows}
    pj = d / f"{base}.json"
    pj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def sec_title(code):
        return perbaiki_teks(rules_by_code.get(code, {}).get("section_title", ""))

    lines = [f"# Checklist: {doc_name}", "",
             f"- Jenis dokumen: **{kind}**",
             f"- Ringkasan: LOLOS {counts['LOLOS']}, TIDAK {counts['TIDAK']}, MANUAL {counts['MANUAL']}, total {len(rows)}",
             f"- Pipeline: `{versions.get('pipeline')}`",
             f"- Rules version: `{versions.get('rules_version')}` (`{versions.get('rules_sha256')}`)",
             f"- Model version: `{versions.get('model_version')}`", ""]
    for code in section_codes:
        sec_rows = [r for r in rows if r.get("section_code") == code]
        sc = {s: sum(r["status"] == s for r in sec_rows) for s in ALLOWED_STATUSES}
        lines += [f"## Section {code} - {sec_title(code)}", "",
                  f"Ringkasan: LOLOS {sc['LOLOS']}, TIDAK {sc['TIDAK']}, MANUAL {sc['MANUAL']}", ""]
        if code == "IV":
            if any(r["status"] != "LOLOS" for r in sec_rows):
                headers = ["Kriteria", "Status", "Pertanyaan", "Feedback", "Alasan", "Lokasi"]
                body = [[r.get("kriteria_kode"), r["status"],
                         r.get("pertanyaan", "") if r["status"] != "LOLOS" else "",
                         r.get("feedback", "") if r["status"] != "LOLOS" else "",
                         r.get("alasan"), loc_str(r.get("evidence"))] for r in sec_rows]
            else:
                headers = ["Kriteria", "Status", "Alasan", "Lokasi"]
                body = [[r.get("kriteria_kode"), r["status"], r.get("alasan"), loc_str(r.get("evidence"))]
                        for r in sec_rows]
        else:
            headers = ["Kriteria", "Pertanyaan", "Status", "Metode", "Evidence pola",
                       "Catatan", "Feedback", "Alasan", "Kalimat bukti", "Lokasi"]
            body = [[r.get("kriteria_kode"), r.get("pertanyaan"), r["status"],
                     (r.get("metode_checklist") or {}).get("deskripsi", ""),
                     ", ".join(r.get("evidence_pola", [])), r.get("catatan"), r.get("feedback"),
                     r.get("alasan"), (r.get("evidence") or {}).get("sentence"),
                     loc_str(r.get("evidence"))] for r in sec_rows]
        lines += md_table(headers, body) + [""]
        for r in sec_rows:
            if r.get("detail_rows"):
                lines += [f"### Detail {r['kriteria_kode']}", ""] + md_table(r["detail_headers"], r["detail_rows"]) + [""]
            if r.get("summary_rows"):
                lines += [f"### Ringkasan {r['kriteria_kode']}", ""] + md_table(r["summary_headers"], r["summary_rows"]) + [""]
    pm = d / f"{base}.md"
    pm.write_text("\n".join(lines), encoding="utf-8")
    return {"json": pj, "md": pm}

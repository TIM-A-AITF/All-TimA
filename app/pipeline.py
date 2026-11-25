#!/usr/bin/env python3
"""Orkestrator TOR + RAB:
upload -> klasifikasi -> ekstraksi (+VLM) -> checklist -> pemasangan TOR-RAB -> laporan -> ZIP.

Struktur satu run (di TOR_RAB_WORKSPACE/runs/<run_id>/):
    upload/                 PDF asli
    hasil/pasangan_XX/      TOR/ (JSON detail, aset, html, llm2, checklist_rule_based) + RAB/*.json
    hasil/tanpa_pasangan/
    laporan/                checklist_tor.*, checklist_rab.*, llm_tor.json, hasil_gabungan.json, ringkasan.html
    manifest.json, status.json, hasil_<run_id>.zip

CLI:
    python -m app.pipeline TOR.pdf RAB1.pdf RAB2.pdf
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional

from . import config

config.setup()

import pdfplumber                     # noqa: E402

import ExtractorRAB as rab            # noqa: E402
import tor_extractor as tor           # noqa: E402
import checklist_rab as crab          # noqa: E402
import checklist_tor as ctor          # noqa: E402

from . import report                  # noqa: E402
from .vlm_client import RemoteFigureVLM  # noqa: E402

log = logging.getLogger("pipeline")

TOR_RE = re.compile(r"\b(TOR|KAK|KERANGKA\s+ACUAN\s+KERJA|TERM\s+OF\s+REFERENCE)\b", re.I)
RAB_RE = re.compile(r"\bRAB\b|RINCIAN\s+ANGGARAN\s+(BIAYA|BELANJA)", re.I)
STOP = {"dan", "yang", "untuk", "dengan", "pada", "dari", "dalam", "kegiatan", "output", "rincian", "program", "tahun"}
ALLOWED_EXT = {".pdf"}

_RUN_LOCK = threading.Lock()          # CFG ekstraktor & cache checklist bersifat global: satu run pada satu waktu


# ---------- util ----------
def safe_filename(name: str) -> str:
    """Nama berkas upload yang aman (tanpa path, tanpa karakter terlarang)."""
    base = Path(str(name).replace("\\", "/")).name
    base = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", base).strip(" .")
    return base[:180] or f"berkas_{uuid.uuid4().hex[:6]}.pdf"


def read_status(run_dir: Path) -> dict:
    p = run_dir / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _write_status(run_dir: Path, **kw) -> None:
    """Tulis atomik (tmp + replace) agar API tidak pernah membaca berkas setengah jadi."""
    cur = read_status(run_dir)
    cur.update(kw, diperbarui=datetime.datetime.now().isoformat(timespec="seconds"))
    tmp = run_dir / f".status.{uuid.uuid4().hex[:6]}.tmp"
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "status.json")


# ---------- 1. Klasifikasi ----------
def first_page_text(pdf: Path, max_chars: int = 600) -> str:
    try:
        with pdfplumber.open(pdf) as p:
            return (p.pages[0].extract_text() or "")[:max_chars]
    except Exception:
        return ""


def classify(pdf: Path) -> tuple[Optional[str], str]:
    """Judul halaman 1 diutamakan, lalu nama file. Judul yang memuat TOR dan RAB sekaligus = tidak pasti."""
    for sumber, teks in (("judul halaman 1", first_page_text(pdf)), ("nama file", pdf.stem.replace("_", " "))):
        has_tor, has_rab = bool(TOR_RE.search(teks)), bool(RAB_RE.search(teks))
        if has_tor and has_rab:
            if sumber == "judul halaman 1":
                continue                    # coba nama file sebagai pemutus
            return None, f"{sumber} memuat TOR dan RAB sekaligus"
        if has_tor or has_rab:
            return ("TOR" if has_tor else "RAB"), sumber
    return None, "tidak dikenali"


# ---------- 2. Kunci pemasangan ----------
def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", (s or "").lower())) - STOP


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------- 3. Ekstraksi + checklist ----------
def process_tor(pdf: Path, work: Path, vlm: Optional[RemoteFigureVLM]) -> dict:
    out = work / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    doc = tor.extract_pdf(pdf, str(out), write=True)

    vlm_info = {"status": "dilewati", "alasan": "VLM_ENDPOINT_URL kosong"}
    if vlm is not None and vlm.enabled:
        n_fig = len(tor.figures_to_describe(doc, config.VLM_MIN_AREA))
        if n_fig:
            doc = tor.describe_figures(doc, str(pdf), vlm, min_area=config.VLM_MIN_AREA)
            (out / f"{pdf.stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            v = doc.get("vlm", {})
            vlm_info = {"status": "ok" if not v.get("failed") else "sebagian_gagal",
                        "gambar": n_fig, "dideskripsikan": v.get("figures_described", 0), "gagal": len(v.get("failed", []))}
        else:
            vlm_info = {"status": "dilewati", "alasan": "tidak ada gambar"}

    info = tor.write_outputs(doc, out, pdf.stem)
    fields = tor.checklist_fields(doc)
    f = fields["fields"]
    doc_name = f"{pdf.stem}.json"
    item = {"kind": "TOR", "stem": pdf.stem, "doc_name": doc_name, "dir": out, "status": doc["status"],
            "key": tokens(f"{f.get('I.F', {}).get('value', '')} {f.get('I.J', {}).get('value', '')}"),
            "vlm": vlm_info, "rows": None,
            "meta": {"pdf": pdf.name, "status_ekstraksi": doc["status"], "halaman": doc["source"]["pages"]},
            "errors": {k: v for k, v in info.items() if k.endswith("_error")}}
    try:
        item["rows"] = ctor.evaluate_document(fields, doc_name)
    except Exception as e:
        item["errors"]["checklist_error"] = repr(e)[:300]
        log.exception("checklist TOR gagal: %s", pdf.name)
    return item


def process_rab(pdf: Path, work: Path, hasil: Path) -> list[dict]:
    out = work / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    res = rab.parse_rab(pdf, detect_sig=True)
    paths = rab.write_split(res, out, pdf.stem, full=False)
    multi = bool(res.get("multi_rab"))
    items = res["rab"] if multi else [res]
    cands = []
    for it in items:
        m = it.get("metadata", {})
        js = out / (f"{pdf.stem}_rab{it['no_rab']:02d}.json" if multi else f"{pdf.stem}.json")
        c = {"kind": "RAB", "stem": pdf.stem, "doc_name": js.name, "json_files": [js], "rows": None, "errors": {},
             "key": tokens(f"{m.get('kegiatan') or ''} {m.get('kro') or ''} {m.get('unit_eselon2_satker') or ''}"),
             "meta": {"pdf": pdf.name, "no_rab": it.get("no_rab"), "tahun_anggaran": m.get("tahun_anggaran")}}
        try:
            c["rows"] = crab.evaluate_document(js)
        except Exception as e:
            c["errors"]["checklist_error"] = repr(e)[:300]
            log.exception("checklist RAB gagal: %s", js.name)
        cands.append(c)
    ring = out / f"{pdf.stem}_ringkasan.json"
    if multi and ring in paths and ring.exists():
        shutil.copy2(ring, hasil / f"ringkasan_rab_{pdf.stem}.json")
    return cands


# ---------- 4. Pemasangan ----------
def pair(tors: list[dict], rabs: list[dict]):
    if len(tors) == 1 and len(rabs) == 1:              # satu TOR dan satu RAB: pasangkan apa pun skornya
        return [(tors[0], rabs[0], round(jaccard(tors[0]["key"], rabs[0]["key"]), 2))], [], []
    if len(tors) == 1 and rabs:                        # satu TOR + beberapa RAB: semua RAB milik TOR itu
        return [(tors[0], r, round(jaccard(tors[0]["key"], r["key"]), 2)) for r in rabs], [], []
    scored = sorted(((jaccard(t["key"], r["key"]), i, j)
                     for i, t in enumerate(tors) for j, r in enumerate(rabs)), reverse=True)
    used_t, used_r, pairs = set(), set(), []
    for s, i, j in scored:
        if s < config.PAIR_MIN_SCORE:
            break
        if j in used_r:
            continue
        used_t.add(i); used_r.add(j)                    # satu TOR boleh punya beberapa RAB; satu RAB hanya satu TOR
        pairs.append((tors[i], rabs[j], round(s, 2)))
    return (pairs,
            [t for i, t in enumerate(tors) if i not in used_t],
            [r for j, r in enumerate(rabs) if j not in used_r])


def place(item: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if item["kind"] == "TOR":
        target = dest / "TOR"
        if item["dir"].exists() and not target.exists():
            shutil.move(str(item["dir"]), str(target))   # path di dalam JSON relatif, aman dipindah
            item["dir"] = target
        elif item["dir"].exists():
            shutil.copytree(item["dir"], target, dirs_exist_ok=True)
    else:
        (dest / "RAB").mkdir(exist_ok=True)
        for p in item["json_files"]:
            if p.exists():
                shutil.copy2(p, dest / "RAB" / p.name)


# ---------- 5. Run ----------
def create_run(root: Optional[Path] = None) -> Path:
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = (root or config.RUNS_DIR) / run_id
    (run_dir / "upload").mkdir(parents=True, exist_ok=True)
    _write_status(run_dir, run_id=run_id, status="menunggu")
    return run_dir


def process_run(run_dir: Path, use_vlm: Optional[bool] = None) -> dict:
    """Proses semua PDF di run_dir/upload. Return manifest (juga ditulis ke manifest.json)."""
    with _RUN_LOCK:
        try:
            return _process_run(run_dir, use_vlm)
        except Exception as e:
            _write_status(run_dir, status="gagal", error=repr(e)[:500], trace=traceback.format_exc()[-2000:])
            raise


def _process_run(run_dir: Path, use_vlm: Optional[bool]) -> dict:
    run_id = run_dir.name
    upload, work, hasil, laporan = run_dir / "upload", run_dir / "_work", run_dir / "hasil", run_dir / "laporan"
    for d in (work / "tor", work / "rab", hasil, laporan):
        d.mkdir(parents=True, exist_ok=True)
    _write_status(run_dir, status="berjalan", tahap="klasifikasi")

    vlm = RemoteFigureVLM() if (use_vlm is not False and config.VLM_ENDPOINT_URL) else None
    manifest = {"run_id": run_id, "dibuat": datetime.datetime.now().isoformat(timespec="seconds"),
                "versi": {"tor_extractor": tor.EXTRACTOR_VERSION, "rab_parser": rab.PARSER_VERSION,
                          "checklist_tor": ctor.PIPELINE_VERSION, "checklist_rab": crab.PIPELINE_VERSION,
                          "sft_model": ctor.model_version, "vlm": config.VLM_MODEL_ID if vlm else "tidak dikonfigurasi"},
                "berkas": [], "pasangan": [], "tanpa_pasangan": {"TOR": [], "RAB": []}, "tidak_dikenali": []}
    tors, rabs = [], []
    files = sorted(p for p in upload.iterdir() if p.is_file())
    for k, f in enumerate(files, 1):
        entry = {"berkas": f.name}
        if f.suffix.lower() not in ALLOWED_EXT:
            entry.update(jenis=None, alasan="bukan PDF")
            manifest["tidak_dikenali"].append(f.name); manifest["berkas"].append(entry)
            continue
        kind, alasan = classify(f)
        entry.update(jenis=kind, alasan=alasan)
        _write_status(run_dir, tahap=f"ekstraksi {k}/{len(files)}: {f.name} ({kind or '?'})")
        try:
            if kind == "TOR":
                t = process_tor(f, work / "tor", vlm)
                tors.append(t)
                entry.update(status=t["status"], vlm=t["vlm"], **t["errors"])
            elif kind == "RAB":
                cs = process_rab(f, work / "rab", hasil)
                rabs += cs
                entry.update(jumlah_rab=len(cs), json=[c["doc_name"] for c in cs],
                             **{f"{c['doc_name']}_error": c["errors"]["checklist_error"] for c in cs if c["errors"]})
            else:
                manifest["tidak_dikenali"].append(f.name)
        except Exception as e:                         # satu berkas gagal tidak menghentikan run
            entry["error"] = repr(e)[:300]
            log.exception("gagal memproses %s", f.name)
        manifest["berkas"].append(entry)

    _write_status(run_dir, tahap="pemasangan")
    pairs, sisa_t, sisa_r = pair(tors, rabs)
    folders: dict[str, Path] = {}
    for t, r, s in pairs:                               # satu folder per TOR; RAB-nya digabung di folder yang sama
        if t["doc_name"] not in folders:
            folders[t["doc_name"]] = hasil / f"pasangan_{len(folders) + 1:02d}"
            place(t, folders[t["doc_name"]])
        dest = folders[t["doc_name"]]
        place(r, dest)
        manifest["pasangan"].append({"folder": dest.name, "TOR": t["doc_name"], "RAB": r["doc_name"], "skor": s})
    for t in sisa_t:
        place(t, hasil / "tanpa_pasangan"); manifest["tanpa_pasangan"]["TOR"].append(t["doc_name"])
    for r in sisa_r:
        place(r, hasil / "tanpa_pasangan"); manifest["tanpa_pasangan"]["RAB"].append(r["doc_name"])

    _write_status(run_dir, tahap="laporan")
    tor_res = [{"doc_name": t["doc_name"], "rows": t["rows"], "meta": t["meta"]} for t in tors if t["rows"] is not None]
    rab_res = [{"doc_name": r["doc_name"], "rows": r["rows"], "meta": r["meta"]} for r in rabs if r["rows"] is not None]
    report.write_tor_outputs(tor_res, laporan)
    report.write_rab_outputs(rab_res, laporan)
    report.write_combined(tor_res, rab_res, laporan, manifest)
    report.write_html(tor_res, rab_res, laporan, manifest)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not config.KEEP_WORK_DIR:
        shutil.rmtree(work, ignore_errors=True)
    zip_path = report.build_run_zip(run_dir, run_dir / f"hasil_{run_id}.zip", extra={"run_id": run_id})
    _write_status(run_dir, status="selesai", tahap="selesai", zip=zip_path.name,
                  ringkasan={"TOR": len(tors), "RAB": len(rabs), "pasangan": len(pairs),
                             "tidak_dikenali": len(manifest["tidak_dikenali"])})
    return manifest


def run(files: list[Path], root: Optional[Path] = None, use_vlm: Optional[bool] = None) -> Path:
    """Satu langkah untuk CLI/skrip: salin berkas -> proses -> return run_dir."""
    run_dir = create_run(root)
    for f in files:
        shutil.copy2(f, run_dir / "upload" / safe_filename(f.name))
    process_run(run_dir, use_vlm)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("pemakaian: python -m app.pipeline TOR.pdf RAB1.pdf [RAB2.pdf ...]")
        sys.exit(2)
    print(run([Path(p) for p in sys.argv[1:]]))

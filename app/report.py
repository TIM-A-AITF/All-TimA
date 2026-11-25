"""Pengolahan hasil checklist TOR & RAB menjadi laporan dan paket ZIP.

Keluaran (di folder `laporan/` setiap run):
    checklist_tor.json / .md     ringkasan + hasil per kriteria semua TOR
    checklist_rab.json / .md     ringkasan + hasil per kriteria semua RAB (termasuk tabel detail IV.B/IV.C/IV.D)
    llm_tor.json                 kriteria I.M-I.R untuk Tim B
    hasil_gabungan.json          pasangan TOR-RAB + ringkasan masing-masing
    ringkasan.html               tabel ringkas untuk dibuka di browser
    TOR/<nama>.md, RAB/<nama>.md laporan per dokumen
"""
from __future__ import annotations

import html
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("LOLOS", "TIDAK", "MANUAL")
LLM_CODES_PREFIX = ("I.M", "I.N", "I.O", "I.P", "I.Q", "I.R")  # I.M–I.R untuk Tim B


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_stem(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|#\x00-\x1f]', "_", Path(str(name)).stem).strip(" .")[:120] or "dokumen"


def _normalize_row(row: dict, kind: str) -> dict:
    """Pastikan setiap baris memiliki kunci standar; tidak mengubah status."""
    ev = row.get("evidence") or {}
    loc = ev.get("location") or {}
    status = row.get("status") if row.get("status") in STATUSES else "MANUAL"
    out = {
        "kriteria_kode": row.get("kriteria_kode", ""),
        "section_code": row.get("section_code", ""),
        "pertanyaan": row.get("pertanyaan", ""),
        "status": status,
        "alasan": row.get("alasan", ""),
        "feedback": row.get("feedback", "") if status != "LOLOS" else "",
        "catatan": row.get("catatan", "") if status != "LOLOS" else "",
        "metode": (row.get("metode_checklist") or {}).get("kode", row.get("metode", "")),
        "evidence": {
            "sentence": ev.get("sentence"),
            "quote": ev.get("quote"),
            "json_path": loc.get("json_path"),
            "block_id": loc.get("block_id"),
            "page": loc.get("page"),
        },
        "perlu_review": bool(row.get("review_required", status != "LOLOS")),
        "jenis": kind,
    }
    if row.get("detail_rows"):                       # IV.B / IV.C / IV.D: rincian per akun/item
        out["detail"] = {"headers": row.get("detail_headers", []), "rows": row["detail_rows"]}
    if row.get("summary_rows"):
        out["ringkasan"] = {"headers": row.get("summary_headers", []), "rows": row["summary_rows"]}
    return out


def _summary(rows: list[dict]) -> dict:
    c = Counter(r["status"] for r in rows)
    return {**{s: c.get(s, 0) for s in STATUSES}, "total": len(rows),
            "perlu_review": sum(r["perlu_review"] for r in rows)}


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda v: str(v if v is not None else "").replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _render_md(title: str, kind: str, doc_name: str, rows: list[dict], meta: dict) -> str:
    s = _summary(rows)
    head = [
        f"# {title}", "",
        f"- Dokumen: `{doc_name}`",
        f"- Jenis: **{kind}**",
        f"- Ringkasan: LOLOS {s['LOLOS']}, TIDAK {s['TIDAK']}, MANUAL {s['MANUAL']} (total {s['total']})",
        f"- Perlu review: {s['perlu_review']}",
        *(f"- {k}: `{v}`" for k, v in meta.items()), "", "",
    ]
    body = _md_table(
        ["Kriteria", "Status", "Alasan", "Feedback", "Kalimat bukti", "Lokasi"],
        [[r["kriteria_kode"], r["status"], r["alasan"], r["feedback"],
          r["evidence"]["sentence"] or "", r["evidence"]["json_path"] or ""] for r in rows],
    )
    extra = []
    for r in rows:
        if r.get("detail"):
            extra += ["", f"## Detail {r['kriteria_kode']}", "", _md_table(r["detail"]["headers"], r["detail"]["rows"])]
        if r.get("ringkasan"):
            extra += ["", f"## Ringkasan {r['kriteria_kode']}", "", _md_table(r["ringkasan"]["headers"], r["ringkasan"]["rows"])]
    return "\n".join(head) + body + "\n" + "\n".join(extra) + "\n"


def _write_md_per_doc(folder: Path, doc_name: str, title: str, kind: str, rows: list[dict], meta: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{_safe_stem(doc_name)}.md"
    p.write_text(_render_md(title, kind, doc_name, rows, meta), encoding="utf-8")
    return p


def _render_index(title: str, checklist: list[dict]) -> str:
    rows = [[c["file"], c["summary"]["LOLOS"], c["summary"]["MANUAL"],
             c["summary"]["TIDAK"], c["summary"]["perlu_review"]] for c in checklist]
    return f"# {title}\n\n" + _md_table(["Dokumen", "LOLOS", "MANUAL", "TIDAK", "Perlu review"], rows) + "\n"


def write_tor_outputs(tor_results: list[dict], out: Path) -> dict:
    """tor_results: [{'doc_name', 'rows', 'meta'}]."""
    checklist, llm_rows = [], []
    for res in tor_results:
        rows = [_normalize_row(r, "TOR") for r in res["rows"]]
        checklist.append({"file": res["doc_name"], "meta": res.get("meta", {}), "summary": _summary(rows), "hasil": rows})
        llm_rows += [{"file": res["doc_name"], **r} for r in rows
                     if r["kriteria_kode"].upper().startswith(LLM_CODES_PREFIX)]
        _write_md_per_doc(out / "TOR", res["doc_name"], "Checklist TOR", "TOR", rows, res.get("meta", {}))
    p1 = _write_json(out / "checklist_tor.json", {"generated_at": _now(), "dokumen": checklist})
    p2 = _write_json(out / "llm_tor.json", {"generated_at": _now(), "kriteria": llm_rows})
    p3 = out / "checklist_tor.md"
    p3.write_text(_render_index("Checklist TOR", checklist), encoding="utf-8")
    return {"checklist_tor.json": p1, "checklist_tor.md": p3, "llm_tor.json": p2}


def write_rab_outputs(rab_results: list[dict], out: Path) -> dict:
    """rab_results: [{'doc_name', 'rows', 'meta'}] (satu entri per RAB; PDF gabungan = beberapa entri)."""
    checklist = []
    for res in rab_results:
        rows = [_normalize_row(r, "RAB") for r in res["rows"]]
        checklist.append({"file": res["doc_name"], "meta": res.get("meta", {}), "summary": _summary(rows), "hasil": rows})
        _write_md_per_doc(out / "RAB", res["doc_name"], "Checklist RAB", "RAB", rows, res.get("meta", {}))
    p1 = _write_json(out / "checklist_rab.json", {"generated_at": _now(), "dokumen": checklist})
    p2 = out / "checklist_rab.md"
    p2.write_text(_render_index("Checklist RAB", checklist), encoding="utf-8")
    return {"checklist_rab.json": p1, "checklist_rab.md": p2}


def write_combined(tor_results: list[dict], rab_results: list[dict], out: Path, pairing: dict) -> dict:
    """hasil_gabungan.json: pasangan TOR-RAB beserta ringkasan masing-masing."""
    def pack(res, kind):
        rows = [_normalize_row(r, kind) for r in res["rows"]]
        return {"file": res["doc_name"], "summary": _summary(rows)}
    by_tor = {t["doc_name"]: t for t in tor_results}
    by_rab = {r["doc_name"]: r for r in rab_results}
    gabungan = {
        "generated_at": _now(),
        "run_id": pairing.get("run_id"),
        "pasangan": [
            {"folder": p["folder"], "skor": p.get("skor"),
             "TOR": pack(by_tor[p["TOR"]], "TOR") if p["TOR"] in by_tor else None,
             "RAB": pack(by_rab[p["RAB"]], "RAB") if p["RAB"] in by_rab else None}
            for p in pairing.get("pasangan", [])
        ],
        "tanpa_pasangan": pairing.get("tanpa_pasangan", {}),
        "tidak_dikenali": pairing.get("tidak_dikenali", []),
    }
    return {"hasil_gabungan.json": _write_json(out / "hasil_gabungan.json", gabungan)}


def write_html(tor_results: list[dict], rab_results: list[dict], out: Path, pairing: dict | None = None) -> Path:
    e = html.escape

    def card(kind, res):
        s = _summary([_normalize_row(r, kind) for r in res["rows"]])
        return (f"<tr><td>{kind}</td><td>{e(res['doc_name'])}</td><td>{s['LOLOS']}</td>"
                f"<td>{s['MANUAL']}</td><td>{s['TIDAK']}</td><td>{s['perlu_review']}</td></tr>")
    body = "".join(card("TOR", r) for r in tor_results) + "".join(card("RAB", r) for r in rab_results)
    pair_html = ""
    if pairing:
        prs = "".join(f"<tr><td>{e(p['folder'])}</td><td>{e(p['TOR'])}</td><td>{e(p['RAB'])}</td><td>{p.get('skor')}</td></tr>"
                      for p in pairing.get("pasangan", []))
        sisa = pairing.get("tanpa_pasangan", {})
        lain = ", ".join(e(x) for x in sisa.get("TOR", []) + sisa.get("RAB", []) + pairing.get("tidak_dikenali", [])) or "-"
        pair_html = (f"<h2>Pasangan TOR–RAB</h2><table><tr><th>Folder</th><th>TOR</th><th>RAB</th><th>Skor</th></tr>{prs}</table>"
                     f"<p>Tanpa pasangan / tidak dikenali: {lain}</p>")
    page = f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>Ringkasan checklist</title>
<style>body{{font:14px system-ui;margin:24px}}table{{border-collapse:collapse;width:100%;margin-bottom:16px}}
td,th{{border:1px solid #ccc;padding:6px 8px;text-align:left}}th{{background:#f3f5f8}}</style></head>
<body><h1>Ringkasan checklist</h1><p>Dibuat: {_now()}{(' · run ' + e(str(pairing.get('run_id')))) if pairing else ''}</p>
<table><tr><th>Jenis</th><th>Dokumen</th><th>LOLOS</th><th>MANUAL</th><th>TIDAK</th><th>Perlu review</th></tr>
{body}</table>{pair_html}</body></html>"""
    p = out / "ringkasan.html"
    p.write_text(page, encoding="utf-8")
    return p


EXCLUDE_FROM_ZIP = {"_work", "status.json"}


def build_run_zip(run_dir: Path, zip_path: Path, extra: dict | None = None) -> Path:
    """Kemas seluruh hasil run (tanpa folder _work) dalam satu ZIP."""
    run_dir, zip_path = Path(run_dir).resolve(), Path(zip_path).resolve()
    keep = lambda p: (p.is_file() and p != zip_path and p.suffix != ".zip"
                      and not EXCLUDE_FROM_ZIP & set(p.relative_to(run_dir).parts))
    manifest = {"generated_at": _now(), "berkas": sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if keep(p))}
    if extra:
        manifest.update(extra)
    _write_json(run_dir / "manifest_paket.json", manifest)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run_dir.rglob("*")):
            if keep(p):
                z.write(p, p.relative_to(run_dir).as_posix())
    return zip_path

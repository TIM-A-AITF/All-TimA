#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rab_parser.py — Parser Rincian Anggaran Biaya (RAB) K/L -> JSON terstruktur.

Input  : PDF (pdfplumber), XLSX (openpyxl), atau TXT/MD (hasil ekstraksi markdown).
Output : JSON berisi
  - metadata      : kementerian, satker, kegiatan, KRO, volume, satuan, alokasi, penandatangan
  - struktur      : pohon KRO > RO > Komponen > Subkomponen > Akun > Kelompok > Item
  - items         : daftar datar seluruh item rincian (uraian, rincian perhitungan,
                    volume, satuan, harga satuan, jumlah) beserta konteks hierarkinya
  - peringatan    : hal yang tidak bisa diurai dengan yakin
Parser hanya MENGEKSTRAK angka yang tercantum di dokumen; tidak ada perhitungan total/subtotal,
selisih, ringkasan per akun, maupun blok validasi aritmetika. Kecocokan aritmetika dokumen HANYA dipakai
untuk menentukan letak node di pohon (judul yang tercetak di atas akunnya, kelompok bertingkat, baris
bernominal yang judulnya tersamar) -- angka tidak pernah diubah atau ditambah. Selisih hitung milik
dokumen dibiarkan apa adanya (dinilai di tahap reviu, bukan oleh parser).

Pemakaian:
  python rab_parser.py RAB.pdf -o rab.json
  python rab_parser.py RAB.pdf -o rab.json --csv items.csv
  python rab_parser.py RAB.pdf --debug-lines        # lihat baris hasil rekonstruksi + posisi x
  python rab_parser.py folder_rab/ -o hasil/        # batch
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

PARSER_VERSION = "1.21-ekstraksi"

# --------------------------------------------------------------------------------------
# Utilitas angka & teks
# --------------------------------------------------------------------------------------

def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "").replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", s).strip()


# Gaya angka dokumen. Dokumen gabungan memuat RAB dari satker berbeda: ada yang memakai titik
# sebagai pemisah ribuan ('99.557.000'), ada yang koma ('99,557,000'). Gaya dideteksi per RAB
# (detect_number_style) lalu dipasang lewat set_number_style sebelum RAB itu diurai.
_COMMA_THOUSANDS = False


def set_number_style(comma_thousands: bool) -> None:
    global _COMMA_THOUSANDS
    _COMMA_THOUSANDS = bool(comma_thousands)


def detect_number_style(texts) -> bool:
    """True bila koma dipakai sebagai pemisah ribuan (>= 2 kelompok, mis. '4,393,138,000')."""
    comma = dot = 0
    for t in texts:
        comma += len(re.findall(r"(?<![\d.,])\d{1,3}(?:,\d{3}){2,}(?![\d,])", t))
        dot += len(re.findall(r"(?<![\d.,])\d{1,3}(?:\.\d{3}){2,}(?![\d.])", t))
    return comma > dot


def parse_num(tok: str) -> Optional[int | float]:
    """Angka: 3.829.000 | 1.800 | 86000 | 1,5 | Rp404.708.000 (gaya titik) atau 99,557,000 (gaya koma).
    Angka negatif: -4.267.500 | -Rp 4.267.500 | Rp-4.267.500 | (4.267.500)."""
    t = tok.strip()
    neg = False
    if re.fullmatch(r"\(.+\)", t):
        neg, t = True, t[1:-1].strip()
    m = re.match(r"^(-)?\s*Rp\.?\s*(-)?\s*", t, flags=re.I)
    if m:
        neg = neg or bool(m.group(1) or m.group(2))
        t = t[m.end():]
    if t.startswith("-") and len(t) > 1:
        neg, t = True, t[1:].strip()
    v = _parse_num_abs(t)
    return None if v is None else (-v if neg else v)


def _parse_num_abs(t: str) -> Optional[int | float]:
    # Angka berpemisah ribuan tidak boleh diawali nol ('00.000.000' adalah pecahan '900.000.000',
    # bukan 0); bila diterima, hitungan 0 x h = 0 tampak 'cocok' dan menutup perbaikan yang benar.
    if _COMMA_THOUSANDS:
        if re.fullmatch(r"[1-9]\d{0,2}(,\d{3})+(\.\d+)?", t):
            t = t.replace(",", "")
        elif re.fullmatch(r"(?:0|[1-9]\d*)(\.\d{1,2})?", t):
            pass
        else:
            return None
    elif re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+(,\d+)?", t):
        t = t.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"(?:0|[1-9]\d*),\d+", t):
        t = t.replace(",", ".")
    elif re.fullmatch(r"(?:0|[1-9]\d*)(\.\d{1,2})?", t):
        pass
    else:
        return None
    v = float(t)
    return int(v) if v.is_integer() else v


def parse_amount(text: str) -> Optional[int | float]:
    """
    Nominal rupiah yang toleran terhadap format header antar-satker, dan tidak bergantung gaya angka:
    'Rp 960.000.000,-' | 'Rp 4 ,435,894,000' | 'Rp4,393,138,000,-' | '1 1.361.030.000' | '15,707,353,000'.
    """
    t = re.sub(r"^\s*(?:Rp\.?|IDR|Rupiah)\s*", "", (text or "").strip(), flags=re.I)
    m = re.match(r"\d[\d\s.,]*", t)
    if not m:
        return None
    t = m.group(0).rstrip(" .,")
    t = re.sub(r"(?<=\d)\s+(?=[\d.,])", "", t)          # '4 ,435' / '1 1.361' -> gabung
    t = re.sub(r"(?<=[.,])\s+(?=\d)", "", t)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):
        digits = re.sub(r"[.,]", "", t)
    elif re.fullmatch(r"(\d{1,3}(?:\.\d{3})*),\d{1,2}", t):          # 960.000.000,00
        digits = t.split(",")[0].replace(".", "")
    elif re.fullmatch(r"(\d{1,3}(?:,\d{3})*)\.\d{1,2}", t):          # 960,000,000.00
        digits = t.split(".")[0].replace(",", "")
    elif re.fullmatch(r"\d+", t):
        digits = t
    else:
        return None
    return int(digits)


_ANGKA_KATA = {"nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7,
               "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11, "dua belas": 12,
               "tiga belas": 13, "empat belas": 14, "lima belas": 15, "dua puluh": 20}


def parse_volume(val: str) -> dict:
    """Nilai header 'Volume': '1 (satu)' | '12 Layanan' | '1,5 Paket' | '12 (dua belas) Bulan' | 'satu'.
    Hasil: {'angka': int|float|None, 'teks': str, 'satuan': str|None (satuan yang menempel di nilai)}."""
    raw = clean_text(val or "").lstrip(":").strip()
    out = {"angka": None, "teks": raw, "satuan": None}
    m = re.match(r"^(\d[\d.,]*)\s*(?:\(([^)]*)\))?\s*(.*)$", raw)
    if m:
        out["angka"] = parse_num(m.group(1).rstrip(".,"))
        sat = re.sub(r"^[\-/:\s]+", "", m.group(3)).strip()
        if sat and re.match(r"[A-Za-z]", sat) and len(sat) <= 40 and not MASK_ANY_RE.search(sat):
            out["satuan"] = sat
        return out
    kata = re.sub(r"\s*\(.*?\)", "", raw).strip().lower()
    for w, n in sorted(_ANGKA_KATA.items(), key=lambda kv: -len(kv[0])):
        if kata == w or kata.startswith(w + " "):
            out["angka"] = n
            rest = kata[len(w):].strip()
            if rest and re.match(r"[a-z]", rest) and len(rest) <= 40:
                out["satuan"] = rest.title()
            break
    return out


def is_num(tok: str) -> bool:
    return parse_num(tok) is not None


def normalize_tokens(toks: list[str]) -> list[str]:
    """'-Rp 85.350.000' -> '-85.350.000'; 'Rp -' / '-' di ujung baris angka -> '0'."""
    out: list[str] = []
    # format akuntansi Excel '_-Rp* #,##0_-' -> '-Rp 10,156,000-': minus penutup hanyalah
    # padding format, bukan tanda negatif. Buang pada angka yang diawali 'Rp' / '-Rp'.
    pre: list[str] = []
    for tk in toks:
        # harga satuan dalam kurung: '(Rp 19,750,000)' / '(Rp19,750,000)' -> '19,750,000'
        mp = re.fullmatch(r"\(Rp\.?(\d[\d.,]*)\)", tk, re.I)
        if mp:
            pre.append(mp.group(1))
            continue
        if pre and re.fullmatch(r"\d[\d.,]*\)", tk) and re.fullmatch(r"\(Rp\.?", pre[-1], re.I):
            pre[-1] = tk[:-1]
            continue
        if pre and re.fullmatch(r"\d[\d.,]*-", tk) and re.fullmatch(r"-?\s*Rp\.?", pre[-1], re.I):
            pre[-1] = tk[:-1]          # '-Rp' + '10,156,000-' -> '10,156,000' (positif)
        else:
            pre.append(tk)
    # 'Rp13.067.000,00' (Rp menempel ke nominal) -> '13.067.000,00'
    toks = [re.sub(r"^Rp\.?(?=\d)", "", tk, flags=re.I) if re.fullmatch(r"Rp\.?\d[\d.,]*", tk, re.I) else tk
            for tk in pre]
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if re.fullmatch(r"-\s*Rp\.?", t, re.I) and nxt and re.fullmatch(r"[\d.,]+", nxt):
            out.append("-" + nxt)
            i += 2
            continue
        if t.lower() in ("rp", "rp.") and nxt and re.fullmatch(r"-[\d.,]+", nxt):
            out.append(nxt)
            i += 2
            continue
        if t in ("-", "\u2013") and out and len(toks) >= 2 \
                and (i == len(toks) - 1 or (nxt is not None and re.fullmatch(r"[\d.,]+", nxt))
                     # 'Rp - Rp -' (harga & jumlah nol): '-' pertama diikuti 'Rp' kolom berikutnya
                     or (nxt is not None and nxt.lower() in ("rp", "rp.") and out[-1].lower() in ("rp", "rp."))) \
                and (out[-1].lower() in ("rp", "rp.") or is_num(out[-1])
                     or re.fullmatch(UNIT, out[-1]) and AKUN_RE_EARLY.match(toks[0]) is not None):
            out.append("0")
            i += 1
            continue
        out.append(t)
        i += 1
    return out


AKUN_RE_EARLY = re.compile(r"^\d{6}$")


def fmt_idr(v: Optional[int | float]) -> Optional[str]:
    if v is None:
        return None
    return "Rp" + f"{int(round(v)):,}".replace(",", ".")


UNIT = r"[A-Za-z][A-Za-z0-9²]*\.?"          # 'orang', 'kali', 'OK', 'M2', 'm²'
RINC_RE = re.compile(rf"\d[\d.,]*\s+(?:x\s+)?{UNIT}(?:\s+x\s+\d[\d.,]*\s+{UNIT})*$")
# versi longgar: faktor boleh berurutan tanpa 'x' ('5 orang 12 Bulan'); hanya dipakai bila
# hasil kalinya cocok dengan volume / harga x jumlah, supaya uraian ('A4 80 Gram') tidak tertelan.
LOOSE_RINC_RE = re.compile(rf"\d[\d.,]*\s+(?:x\s+)?{UNIT}(?:\s+(?:x\s+)?\d[\d.,]*\s+{UNIT})*")
FACTOR_RE = re.compile(rf"(\d[\d.,]*)\s+(?:x\s+)?({UNIT})")


def find_rincian(tokens: list[str]) -> Optional[int]:
    """Indeks token awal 'Rincian Perhitungan' (mis. '4 orang x 4 kali', '30 OK', '53 x kegiatan')."""
    for i, t in enumerate(tokens):
        if not is_num(t) or i + 1 >= len(tokens):
            continue
        nxt = tokens[i + 1]
        if nxt.lower() == "x" or re.fullmatch(UNIT, nxt):
            if RINC_RE.fullmatch(" ".join(tokens[i:])):
                return i
    return None


def find_rincian_for(tokens: list[str], vol=None, harga=None, jumlah=None) -> Optional[int]:
    """
    Seperti find_rincian, tetapi memilih awal rincian (dari kiri = yang terlengkap) yang hasil kali
    faktornya cocok dengan volume, atau dengan jumlah / harga satuan. Bila tak ada yang cocok,
    kembali ke pencarian ketat.
    """
    for i, t in enumerate(tokens):
        if not is_num(t) or i + 1 >= len(tokens):
            continue
        s = " ".join(tokens[i:])
        if not LOOSE_RINC_RE.fullmatch(s):
            continue
        _, prod, _ = rincian_info(s)
        if prod is None:
            continue
        if vol is not None and abs(prod - vol) < 1e-9:
            # '... 20 ORG x 4 KEG x 1 HR 80 OH': potongan '1 HR 80 OH' memotong rumus di tengah (didahului 'x');
            # pasangan kolom volume+satuan di ujung ('80 OH') yang menjadi rincian, rumus tetap di uraian.
            if i > 0 and tokens[i - 1].lower() in ("x", "×") and len(tokens) - 2 > i \
                    and is_num(tokens[-2]) and parse_num(tokens[-2]) == vol and re.fullmatch(UNIT, tokens[-1]):
                return len(tokens) - 2
            return i
        if vol is None and harga is not None and jumlah is not None and \
                (abs(prod * harga - jumlah) < 0.5 or abs(prod * jumlah - harga) < 0.5):   # kolom tertukar
            if i > 0 and tokens[i - 1].lower() in ("x", "×") and len(tokens) - 2 > i \
                    and is_num(tokens[-2]) and abs(parse_num(tokens[-2]) - prod) < 1e-9 \
                    and re.fullmatch(UNIT, tokens[-1]):
                return len(tokens) - 2              # lihat kasus volume di atas
            return i
    return find_rincian(tokens)


def split_tail(tokens: list[str], max_nums: int = 3):
    """Pisahkan angka di ekor baris: (token_depan, [nilai...], [token_mentah...])."""
    nums: list[tuple[str, int | float]] = []
    j = len(tokens)
    while j > 0 and len(nums) < max_nums and is_num(tokens[j - 1]):
        nums.insert(0, (tokens[j - 1], parse_num(tokens[j - 1])))
        j -= 1
    return tokens[:j], [v for _, v in nums], [t for t, _ in nums]


NUMISH = re.compile(r"[\d.,]+")
FRAG = re.compile(r"\d{1,3}")


def _join_ok(group: list[str]) -> bool:
    """Gabungan token hanya sah bila semua token kecuali yang terakhir adalah potongan digit murni."""
    return all(FRAG.fullmatch(t) for t in group[:-1])


def resolve_tail(toks: list[str]):
    """
    Urai ekor angka sebuah baris item -> (head, nums, raw, catatan|None).

    Ekstraksi PDF sering memecah angka: '1 1.760.000' (=11.760.000), '3 00' (=300), '1 0.000' (=10.000),
    '3 .000.000'. Fragmen digabung HANYA bila hasilnya membuat hitungan cocok:
      - 3 angka: volume x harga = jumlah
      - 2 angka: (rincian perhitungan) x harga = jumlah   (volume tertulis di rincian, mis. '12 bln')
      - 1 angka: subtotal kelompok yang terpecah ('5 76.000.000')
    Bila tak ada gabungan yang cocok, hasil baca apa adanya dikembalikan apa adanya.
    """
    base_head, base_nums, base_raw = split_tail(toks, 3)
    if len(base_nums) == 3 and all(v > 0 for v in base_nums) \
            and abs(base_nums[0] * base_nums[1] - base_nums[2]) < 0.5:
        return base_head, base_nums, base_raw, None, None

    j = len(toks)
    while j > 0 and len(toks) - j < 12 and NUMISH.fullmatch(toks[j - 1]):
        j -= 1
    tail = toks[j:]
    cands = []          # (jenis, gabungan, dilewati, groups, vals, head, indeks_rincian)
    for s in range(len(tail)):       # s = token awal ekor yang sebenarnya milik uraian/rincian
        sub = tail[s:]
        m = len(sub)
        for k in (3, 2, 1):
            if m < k:
                continue
            for cuts in combinations(range(1, m), k - 1):
                b = (0,) + cuts + (m,)
                parts = [sub[b[i]:b[i + 1]] for i in range(k)]
                if not all(_join_ok(p) for p in parts):
                    continue
                groups = ["".join(p) for p in parts]
                vals = [parse_num(g) for g in groups]
                if None in vals:
                    continue
                joins = m - k
                head = toks[:j + s]
                if k == 3 and abs(vals[0] * vals[1] - vals[2]) < 0.5:
                    cands.append((0, joins, s, groups, vals, head, None))
                elif k == 2:
                    ri = find_rincian_for(head, harga=vals[0], jumlah=vals[1])
                    if ri is not None:
                        _, prod, _ = rincian_info(" ".join(head[ri:]))
                        if prod is not None and abs(prod * vals[0] - vals[1]) < 0.5:
                            cands.append((1, joins, s, groups, vals, head, ri))
                            continue
                    if vals[0] == vals[1] and vals[0] >= 1000:      # harga satuan = jumlah (template lumpsum)
                        cands.append((2, joins, s, groups, vals, head, None))
                elif k == 1 and joins > 0 and _CONTENT_JOIN and find_rincian(head) is None and vals[0] >= 1000:
                    cands.append((3, joins, s, groups, vals, head, None))
    if not cands:
        return base_head, base_nums, base_raw, None, None
    _, joins, s, groups, vals, head, ri = min(cands, key=lambda c: (c[0], c[1], c[2]))
    note = None
    if joins:
        note = ("angka terpecah di PDF digabung menjadi " + " | ".join(groups)
                + f" (dari '{' '.join(tail[s:])}')")
    return head, vals, groups, note, ri


_CONTENT_JOIN = True


def set_content_join(on: bool) -> None:
    """PDF/XLSX: angka terpecah sudah digabung berdasar koordinat; menggabung berdasar isi justru
    menelan angka milik uraian ('Kendaraan Roda 2' + '35.140.000' -> 235.140.000)."""
    global _CONTENT_JOIN
    _CONTENT_JOIN = bool(on)


def split_amount_tail(tokens: list[str]):
    """Ekor SATU nominal yang mungkin terpecah (baris KRO/RO/komponen/akun): (head, nilai|None, raw|None)."""
    j = len(tokens)
    while j > 0 and len(tokens) - j < 4 and NUMISH.fullmatch(tokens[j - 1]):
        j -= 1
    tail = tokens[j:]
    for s in range(len(tail)):                    # gabungan terpanjang lebih dulu
        sub = tail[s:]
        if not _CONTENT_JOIN and len(sub) > 1:
            continue
        if _join_ok(sub):
            raw = "".join(sub)
            v = parse_num(raw)
            if v is not None:
                return tokens[:j + s], v, raw
    return tokens, None, None


_URAIAN_FACTOR_RE = re.compile(r"[\(\[]\s*(\d+(?:[.,]\d+)?)\s*[A-Za-z]+\.?\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*[A-Za-z]+\.?\s*[\)\]]")


def _uraian_product(uraian: Optional[str]):
    m = _URAIAN_FACTOR_RE.search(uraian or "")
    if not m:
        return None
    a, b = parse_num(m.group(1)), parse_num(m.group(2))
    return None if a is None or b is None else a * b


# Rumus yang ikut tertulis di sel uraian ('Kertas HVS F4 2 Rim x 6 Keg =') padahal sudah ada di kolom
# Rincian/Volume. Dibuang dari uraian HANYA bila hasil kali faktornya sama dengan produk rincian /
# volume (jadi '... A4 80 Gram' atau rumus yang berbeda dari rincian tidak ikut terhapus).
_EXPR = rf"(\d[\d.,]*\s+(?:x\s+)?{UNIT}(?:\s+x\s+\d[\d.,]*\s+{UNIT})*)"
# bentuk 1: 'Kertas 2 Rim x 6 Keg =' (boleh diikuti salinan rincian '12 RIM' yang terbawa ke uraian)
RUMUS_EQ_RE = re.compile(rf"\s*(?<![A-Za-z0-9.,]){_EXPR}\s*=\s*(?:\d[\d.,]*\s+{UNIT})?\s*$")
# bentuk 2: tanpa '=', wajib memuat minimal satu perkalian 'x' ('Sewa 2 Unit x 3 Hari'); '... 80 Gram' aman
RUMUS_X_RE = re.compile(rf"\s*(?<![A-Za-z0-9.,])(\d[\d.,]*\s+{UNIT}(?:\s+x\s+\d[\d.,]*\s+{UNIT})+)\s*$")


def strip_rumus_uraian(uraian: str, refs: list):
    """-> (uraian_bersih, rumus|None, ditemukan_tapi_beda: bool). refs = produk rincian dan/atau volume.
    Rumus dibuang dari uraian HANYA bila hasil kali faktornya sama dengan salah satu refs."""
    if not uraian or not refs:
        return uraian, None, False
    for rx, has_eq in ((RUMUS_EQ_RE, True), (RUMUS_X_RE, False)):
        m = rx.search(uraian)
        if not m:
            continue
        _, prod, _ = rincian_info(m.group(1))
        if prod is None:
            continue
        if any(abs(prod - r) < 1e-9 for r in refs):
            bersih = clean_text(uraian[:m.start()])
            if bersih:
                return bersih, m.group(1), False
        elif has_eq:
            return uraian, None, True          # ada '=' tapi hasil tidak sama: biarkan + beri peringatan
    return uraian, None, False


def rincian_info(rinc: Optional[str]):
    """-> (faktor[list[{jumlah,satuan}]], produk, satuan_akhir)"""
    if not rinc:
        return [], None, None
    fac = [(parse_num(a), b.rstrip(".")) for a, b in FACTOR_RE.findall(rinc)]
    fac = [(a, b) for a, b in fac if a is not None]
    if not fac:
        return [], None, None
    prod = 1
    for a, _ in fac:
        prod *= a
    return [{"jumlah": a, "satuan": b} for a, b in fac], prod, fac[-1][1]


# --------------------------------------------------------------------------------------
# Representasi baris
# --------------------------------------------------------------------------------------

@dataclass
class Line:
    text: str
    page: int = 1
    x0: Optional[float] = None          # posisi x token pertama (PDF) / indeks kolom (XLSX)
    kode_limit: Optional[float] = None  # batas kanan kolom Kode (jika header terdeteksi)
    src: str = "text"                   # pdf | xlsx | text  (pdf/xlsx: angka sudah digabung berdasar geometri)
    pw: Optional[float] = None          # lebar halaman (PDF) -> posisi relatif x0/pw
    top: Optional[float] = None         # geometri baris (PDF): dipakai merakit judul sel yang rata-tengah
    bottom: Optional[float] = None
    x1: Optional[float] = None          # ujung kanan kata terakhir
    title_pre: str = ""                 # teks judul yang berada DI ATAS baris kode (sel rata-tengah)
    jenis: Optional[str] = None         # nilai kolom Jenis Komponen (Utama/Pendukung) pada baris ini (PDF)
    text_with_jenis: Optional[str] = None   # teks baris bila kata Jenis TIDAK dikeluarkan (cadangan)
    keterangan: Optional[str] = None        # isi kolom KETERANGAN pada baris ini (PDF)
    kv: Optional[tuple[str, str]] = None    # kop: (label, nilai) yang dipisah berdasarkan GEOMETRI (lihat _kop_kv_split / _pair_kop_lines)

    def in_kode_col(self) -> bool:
        if self.x0 is None or self.kode_limit is None:
            return True
        return self.x0 < self.kode_limit


# --------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------

_VALID_AMT_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?")


def split_overlapped_numbers(page, words: list[dict]) -> list[dict]:
    """
    Dua angka yang tercetak saling menimpa (mis. subtotal '88.020.000' tepat di atas '8.640.000') dibaca
    pdfplumber sebagai SATU kata rusak ('8,0,62400,,000000') karena selisih y < y_tolerance. Kata numerik yang
    bentuknya mustahil dipecah ulang dari karakter aslinya menurut posisi y (selisih >= 0,5 pt).
    Sengaja TIDAK memakai y_tolerance kecil secara global: superskrip 'm²' / 'M2' akan terpisah dari satuannya.
    """
    out: list[dict] = []
    chars = None
    for w in words:
        t = w["text"]
        if len(t) >= 5 and re.fullmatch(r"[\d.,]+", t) and not _VALID_AMT_RE.fullmatch(t):
            if chars is None:
                chars = page.chars
            cs = [c for c in chars if c["x0"] >= w["x0"] - 0.6 and c["x1"] <= w["x1"] + 0.6
                  and c["top"] >= w["top"] - 2 and c["bottom"] <= w["bottom"] + 2 and c["text"].strip()]
            groups: list[list[dict]] = []
            for c in sorted(cs, key=lambda c: c["top"]):
                if groups and abs(c["top"] - groups[-1][0]["top"]) <= 0.45:
                    groups[-1].append(c)
                else:
                    groups.append([c])
            if len(groups) >= 2:
                for g in groups:
                    g.sort(key=lambda c: c["x0"])
                    nw = dict(w)
                    nw.update(text="".join(c["text"] for c in g), x0=g[0]["x0"], x1=g[-1]["x1"],
                              top=min(c["top"] for c in g), bottom=max(c["bottom"] for c in g))
                    out.append(nw)
                continue
        out.append(w)
    return out


def cluster_rows(words: list[dict], y_tol: float = 3.0) -> list[dict]:
    """
    Kelompokkan kata PDF menjadi baris. Versi lama memakai rata-rata y yang 'bergeser' (chaining) dan
    toleransi tetap 3 pt, sehingga dua baris tabel yang berjarak <= 3 pt (mis. baris Komponen/Akun yang
    selang 3,3 pt dari baris sebelumnya, atau sel yang rata-bawah) menyatu menjadi satu baris palsu.
    Sekarang:
      - jangkar = 'top' kata pertama baris (tidak bergeser), toleransi = 30% tinggi huruf (maks y_tol)
      - dua kata yang saling menimpa secara horizontal TIDAK mungkin satu baris -> pecah baris
    """
    rows: list[dict] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        h = max(1.0, w["bottom"] - w["top"])
        tol = min(y_tol, max(1.2, 0.30 * h))
        target = None
        for r in reversed(rows[-4:]):
            if abs(w["top"] - r["top"]) > tol:
                continue
            clash = any(min(w["x1"], o["x1"]) - max(w["x0"], o["x0"]) > 0.35 * min(w["x1"] - w["x0"], o["x1"] - o["x0"])
                        for o in r["words"])
            if not clash:
                target = r
                break
        if target is None:
            rows.append({"top": w["top"], "cy": w["cy"], "words": [w]})
        else:
            target["words"].append(w)
    return _fix_hierarchy_order(rows)


def _struct_rank(first: str) -> int:
    """Peringkat hierarki dari token pertama baris: KRO<RO<Komponen<Sub<Akun; 9 = bukan baris struktur."""
    if KRO_RE.match(first):
        return 1
    if RO_RE.match(first):
        return 2
    if COMP_FULL_RE.match(first) or KOMP_RE.match(first):
        return 3
    if SUBCODE_RE.match(first) or SUB_RE.match(first):
        return 4
    if AKUN_RE.match(first):
        return 5
    return 9


def _fix_hierarchy_order(rows: list[dict], near: float = 7.0) -> list[dict]:
    """
    Sel Komponen/Subkomponen sering rata-tengah pada sel setinggi beberapa baris, sehingga 'top'-nya
    sedikit LEBIH BAWAH dari baris Akun pertama di bawahnya. Urutan murni berdasarkan y lalu membalik
    hierarki (Akun muncul sebelum Komponen induknya). Baris yang berdekatan (<= near pt) ditukar bila
    peringkat hierarkinya terbalik.
    """
    def first_tok(r):
        ws = sorted(r["words"], key=lambda w: w["x0"])
        return ws[0]["text"] if ws else ""

    ranks = [_struct_rank(first_tok(r)) for r in rows]
    for _ in range(3):
        changed = False
        for i in range(len(rows) - 1):
            a, b = ranks[i], ranks[i + 1]
            if b < a <= 5 and abs(rows[i + 1]["top"] - rows[i]["top"]) <= near:
                rows[i], rows[i + 1] = rows[i + 1], rows[i]
                ranks[i], ranks[i + 1] = b, a
                changed = True
        if not changed:
            break
    return rows


def drop_occluded_text(page):
    """
    Beberapa RAB menimpa sel dengan teks baru tanpa menghapus teks lama: teks lama TETAP ada di lapisan teks
    (mis. 'Infrastruktur Pemerintah Digital Nasional' di bawah 'AAG - Peraturan Menteri'), tetapi tidak tampak.
    Bila dibaca per posisi, hasilnya acak ('AInAfGra -s tPreurkattuurra ...').
    Karakter dikelompokkan menjadi untai (run) = karakter berurutan di stream, font/ukuran sama, satu baris.
    Untai BERHURUF yang >= 30% hurufnya tertutup (>= 50% luas) oleh huruf yang digambar KEMUDIAN dibuang UTUH,
    termasuk bagian yang menjulur keluar dari area tumpang-tindih. Untai berisi angka tidak disentuh
    (subtotal yang dicetak menumpuk ditangani split_overlapped_numbers).
    """
    chars = page.chars
    if len(chars) < 2:
        return page
    runs: list[list[int]] = []
    for i, c in enumerate(chars):
        if runs:
            k = runs[-1][-1]
            pc = chars[k]
            if (pc["fontname"] == c["fontname"] and abs(pc["size"] - c["size"]) < 0.05
                    and abs(pc["top"] - c["top"]) < 0.3 and -0.5 <= c["x0"] - pc["x1"] <= 3 * c["size"]):
                runs[-1].append(i)
                continue
        runs.append([i])
    run_of: dict[int, int] = {}
    for r, idxs in enumerate(runs):
        for i in idxs:
            run_of[i] = r
    textual = []
    for idxs in runs:
        vis = [chars[i]["text"] for i in idxs if chars[i]["text"].strip()]
        textual.append(bool(vis) and sum(t.isalpha() for t in vis) >= 0.7 * len(vis))
    grid: dict[tuple, list[int]] = {}
    for i, c in enumerate(chars):
        if c["text"].isalpha() and textual[run_of[i]]:
            grid.setdefault((int(c["x0"] // 12), int(c["top"] // 12)), []).append(i)
    covered: dict[int, set[int]] = {}
    for j, cj in enumerate(chars):
        if not (cj["text"].isalpha() and textual[run_of[j]]):
            continue
        aj = max(cj["x1"] - cj["x0"], 0.1) * max(cj["bottom"] - cj["top"], 0.1)
        gx, gy = int(cj["x0"] // 12), int(cj["top"] // 12)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in grid.get((gx + dx, gy + dy), ()):
                    if i >= j or run_of[i] == run_of[j]:
                        continue
                    ci = chars[i]
                    ix = min(ci["x1"], cj["x1"]) - max(ci["x0"], cj["x0"])
                    iy = min(ci["bottom"], cj["bottom"]) - max(ci["top"], cj["top"])
                    if ix <= 0 or iy <= 0:
                        continue
                    ai = max(ci["x1"] - ci["x0"], 0.1) * max(ci["bottom"] - ci["top"], 0.1)
                    if ix * iy >= 0.25 * min(ai, aj):
                        covered.setdefault(run_of[i], set()).add(i)
    dropped: set[int] = set()
    for r, cov in covered.items():
        n_alpha = sum(1 for i in runs[r] if chars[i]["text"].isalpha())
        if n_alpha and len(cov) >= 0.3 * n_alpha:
            dropped.update(id(chars[i]) for i in runs[r])
    if not dropped:
        return page
    return page.filter(lambda o: o.get("object_type") != "char" or id(o) not in dropped)


_TTE_STAMP_RE = re.compile(r"^\s*(?:Catatan\s*:|-\s*UU\s+ITE\b|[\"“]?\s*Informasi\s+Elektronik\b|"
                           r"-\s*Dokumen\s+ini\s+telah\b)", re.I)


def drop_tte_stamp(page):
    """Stempel catatan TTE ('Catatan : - UU ITE No 11 Tahun 2008 ...' / '"Informasi Elektronik ...' /
    '- Dokumen ini telah ditandatangani ...') kadang dicetak MENIMPA baris tabel di dasar halaman. Bila dibaca
    per posisi, huruf keduanya bercampur ('1T1r aTnahsupno 2r0t0 P8P' = '11 Tahun 2008' + 'Transport PP') dan
    seluruh baris dibuang sebagai catatan kaki -> item hilang (BIF.002 hal. 6). Baris stempel di bawahnya
    ('"Informasi Elektronik ...') juga bisa turun menimpa bagian bawah huruf tabel, sehingga drop_occluded_text
    membuang isi tabel itu.
    Baris stempel = untai yang cocok _TTE_STAMP_RE + untai lain berfont/ukuran sama di ketinggian yang sama
    (lanjutan 'sertifikat elektronik yang diterbitkan BSrE'). Baris itu DIBUANG UTUH hanya bila kotak hurufnya
    BERTUMPUK (>= 0,5pt tegak) dengan >= 3 huruf/angka non-stempel di rentang x-nya. Stempel yang sekadar
    bersebelahan dengan teks lain (blok TTD) dibiarkan dan tetap disaring sebagai catatan kaki."""
    chars = page.chars
    if len(chars) < 2:
        return page
    runs: list[list[int]] = []
    for i, c in enumerate(chars):
        if runs:
            pc = chars[runs[-1][-1]]
            if (pc["fontname"] == c["fontname"] and abs(pc["size"] - c["size"]) < 0.05
                    and abs(pc["top"] - c["top"]) < 0.3 and -0.5 <= c["x0"] - pc["x1"] <= 3 * c["size"]):
                runs[-1].append(i)
                continue
        runs.append([i])
    heads = [r for r in runs if _TTE_STAMP_RE.match("".join(chars[i]["text"] for i in r))]
    if not heads:
        return page
    def fam(c):        # 'Helvetica' / 'Helvetica-Bold' / 'ABCDEF+Helvetica-Oblique' -> 'helvetica'
        return c["fontname"].split("+")[-1].split("-")[0].split(",")[0].lower()

    groups: list[list[int]] = []
    for h in heads:
        h0 = chars[h[0]]
        g = [i for r in runs
             if fam(chars[r[0]]) == fam(h0) and abs(chars[r[0]]["size"] - h0["size"]) < 0.05
             and abs(chars[r[0]]["top"] - h0["top"]) < 0.5 for i in r]
        groups.append(g)
    in_stamp = {i for g in groups for i in g}
    dropped: set[int] = set()
    for g in groups:
        gx0, gx1 = min(chars[i]["x0"] for i in g), max(chars[i]["x1"] for i in g)
        gt, gb = min(chars[i]["top"] for i in g), max(chars[i]["bottom"] for i in g)
        hit = 0
        for j, c in enumerate(chars):
            if j in in_stamp or not c["text"].isalnum():
                continue
            if gx0 < (c["x0"] + c["x1"]) / 2 < gx1 and min(gb, c["bottom"]) - max(gt, c["top"]) >= 0.5:
                hit += 1
                if hit >= 3:
                    break
        if hit >= 3:
            dropped.update(id(chars[i]) for i in g)
    if not dropped:
        return page
    return page.filter(lambda o: o.get("object_type") != "char" or id(o) not in dropped)


# Kolom 'Jenis Komponen (Utama/Pendukung)': nilainya dibaca berdasarkan POSISI kolom (bukan posisi di
# ujung teks), dikeluarkan dari teks baris, lalu disimpan di Line.jenis -> field 'jenis_komponen'.
# Batas kolom diambil dari kepala tabel ('Jenis' / '(Utama/' / 'Pendukung)'); halaman lanjutan tanpa
# kepala tabel memakai batas halaman sebelumnya (lebar halaman harus sama).
# Kepala kolom Jenis: '(Utama/', 'Utama/', '(UTAMA/PENDUK', 'Pendukung)', '(Utama/Pendukung)', atau hanya
# 'Jenis Komponen' (sebaris / bertumpuk). 'Utama'/'Pendukung' polos adalah NILAI kolom.
_JENIS_HDR_RE = re.compile(r"^(\(utama|utama/)", re.I)
_JENIS_HDR_ANY_RE = re.compile(r"^(jenis|\(utama.*|utama/.*|\(?pendukung\)|\(?penduk\)?)$", re.I)
_JENIS_VAL_RE = re.compile(r"^(utama|pendukung|pendukun|penduk)$", re.I)


def _jenis_col_range(words: list[dict]) -> Optional[tuple[float, float, float]]:
    """(x0, x1, bottom kepala) kolom Jenis Komponen, atau None."""
    anchors = [w for w in words if _JENIS_HDR_RE.match(w["text"])]
    for j in words:                                      # 'Jenis' + 'Komponen' sebaris/bertumpuk
        if j["text"].lower() != "jenis":
            continue
        for k in words:
            if k["text"].lower() != "komponen":
                continue
            dy = k["top"] - j["top"]
            if (abs(dy) <= 3 and 0 <= k["x0"] - j["x1"] <= 6) or                     (3 < dy <= 15 and min(j["x1"], k["x1"]) - max(j["x0"], k["x0"]) > 0):
                anchors += [j, k]
    if not anchors:
        return None
    a = min(anchors, key=lambda w: w["top"])
    near = [w for w in words if (w in anchors or _JENIS_HDR_ANY_RE.match(w["text"]))
            and -3 <= w["top"] - a["top"] <= 30
            and min(w["x1"], a["x1"]) - max(w["x0"], a["x0"]) > -15]
    return min(w["x0"] for w in near), max(w["x1"] for w in near), max(w["bottom"] for w in near)


def _in_phrase(w: dict, words: list[dict], gap: float = 3.0) -> bool:
    """Kata menempel (< gap pt) ke kata huruf di kirinya pada baris yang sama -> bagian kalimat uraian
    ('Tenaga Pendukung Lainnya'), bukan sel kolom Jenis yang selalu berjarak dari kolom di kirinya."""
    for o in words:
        if o is w or abs(o["top"] - w["top"]) > 2 or not re.search(r"[A-Za-z]", o["text"]):
            continue
        if 0 <= w["x0"] - o["x1"] < gap:
            return True
    return False


def split_jenis_column(words: list[dict], col: tuple[float, float, Optional[float]],
                       tol: float = 10.0) -> tuple[list[dict], list[dict]]:
    """Pisahkan kata di kolom Jenis Komponen. Hasil: (kata sisa, kata jenis {'jenis', 'cy'}).
    Kepala kolom ('(Utama/', 'Pendukung)') dibuang tanpa dicatat. Angka yang menempel ke kata itu
    (mis. 'pendukun1g' = 'pendukung' + volume '1') dipertahankan sebagai kata sendiri."""
    lo, hi, hdr_bottom = col[0] - tol, col[1] + tol, col[2]
    rest, jen = [], []
    for w in words:
        cx = (w["x0"] + w["x1"]) / 2
        if lo <= cx <= hi and not _in_phrase(w, words):
            t = w["text"]
            letters = re.sub(r"[^A-Za-z()/]", "", t)
            is_val = bool(_JENIS_VAL_RE.match(letters))
            if is_val or _JENIS_HDR_ANY_RE.match(letters):
                digits = re.sub(r"\D", "", t)
                if digits:
                    rest.append({**w, "text": digits})
                row_txt = [o["text"].lower() for o in words if abs(o["top"] - w["top"]) <= 2]
                in_body = (hdr_bottom is None or w["top"] > hdr_bottom + 1) and                     not all(t in HEADER_VOCAB for t in row_txt)
                if is_val and in_body:
                    v = "Utama" if letters.lower() == "utama" else "Pendukung"
                    jen.append({"jenis": v, "cy": (w["top"] + w["bottom"]) / 2})
                continue
        rest.append(w)
    return rest, jen


# Kolom 'Data Dukung' / 'Dokumen Pendukung' / 'Data Pendukung' (biasanya paling kanan, berisi tautan
# Google Drive, nama berkas, dsb.) juga tidak diurai: semua kata di kolom itu dibuang saat memuat PDF.
_DUKUNG_A_RE = re.compile(r"^(data|dokumen|dok\.?)$", re.I)
_DUKUNG_B_RE = re.compile(r"^(dukung|pendukung)$", re.I)
_AMOUNT_TOK_RE = re.compile(r"^[-(]?(rp\.?)?[\d.,]+\)?(,-)?$", re.I)


_THOUSANDS_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+")
_HDR_WORDS = {"kode", "uraian", "program", "kegiatan", "output", "akun", "volume", "vol", "satuan", "sat", "harga",
              "jumlah", "total", "rincian", "perhitungan", "komponen", "subkomponen", "keg", "detil", "detail",
              "biaya", "standar", "suboutput", "keterangan", "kro", "ro"}
_AMT_FULL_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?:,\d{2})?")     # nominal utuh, boleh ',00'
_AMOUNT_HDR_RE = re.compile(r"^(jumlah|jml\.?|jlh|harga|satuan|pagu|biaya|total|\(?rp\.?\)?)$", re.I)


def _dukung_col_start(words: list[dict]) -> Optional[tuple[float, bool]]:
    """(x awal kolom Data Dukung, boleh_luber) dari kepala tabel: 'DATA DUKUNG' sebaris atau bertumpuk.
    boleh_luber = kolom tepat di kirinya kolom teks (mis. KETERANGAN) yang isinya bisa meluber ke kanan."""
    for a in words:
        if not _DUKUNG_A_RE.match(a["text"]):
            continue
        for b in words:
            if not _DUKUNG_B_RE.match(b["text"]):
                continue
            dy = b["top"] - a["top"]
            same_row = abs(dy) <= 3 and 0 <= b["x0"] - a["x1"] <= 15
            stacked = 3 < dy <= 20 and min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) > 0
            if same_row or stacked:
                start = min(a["x0"], b["x0"])
                left = [w for w in words if abs(w["top"] - a["top"]) <= 3 and w["x1"] <= start]
                prev = max(left, key=lambda w: w["x1"]) if left else None
                return start, bool(prev and not _AMOUNT_HDR_RE.match(prev["text"]))
    return None


def drop_dukung_column(words: list[dict], start: float, spill_ok: bool = False,
                       slack: float = 30.0, gap: float = 4.0) -> list[dict]:
    """Buang isi kolom Data Dukung. Nominal kolom Jumlah yang rata kanan bisa melewati sedikit batas
    kepala kolom, jadi di zona 'slack' sebelum batas hanya token non-angka yang dibuang.
    Bila kolom di kirinya kolom teks (spill_ok), frasa yang dimulai di kiri zona dan meluber ke kanan
    tanpa jeda (kata-kata rapat) tidak dipotong."""
    # Hanya di dalam tabel: blok tanda tangan ('Jakarta, 2 Februari 2026', nama, NIP) sering berada tepat
    # di bawah kolom Data Dukung. Batas bawah tabel = baris terakhir yang memuat nominal berpemisah ribuan
    # di kiri kolom (+ sedikit ruang untuk tautan yang terbungkus ke baris berikutnya).
    amt_bottoms = [w["bottom"] for w in words if w["x0"] < start and _THOUSANDS_RE.search(w["text"])]
    table_bottom = max(amt_bottoms) + 8 if amt_bottoms else None
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        for r in rows:
            if abs(r[0]["top"] - w["top"]) <= 2:
                r.append(w)
                break
        else:
            rows.append([w])
    keep_ids = set()
    for r in rows:
        r.sort(key=lambda w: w["x0"])
        prev_kept = None
        for w in r:
            spill = (spill_ok and prev_kept is not None and w["x0"] - prev_kept["x1"] < gap)
            below_table = table_bottom is None or w["top"] > table_bottom
            if below_table or spill or w["x0"] < start - slack or (w["x0"] < start - 3 and _AMOUNT_TOK_RE.match(w["text"])):
                keep_ids.add(id(w))
                prev_kept = w
            else:
                prev_kept = None
    return [w for w in words if id(w) in keep_ids]


_KET_HDR_RE = re.compile(r"^KETERANGAN$", re.I)
_GLUED_AMT_RE = re.compile(r"^(-?\d{1,3}(?:[.,]\d{3})+(?:,\d{2})?-?)(\S+)$")


def _ket_col_start(words: list[dict]) -> Optional[tuple[float, float]]:
    """(batas kiri kolom KETERANGAN, bawah kepala) = ujung kanan kolom Jumlah: nominal berpemisah ribuan
    terjauh yang berakhir sebelum kepala 'KETERANGAN'. Hanya bila kolom itu memang paling kanan di tabel
    nominal (kolom teks bebas yang suka menempel ke angka)."""
    hdr = [w for w in words if _KET_HDR_RE.match(w["text"])]
    if not hdr:
        return None
    h = hdr[0]
    amts = [w for w in words if w["top"] > h["bottom"] and _THOUSANDS_RE.search(w["text"]) and w["x0"] < h["x0"]]
    if not amts:
        return None
    ends = []
    for w in amts:
        m = _GLUED_AMT_RE.match(w["text"])
        if m:
            # '8.214.000un': ujung kanan nominal diperkirakan dari proporsi karakter
            ends.append(w["x0"] + (w["x1"] - w["x0"]) * len(m.group(1)) / len(w["text"]))
        elif re.fullmatch(r"-?(?:Rp\.?)?\d{1,3}(?:[.,]\d{3})+(?:,\d{2})?-?", w["text"], re.I):
            ends.append(w["x1"])
    if not ends:
        return None
    return max(ends) + 0.5, h["bottom"]


def split_ket_column(words: list[dict], col: tuple[float, Optional[float]]) -> tuple[list[dict], list[dict]]:
    """Pisahkan isi kolom KETERANGAN (di kanan batas, di dalam badan tabel). Kata yang menempel ke nominal
    ('8.214.000un') dipecah. Hasil: (kata sisa, kata keterangan)."""
    start, hdr_bottom = col
    amt_bottoms = [w["bottom"] for w in words if w["x0"] < start and _THOUSANDS_RE.search(w["text"])]
    if not amt_bottoms:
        return words, []
    table_bottom = max(amt_bottoms) + 2
    keep, ket = [], []
    for w in words:
        in_body = (hdr_bottom is None or w["top"] > hdr_bottom) and w["top"] <= table_bottom
        if not in_body or w["x1"] <= start:
            keep.append(w)
            continue
        m = _GLUED_AMT_RE.match(w["text"])
        if w["x0"] < start and m:
            cut = w["x0"] + (w["x1"] - w["x0"]) * len(m.group(1)) / len(w["text"])
            keep.append({**w, "text": m.group(1), "x1": cut})
            ket.append({**w, "text": m.group(2), "x0": cut})
        elif w["x0"] >= start - 1:
            ket.append(w)
        else:
            keep.append(w)
    return keep, ket


def _has_table_header(words: list[dict]) -> bool:
    return any(w["text"].lower() == "kode" for w in words) and any(w["text"].lower() == "uraian" for w in words)


# --------------------------------------------------------------------------------------
# Kop (blok 'Label : Nilai' di atas tabel) yang label & nilainya TIDAK sebaris
# --------------------------------------------------------------------------------------
# Sebagian RAB (terutama hasil masking ulang) mencetak sel label rata-tengah/rata-bawah, sehingga nilai
# (': Program Dukungan Manajemen', top 88.7) berada ~4,5 pt DI ATAS labelnya ('Kegiatan', top 93.2).
# cluster_rows memisahkannya menjadi dua baris, urutannya nilai -> label, dan parser lama menempelkan
# setiap nilai ke label SEBELUMNYA (salah satu langkah) atau ke baris pra-tabel. Pemasangan di sini murni
# geometri: label-saja <-> nilai-saja yang saling TERDEKAT secara vertikal. Kop yang sudah sebaris tidak
# tersentuh (tidak ada baris label-saja / nilai-saja).

def _kop_hdr_top(words: list[dict]) -> Optional[float]:
    """top kepala tabel ('KODE' sebaris dengan 'URAIAN ...' / 'Program/Kegiatan/...') di halaman ini; None bila
    tidak ada."""
    ur = [w for w in words if w["text"].lower().startswith(("uraian", "program/"))]
    tops = [w["top"] for w in words if w["text"].lower() == "kode"
            and any(abs(u["top"] - w["top"]) <= 4 for u in ur)]
    return min(tops) if tops else None


def _kop_colon_x(rows: list[dict], hdr_top: Optional[float]) -> Optional[float]:
    """x kolom titik dua kop: ':' berdiri sendiri (atau ':nilai') yang sekolom (+-2 pt) di >= 3 baris kop."""
    if hdr_top is None:
        return None
    xs = sorted(w["x0"] for r in rows if r["top"] < hdr_top for w in r["words"] if w["text"].startswith(":"))
    best: list[float] = []
    for x in xs:
        grp = [y for y in xs if abs(y - x) <= 2]
        if len(grp) > len(best):
            best = grp
    return best[len(best) // 2] if len(best) >= 3 else None


def _meta_label_key(label: str) -> Optional[str]:
    """Kunci metadata bila SELURUH teks adalah label kop (tanpa nilai). Sisa sesudah pola label yang masih
    berada di kolom label = bagian label yang meluber/tersamar ('Unit Eselon I/Satker Kegiatan',
    'Unit Eselon II[MASKED DATA]'): diterima hanya bila berupa placeholder masking dan/atau satu kata huruf."""
    t = clean_text(label)
    if not t or ":" in t:
        return None
    for key, rx, _req in META_RES:
        m = rx.match(t)
        if not m:
            continue
        rest = MASK_ANY_RE.sub(" ", m.group(2)).strip()
        if not rest or re.fullmatch(r"[A-Za-z]+", rest):
            return key
    return None


def _kop_kv_split(ws: list[dict], colon_x: float) -> Optional[tuple[str, str]]:
    """Baris kop TANPA titik dua ('Unit Eselon I/Satker Kegiatan [MASKED DATA]': placeholder masking menimpa
    ': nilai' tepat di kolom titik dua) -> (label, nilai) dipotong di kolom titik dua. Hanya bila bagian kiri
    memang label metadata yang dikenal."""
    if any(":" in w["text"] for w in ws) or ws[0]["x0"] > colon_x - 40:
        return None
    # placeholder yang menempel ke kata label ('Kegiatan[MASKED', jeda < x_tolerance) dan melintasi kolom
    # titik dua: pecah di '[' bila perkiraan posisinya (proporsi huruf) memang di kolom itu
    sp: list[dict] = []
    for w in ws:
        j = w["text"].find("[", 1)
        if j > 0 and w["x0"] < colon_x - 1.5 < w["x1"]:
            cut = w["x0"] + (w["x1"] - w["x0"]) * j / len(w["text"])
            if abs(cut - colon_x) <= 6:
                sp += [{**w, "text": w["text"][:j], "x1": cut}, {**w, "text": w["text"][j:], "x0": max(cut, colon_x)}]
                continue
        sp.append(w)
    ws = sp
    k = next((i for i, w in enumerate(ws) if w["x0"] >= colon_x - 1.5), None)
    if not k:
        return None
    lab = clean_text(" ".join(w["text"] for w in ws[:k]))
    val = clean_text(" ".join(w["text"] for w in ws[k:]))
    if not val or not _meta_label_key(lab):
        return None
    return lab, val


def _pair_kop_lines(lines: list[Line], colon_x: Optional[float], hdr_top: Optional[float]) -> list[Line]:
    """Kop dengan label & nilai di baris terpisah:
      - label-saja ('Kegiatan' / 'Kegiatan :') di kolom label,
      - nilai-saja (': Program Dukungan Manajemen' / 'Penguatan Lembaga Pers') di kolom nilai (>= kolom ':').
    Setiap baris nilai dipasang ke label yang pusat vertikalnya TERDEKAT, bila selisihnya <= tinggi huruf.
    Satu label boleh mendapat beberapa baris nilai (nilai 2 baris yang mengapit labelnya, SIMS 202).
    Baris gabungan menempati posisi paling atas di antara anggotanya, sehingga baris lanjutan (wrap) nilai
    tetap sesudahnya. Label tanpa baris nilai -- padahal nilai di atas & di bawahnya milik label lain --
    ditandai kosong (kv nilai '') agar parser tidak mengambil baris berikutnya sebagai nilainya (UKW: IKK)."""
    if colon_x is None or hdr_top is None or len(lines) < 2:
        return lines
    labels, values = [], []
    for i, l in enumerate(lines):
        if l.top is None or l.bottom is None or l.x0 is None or l.bottom > hdr_top + 1 or l.kv is not None:
            continue                     # hanya baris yang seluruhnya DI ATAS kepala tabel
        t = clean_text(l.text)
        if l.x0 >= colon_x - 2:
            if (t[1:].strip() if t.startswith(":") else t) and not IKK_NUM_RE.match(t):
                values.append(i)
            continue
        lab = re.sub(r"\s*:$", "", t)    # 'Kegiatan :' (titik dua menempel di ujung label)
        right = colon_x + 6 if t.endswith(":") else colon_x - 2
        if l.x1 is not None and l.x1 < right and l.x0 < colon_x - 40 and _meta_label_key(lab):
            labels.append(i)
    if not labels or not values:
        return lines

    def cy(i):
        return (lines[i].top + lines[i].bottom) / 2

    def h(i):
        return lines[i].bottom - lines[i].top

    got: dict[int, list[int]] = {lb: [] for lb in labels}
    owner: dict[int, int] = {}
    for v in values:
        lb = min(labels, key=lambda j: abs(cy(v) - cy(j)))
        if abs(cy(v) - cy(lb)) <= max(h(v), h(lb), 1.0):
            got[lb].append(v)
            owner[v] = lb
    if not owner:
        return lines
    repl: dict[int, Line] = {}
    drop: set[int] = set()
    for lb, vs in got.items():
        L = lines[lb]
        lab = re.sub(r"\s*:$", "", clean_text(L.text))
        if not vs:
            above = [v for v in values if cy(v) < cy(lb)]
            below = [v for v in values if cy(v) > cy(lb)]
            nb = max(above, key=cy) if above else None
            na = min(below, key=cy) if below else None
            if (nb is None or nb in owner) and (na is None or na in owner):
                repl[lb] = Line(**{**L.__dict__, "kv": (lab, "")})
            continue
        val = clean_text(" ".join(re.sub(r"^:\s*", "", clean_text(lines[v].text)) for v in vs))
        members = [lb] + vs
        first = min(members)
        repl[first] = Line(**{**L.__dict__, "text": f"{lab} : {val}",
                              "top": min(lines[m].top for m in members),
                              "bottom": max(lines[m].bottom for m in members),
                              "x1": max(lines[m].x1 or 0 for m in members), "kv": (lab, val)})
        drop.update(m for m in members if m != first)
    return [repl.get(i, l) for i, l in enumerate(lines) if i not in drop]


def _attach_kop_masks(lines: list[Line], hdr_top: Optional[float]) -> list[Line]:
    """Placeholder masking di ujung baris kop yang tercetak sedikit lebih rendah dari baris aslinya
    ('Persentase (%) penanganan kasus hukum di lingkungan' top 92.5 + '[MASKED DATA]' top 94.7) dipisah
    cluster_rows menjadi baris sendiri. Sambungkan kembali ke baris yang SEJAJAR (pusat vertikal <= setengah
    tinggi huruf) dan berakhir tepat sebelum placeholder (jeda 0..8 pt)."""
    if hdr_top is None or len(lines) < 2:
        return lines
    out = list(lines)
    i = 0
    while i < len(out):
        m = out[i]
        if m.top is None or m.bottom is None or m.x0 is None or m.bottom > hdr_top + 1 \
                or not MASK_RE.fullmatch(clean_text(m.text)):
            i += 1
            continue
        mc, mh = (m.top + m.bottom) / 2, m.bottom - m.top
        host = None
        for j in (i - 1, i + 1):
            if 0 <= j < len(out):
                p = out[j]
                if p.top is None or p.bottom is None or p.x1 is None or MASK_RE.fullmatch(clean_text(p.text)):
                    continue
                if abs((p.top + p.bottom) / 2 - mc) <= 0.5 * max(mh, p.bottom - p.top) \
                        and 0 <= m.x0 - p.x1 <= 8:
                    host = j
                    break
        if host is None:
            i += 1
            continue
        p = out[host]
        mt = clean_text(m.text)
        out[host] = Line(**{**p.__dict__, "text": f"{p.text} {mt}", "x1": m.x1,
                            "top": min(p.top, m.top), "bottom": max(p.bottom, m.bottom),
                            "kv": (p.kv[0], clean_text(f"{p.kv[1]} {mt}")) if p.kv else None})
        del out[i]
        i = max(0, min(i, host))
    return out


def load_pdf_lines(path: Path, y_tol: float = 3.0) -> list[Line]:
    import pdfplumber

    out: list[Line] = []
    dukung_x: Optional[tuple[float, bool]] = None
    jenis_col: Optional[tuple[float, float, Optional[float]]] = None
    jenis_pw: Optional[float] = None
    dukung_pw: Optional[float] = None
    ket_col: Optional[tuple[float, Optional[float]]] = None
    ket_pw: Optional[float] = None
    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            try:
                page = page.dedupe_chars(tolerance=1)      # huruf yang dicetak dua kali di titik yang sama ('MMAASSKKEEDD')
            except Exception:
                pass
            page = drop_tte_stamp(page)            # sebelum drop_occluded_text: stempel jangan 'menutupi' isi tabel
            page = drop_occluded_text(page)
            words = split_overlapped_numbers(page, page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False))
            if not words:
                continue
            pw = float(page.width)
            hdr = _has_table_header(words)
            # Kolom Jenis / Data Dukung terbawa ke halaman lanjutan (tanpa kepala tabel). Batasnya direset
            # bila muncul kepala tabel baru tanpa kolom itu, atau lebar halaman berganti (RAB lain).
            rng = _jenis_col_range(words)
            if rng is not None:
                jenis_col, jenis_pw = rng, pw
            elif hdr or (jenis_pw is not None and abs(jenis_pw - pw) > 1):
                jenis_col = jenis_pw = None
            elif jenis_col is not None:
                jenis_col = (jenis_col[0], jenis_col[1], None)     # halaman lanjutan: tak ada kepala
            jenis_words: list[dict] = []
            if jenis_col is not None:
                words, jenis_words = split_jenis_column(words, jenis_col)
            dx = _dukung_col_start(words)
            if dx is not None:
                dukung_x, dukung_pw = dx, pw
            elif hdr or (dukung_pw is not None and abs(dukung_pw - pw) > 1):
                dukung_x = dukung_pw = None
            if dukung_x is not None:
                words = drop_dukung_column(words, *dukung_x)
            kc = _ket_col_start(words)
            if kc is not None:
                ket_col, ket_pw = kc, pw
            elif hdr or (ket_pw is not None and abs(ket_pw - pw) > 1):
                ket_col = ket_pw = None
            elif ket_col is not None:
                ket_col = (ket_col[0], None)                       # halaman lanjutan: tak ada kepala
            ket_words: list[dict] = []
            if ket_col is not None:
                words, ket_words = split_ket_column(words, ket_col)
            if not words:
                continue
            for w in words:
                w["cy"] = (w["top"] + w["bottom"]) / 2
            rows = cluster_rows(words, y_tol)
            # nilai Jenis -> baris terdekat; sel yang rata-tengah di antara baris wrap -> baris di atasnya
            row_jenis: dict[int, str] = {}
            for jw in jenis_words:
                best = min(range(len(rows)), key=lambda i: abs(_row_cy(rows[i]) - jw["cy"]), default=None)
                if best is None:
                    continue
                if abs(_row_cy(rows[best]) - jw["cy"]) > 2 * y_tol:
                    above = [i for i in range(len(rows)) if _row_cy(rows[i]) < jw["cy"]]
                    if above:
                        best = max(above, key=lambda i: _row_cy(rows[i]))
                row_jenis.setdefault(best, jw["jenis"])
            # isi KETERANGAN -> baris terdekat (teks sel yang terbungkus beberapa baris digabung)
            row_ket: dict[int, list[dict]] = {}
            for kw in ket_words:
                kcy = (kw["top"] + kw["bottom"]) / 2
                best = min(range(len(rows)), key=lambda i: abs(_row_cy(rows[i]) - kcy), default=None)
                if best is not None:
                    row_ket.setdefault(best, []).append(kw)
            # batas kolom Kode dari baris header "Kode ... Uraian"
            kode_limit = None
            for r in rows:
                ws = sorted(r["words"], key=lambda w: w["x0"])
                names = {w["text"].lower(): w for w in ws}
                if "kode" in names and "uraian" in names:
                    kode_limit = (names["kode"]["x0"] + names["uraian"]["x0"]) / 2
                    break
            # kop di atas kepala tabel: kolom titik dua untuk memisah label/nilai berdasarkan posisi
            kop_top = _kop_hdr_top(words)
            colon_x = _kop_colon_x(rows, kop_top)
            page_out: list[Line] = []
            for ri, r in enumerate(rows):
                ws = sorted(r["words"], key=lambda w: w["x0"])

                # pdfplumber kadang memecah angka yang secara visual merupakan SATU token:
                #   3 + 4.785.564.000  -> 34.785.564.000
                #   5 + .134.457.000   -> 5.134.457.000
                # Pecahan tersebut dapat dikenali dari posisi x: x0 token berikutnya
                # tepat/nyaris menempel dengan x1 token sebelumnya. Jangan melakukan
                # penggabungan hanya berdasarkan isi token karena itu dapat mencampur
                # kolom volume dengan harga/jumlah.
                merged: list[dict] = []
                numeric_fragment = re.compile(r"^[+-]?(?:\d[\d.,]*|[.,]\d[\d.,]*)$")
                for w in ws:
                    can_join_numeric = (
                        merged
                        and (w["x0"] - merged[-1]["x1"]) <= 0.75
                        and bool(numeric_fragment.fullmatch(str(merged[-1]["text"])))
                        and bool(numeric_fragment.fullmatch(str(w["text"])))
                    )
                    if can_join_numeric:
                        merged[-1]["text"] += w["text"]
                        merged[-1]["x1"] = max(merged[-1]["x1"], w["x1"])
                    else:
                        merged.append(dict(w))
                text = clean_text(" ".join(w["text"] for w in merged))
                if text:
                    # hanya baris yang seluruhnya di atas kepala tabel (bukan baris 'KODE URAIAN ...' itu sendiri)
                    kv = _kop_kv_split(merged, colon_x) \
                        if colon_x is not None and max(w["top"] for w in merged) < kop_top - 1 else None
                    page_out.append(Line(text, pno, merged[0]["x0"], kode_limit, "pdf", float(page.width),
                                    top=min(w["top"] for w in merged), bottom=max(w["bottom"] for w in merged),
                                    x1=max(w["x1"] for w in merged), jenis=row_jenis.get(ri),
                                    text_with_jenis=_text_with_jenis(merged, row_jenis.get(ri), jenis_col),
                                    keterangan=clean_text(" ".join(
                                        w["text"] for w in sorted(row_ket.get(ri, []), key=lambda w: (round(w["top"]), w["x0"]))))
                                    or None, kv=kv))
            out.extend(_pair_kop_lines(_attach_kop_masks(page_out, kop_top), colon_x, kop_top))
    return out


def _text_with_jenis(merged: list[dict], jenis: Optional[str], col) -> Optional[str]:
    if not jenis or col is None:
        return None
    cx = (col[0] + col[1]) / 2
    left = [w["text"] for w in merged if (w["x0"] + w["x1"]) / 2 < cx]
    right = [w["text"] for w in merged if (w["x0"] + w["x1"]) / 2 >= cx]
    return clean_text(" ".join(left + [jenis] + right))


def _row_cy(r: dict) -> float:
    ws = r["words"]
    return sum((w["top"] + w["bottom"]) / 2 for w in ws) / len(ws)


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if float(v).is_integer():
            return f"{int(v):,}".replace(",", ".")
        return str(v).replace(".", ",")
    return clean_text(str(v))


def load_xlsx_lines(path: Path) -> list[Line]:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    out: list[Line] = []
    for pno, ws in enumerate(wb.worksheets, 1):
        kode_limit = None
        for row in ws.iter_rows():
            cells = [(c.column, _fmt_cell(c.value)) for c in row]
            cells = [(ci, t) for ci, t in cells if t]
            if not cells:
                continue
            if kode_limit is None:
                names = {t.lower(): ci for ci, t in cells}
                if "kode" in names and "uraian" in names:
                    kode_limit = (names["kode"] + names["uraian"]) / 2
            text = clean_text(" ".join(t for _, t in cells))
            out.append(Line(text, pno, float(cells[0][0]), kode_limit, "xlsx"))
    return out


def load_text_lines(path: Path) -> list[Line]:
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", raw) and "-" in raw:
            continue  # pemisah tabel markdown
        text = clean_text(raw.replace("|", " "))
        if text:
            out.append(Line(text))
    return out


def load_lines(path: Path) -> list[Line]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_lines(path)
    if ext in (".xlsx", ".xlsm"):
        return load_xlsx_lines(path)
    return load_text_lines(path)


# --------------------------------------------------------------------------------------
# Model pohon
# --------------------------------------------------------------------------------------

LEVELS = {"kro": 1, "ro": 2, "komponen": 3, "subkomponen": 4, "akun": 5, "kelompok": 6}

KRO_RE = re.compile(r"^\d{4}\.[A-Z]{3}$")
RO_RE = re.compile(r"^\d{4}\.[A-Z]{3}\.\d{3}$")
KOMP_RE = re.compile(r"^\d{3}$")
COMP_FULL_RE = re.compile(r"^\d{4}\.[A-Z]{3}\.\d{3}\.\d{3}$")
SUBCODE_RE = re.compile(r"^\d{3}\.[0-9][A-Z]$")


# baris tanpa uraian: '<volume> <satuan> <nominal...>' (mis. '1 Layanan 2,017,894,000')
_NO_URAIAN_RE = re.compile(r"[1-9]\d*(?:[.,]\d+)?\s+[A-Za-z]+\.?(?:\s+[\d.,]+)+")


def _amount_only_tail(text: str) -> bool:
    """Teks memuat nominal >= 1000 (bakal menjadi item/jumlah)."""
    return any((v := parse_num(t)) is not None and v >= 1000 for t in text.split())


def _amount_only(rest: list[str]) -> bool:
    """Sisa baris hanya satu nominal (boleh terpecah: '1 1,346,597,000') tanpa uraian."""
    head, v, _ = split_amount_tail(rest)
    return not head and v is not None and v >= 1000
SUB_RE = re.compile(r"^[A-Z]$")
AKUN_RE = re.compile(r"^\d{6}$")
MASK_RE = re.compile(r"^\[MASKED[^\]]*\]$", re.I)
LABEL_RE = re.compile(r"^([a-z])[.)]?\s+(\S.*)$")
# kelompok tingkat atas ('A Pembinaan Kearsipan', 'B Pengelolaan ...'): huruf KAPITAL tunggal.
# Posisinya di atas akun, sedangkan kelompok huruf kecil ('a Jakarta - Medan') tetap anak akun.
UPPER_LABEL_RE = re.compile(r"^([A-Z])(?:[.)]?\s+|[.)](?=[A-Z][a-z]))([A-Z0-9]\S*(?:\s.*)?)$")   # 'B. Judul' / 'G.Judul'
_LOKASI_RE = re.compile(r"^Lokasi\s*:\s*(\S.*)$", re.I)


def node_level(node) -> float:
    if node.tipe == "kelompok" and node.label and node.label.isupper() and not node.in_akun:
        return 4.5
    return LEVELS[node.tipe]
JENIS_RE = re.compile(r"\b(Utama|Pendukung)$", re.I)

JENIS_BELANJA = {"51": "Belanja Pegawai", "52": "Belanja Barang", "53": "Belanja Modal"}

HEADER_VOCAB = {
    "kode", "uraian", "suboutput/komponen/subkomponen/akun/detail", "volume", "sub", "output",
    "jenis", "komponen", "kopone", "n", "(utama/", "utama/", "penduk", "pendukung", "pendukung)",
    "(utama/pendukung)", "rincian", "perhitungan", "satuan", "harga", "jumlah", "sat", "jml", "jml.",
    "detil", "detail", "subkomponen/", "komponen/", "akun/", "(", ")", "( rp )", "(rp)", "rp", "biaya",
    "suboutput", "suboutput/komponen/subkomponen/detil", "an", "e", "volum", "perhitung", "uraian",
    # kepala kolom yang terpecah per baris ('KEG/OUTPU VOLUME STANDAR' / 'T BIAYA' / '/ AKUN UMUM')
    "keg/outpu", "keg/output", "t", "/", "akun", "umum", "standar", "g",
}

# baris total keseluruhan tanpa kata 'JUMLAH' ('TOTAL 2.449.072.000', 'GRAND TOTAL Rp 1.000.000')
_BARE_TOTAL_RE = re.compile(r"^(?:GRAND\s+)?TOTAL(?:\s+(?:BIAYA|ANGGARAN|KESELURUHAN))?\s*:?\s*(?:Rp\.?\s*)?"
                            r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?(?:,-)?$", re.I)


def is_header_row(text: str) -> bool:
    toks = text.lower().split()
    if not toks:
        return False
    if all(re.fullmatch(r"[-\u2013\u2014]+", t) for t in toks):          # baris pemisah ' -'
        return True
    if len(toks) >= 2 and all(re.fullmatch(r"\d", t) for t in toks):    # nomor kolom '1 3 4 6 7'
        return True
    if all(t in HEADER_VOCAB for t in toks):
        return True
    up = text.upper()
    return bool(re.match(r"^(KODE|PROGRAM\s*/|PERHITUNGAN\s+TAHUN)", up)
                and re.search(r"PERHITUNGAN|KEGIATAN\s*/\s*OUTPUT|URAIAN|STANDAR\s+BIAYA|SATUAN\s+BIAYA", up))


NOISE_RES = [
    re.compile(r"^Kode\b.*\bUraian\b", re.I),
    re.compile(r"^RINCIAN\s+ANGGARAN\s+(BIAYA|BELANJA)", re.I),
    re.compile(r"^RINCIAN\s+BELANJA\s+SATUAN\s+KERJA", re.I),        # judul 'RINCIAN BELANJA SATUAN KERJA TA 2026'
    re.compile(r"^Uraian\s+Rincian\s+Output\s*/", re.I),              # kepala kolom 'Uraian Rincian Output/ Komponen/...'

    re.compile(r"^Sat\s+Jml\b", re.I),
    re.compile(r"^HARGA\s+SATUAN", re.I),
    re.compile(r"^Volume\s+Sub\s*Output", re.I),
    re.compile(r"^Rincian\s+Perhitungan", re.I),
    re.compile(r"^Jenis\s+Komponen", re.I),
    re.compile(r"^\(?Utama\s*/", re.I),
    re.compile(r"^Pendukung\)$", re.I),
    re.compile(r"^(Halaman|Page|Hal\.)\s*\d+", re.I),
    re.compile(r"^\d{1,4}\s+(Halaman|Page|Hal\.?)\s*\d+(\s*(/|of|dari)\s*\d+)?$", re.I),   # footer '994 Page 2'
    re.compile(r"^(Halaman|Page)\s*\d+\s*(/|of|dari)\s*\d+$", re.I),
    re.compile(r"^Catatan\s*:?\s*$", re.I),                                              # blok catatan TTE
    re.compile(r"^-?\s*UU\s+ITE\b", re.I),
    re.compile(r"^[\"\u201c]?\s*I\s?nformasi\s+Elektronik\b", re.I),
    re.compile(r"^-?\s*Dokumen\s+ini\s+telah\s+ditandatangani", re.I),
    re.compile(r"^Output/Komponen/Subkomponen", re.I),
    re.compile(r"^(Suboutput|Uraian\s+Suboutput)\b.*\b(Detil|Detail)\b", re.I),
    re.compile(r"^(Rincian|Jenis)\s+Komponen\b", re.I),
]

# Kode/angka dalam kurung setelah label: 'Kegiatan (7450) :', 'Sasaran Kegiatan (01) :', 'Program (059.GB) :',
# 'Klasifikasi Rincian Output (7450.SCA) :', 'Indikator KRO (Opsional) :'
_CD = r"(?:\s*\(\s*[\w.\-/ ]{1,20}\s*\))?"
_ESL = r"(?:I{1,3}|[1-3])"
META_DEFS = [
    ("kementerian", r"Kementerian\s*(?:Negara)?\s*/\s*Lembaga", False),
    # Unit: 'Unit Eselon II/Satker', 'Unit Eselon I/II', 'Unit Kerja Eselon I', 'Unit Kerja', 'Satuan Kerja', 'Satker'
    ("unit_eselon2_satker",
     rf"(?:Unit\s+Kerja(?:\s+Eselon\s+{_ESL}(?:\s*/\s*{_ESL})?)?"
     rf"|Unit\s+Eselon\s+{_ESL}(?:\s*/\s*(?:{_ESL}|Sa[kt]{{1,2}}er|Satuan\s+Kerja))?"
     rf"|Unit\s+Organisasi|Satuan\s+Kerja(?:\s+Eselon\s+{_ESL})?)", False),
    ("unit_eselon2_satker", r"Satker", True),
    ("sasaran_program", r"Sasaran\s+Program" + _CD, False),
    ("indikator_kinerja_program", r"Indikator\s+Kinerja\s+Program" + _CD, False),
    ("program", r"Program" + _CD, True),
    ("sasaran_kegiatan", r"Sasaran\s+Kegiatan" + _CD, False),
    ("indikator_kinerja_kegiatan", r"Indikator\s+Kinerja(?:\s+Kegiatan)?(?:\s*\(IKK\))?", False),
    ("kegiatan", r"Kegiatan" + _CD, True),
    # IKRO harus sebelum KRO (label IKRO memuat frasa KRO)
    ("ikro", r"Indikator\s+(?:Klasifikasi\s+Rincian\s+Output|KRO)" + _CD, False),
    ("ikro", r"IKRO", True),
    ("iro", r"Indikator\s+(?:Rincian\s+Output|RO)" + _CD, False),
    ("iro", r"IRO", True),
    # 'Indikator Keluaran (Output) Kegiatan : ...' / 'Keluaran (output) Kegiatan Komunikasi Publik'
    ("iro", r"Indikator\s+Keluaran\s*\(\s*Output\s*\)(?:\s+Kegiatan)?", False),
    ("ro", r"Keluaran\s*\(\s*Output\s*\)(?:\s+Kegiatan)?", False),
    # KRO: 'Klasifikasi Rincian Output (KRO)', '... (7450.SCA)', 'Rincian Output (KRO)', 'Klasifikasi Output', 'KRO :'
    ("kro", r"(?:Kla[sr]ifikasi\s+Rincian\s*Output|Rincian\s+Output\s*\(\s*KRO\s*\)|Klasifikasi\s+Output)" + _CD, False),
    ("kro", r"\(?KRO\)?", True),
    # RO: 'Rincian Output (RO)', 'Rincian Output :', 'RO :', 'Output Kegiatan :'
    ("ro", r"Rincian\s*Output\s*\(\s*RO\s*\)", False),
    ("ro", r"(?:Rincian\s+Output|Output\s+Kegiatan|\(?RO\)?)", True),
    ("kode_header", r"Kode", True),
    ("komponen", r"Komponen", True),
    # 'Target/Satuan : 479 Orang' = volume + satuan dalam satu nilai (satuan_ukur diisi dari satuan yang menempel)
    ("volume", r"Target\s*/\s*Satuan", False),
    # Volume: 'Volume', 'Volume Output', 'Volume Keluaran', 'Volume RO', 'Target Volume', 'Jumlah Volume', 'Vol.'
    ("volume",r"(?:(?:Target|Jumlah)\s+Volume|Vol(?:ume|\.)?)"
               r"(?:\s+(?:Output|Keluaran|Kegiatan|Rincian\s+Output|RO|KRO|Target))?(?:\s*\(\s*(?:RO|KRO|Output)\s*\))?", True),
    # Satuan: 'Satuan Ukur', 'Satuan Ukur Keluaran (Output)', 'Satuan Rincian Output (RO)', 'Sat. Ukur'
    ("satuan_ukur", r"(?:Satuan(?:\s+(?:Ukur(?:an)?|Keluaran|Output|Volume|Rincian|Klasifikasi|K?RO\b))+(?:\s*\(\s*[A-Za-z]{2,10}\s*\))?"
                    r"|Sat\.?\s*Ukur(?:an)?(?:\s+(?:Keluaran|Output))*(?:\s*\(\s*[A-Za-z]{2,10}\s*\))?)", False),
    ("satuan_ukur", r"Satuan", True),
    ("alokasi_dana", r"Alokasi\s+Dana", False),
]
META_RES = [
    (k, re.compile(rf"^(?:{pat})\s*(:)?\s*(.*)$", re.I), req)
    for k, pat, req in META_DEFS
]
# kunci teks bebas yang nilainya boleh melanjut ke baris berikutnya (wrap)
META_CONT_KEYS = {"sasaran_kegiatan", "indikator_kinerja_kegiatan", "ro", "kro", "kegiatan", "ikro", "iro", "komponen",
                  "program", "sasaran_program", "indikator_kinerja_program"}
# kunci baru: hanya dikenali sebelum tabel (agar tidak menyaru jadi uraian item)
META_PRE_ONLY = {"sasaran_kegiatan", "indikator_kinerja_kegiatan", "ro", "ikro", "iro", "program", "sasaran_program",
                 "indikator_kinerja_program", "komponen", "kode_header"}
# label yang boleh kosong (mis. 'Indikator KRO (Opsional) :'): nilainya TIDAK dicari di baris berikutnya
META_NO_WAIT = {"ikro", "iro", "program", "sasaran_program", "indikator_kinerja_program", "komponen", "kode_header"}
# indikator bernomor ("1. Persentase ...", ": 2. Nilai ...") -> daftar, bukan teks bebas / baris pra-tabel
#   '4.Indeks Kepuasan ...' (tanpa spasi sesudah titik) juga butir baru bila langsung diikuti huruf kapital
IKK_NUM_RE = re.compile(r"^:?\s*\d{1,2}\.(?:\s+|(?=[A-Z]))(\S.*)$")
# penanda butir IKK tanpa nomor urut: ': ...' (KIP), 'IKK 02. ...' / 'IKK01.' (BIF), '- ...' / '• ...'
IKK_MARK_RE = re.compile(r"^(?::\s*|[-•–]\s+|IKK\s?\d{1,2}\b)")
# akhir baris yang pasti berlanjut ke baris berikutnya (kata sambung / tanda baca)
_WRAP_END_RE = re.compile(r"(?:\b(?:di|pada|dan|yang|untuk|dari|ke|atau|terhadap|dengan|serta|oleh|bagi|"
                          r"tentang|dalam|melalui|sebagai|per|antar)|[,(/&-])\s*$", re.I)
# label kop yang (di template RAB) tercetak SESUDAH Indikator Kinerja Kegiatan: butir bernomor di bawahnya
# adalah lanjutan nilai label itu (mis. 'Indikator Rincian Output : 1.Jumlah ...' / '2. Capaian ...'), bukan IKK
_AFTER_IKK_KEYS = {"kro", "ikro", "ro", "iro", "volume", "satuan_ukur", "alokasi_dana", "komponen", "kode_header"}


def _split_ikk_block(texts: list[str]) -> list[str]:
    """Baris nilai Indikator Kinerja Kegiatan (baris pertama = nilai di baris label, sisanya baris lanjutan di
    kolom nilai) -> daftar butir.
      - Bila ada baris lanjutan berpenanda (': ', 'IKK 0n', '- '): baris berpenanda = butir baru, baris lain = wrap.
      - Tanpa penanda: butir baru bila baris diawali huruf KAPITAL, > 2 kata, dan baris sebelumnya tidak
        berakhir dengan kata sambung ('di', 'pada', 'dan', ...); selain itu wrap ('2026 di[MASKED DATA]',
        '[MASKED DATA]', 'di Sekretariat KIP', 'Sekretariat KIP')."""
    marked = any(IKK_MARK_RE.match(t) for t in texts[1:])
    items = [re.sub(r"^:\s*", "", texts[0])]
    for t in texts[1:]:
        if marked:
            new = bool(IKK_MARK_RE.match(t))
        else:
            new = t[:1].isupper() and len(t.split()) > 2 and not _WRAP_END_RE.search(items[-1])
        t = re.sub(r"^:\s*", "", t)
        if new:
            items.append(t)
        else:
            items[-1] = clean_text(f"{items[-1]} {t}")
    return [clean_text(x) for x in items if clean_text(x)]

# Penyamaran data: '[MASKED_X]', '*****', 'XXXX', '#####', '.......', '_____'
MASK_ANY_RE = re.compile(r"\[\s*MASK[^\]]*\]|[xX*#_]{4,}|(?<![\w])[.\-]{5,}(?![\w])", re.I)
# Label NIP: 'NIP', 'N I P', 'N.I.P', 'Nip.', 'Nomor Induk Pegawai' (bukan bagian kata lain)
NIP_LABEL_RE = re.compile(
    r"(?<![A-Za-z])(?:N\s?\.?\s?I\s?\.?\s?P|Nomor\s+Induk\s+Pegawai)(?![A-Za-z])\.?", re.I)
# Jabatan penandatangan (jangkar cadangan blok tanda tangan)
JABATAN_RE = re.compile(
    r"\b(Pejabat\s+Pembuat\s+Komitmen|Kuasa\s+Pengguna\s+Anggaran|Pengguna\s+Anggaran|Penanggung\s+Jawab|"
    r"Ketua\s+Tim|Kepala\s+\w+|Sekretaris|Direktur|Koordinator|PPK|KPA)\b", re.I)


def classify_nip_value(rest: str) -> tuple[str, int]:
    """Status nilai NIP setelah label: kosong | tersamar | terbaca_valid | terbaca_tidak_standar."""
    r = (rest or "").strip(" :.-\u2013\u2014\t")
    if not r:
        return "kosong", 0
    if MASK_ANY_RE.search(r):
        return "tersamar", 0
    digits = re.sub(r"\D", "", r)
    if len(digits) >= 8:
        return ("terbaca_valid" if len(digits) == 18 else "terbaca_tidak_standar"), len(digits)
    return "kosong", len(digits)


def redact_digits(t: str, n: int = 60) -> str:
    return re.sub(r"\d", "#", t)[:n]


def nip_value(rest: str, placeholder: bool = False) -> Optional[str]:
    """Nilai NIP: placeholder/teks tersamar apa adanya; angka -> hanya digit ('19800101 200501 1 001' -> 18 digit)."""
    r = clean_text(rest or "").strip(" :.-\u2013\u2014\t")
    if not r:
        return None
    if placeholder or MASK_ANY_RE.search(r):
        return r
    d = re.sub(r"\D", "", r)
    return d if len(d) >= 8 else None


TTD_RE = re.compile(r"^([A-Za-z][A-Za-z .]+?),\s*((?:\d{1,2}\s+)?[A-Za-z]+\s+\d{4})$")


@dataclass
class Node:
    tipe: str
    kode: Optional[str]
    uraian: str
    label: Optional[str] = None
    jumlah_tercantum: Optional[int | float] = None
    volume_suboutput: Optional[int | float] = None
    harga_satuan: Optional[int | float] = None     # node bervolume x harga = jumlah ('A PKN Tk. II 60 11.075.000 664.500.000')
    jenis_komponen: Optional[str] = None
    halaman: Optional[int] = None
    items: list[dict] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    parent: Optional["Node"] = field(default=None, repr=False)
    placeholder: bool = field(default=False, repr=False)   # baris nominal yang judulnya tersamar/tak terbaca
    x0: Optional[float] = field(default=None, repr=False)  # posisi kiri baris judul (indentasi)
    in_akun: bool = field(default=False, repr=False)       # kelompok berlabel kapital yang menjorok di dalam akun
    keterangan: Optional[str] = None                        # isi kolom KETERANGAN pada baris judul
    satuan_suboutput: Optional[str] = None                  # '3 KEGIATAN' -> 'KEGIATAN'
    lokasi: Optional[str] = None                            # 'Lokasi : Jawa Barat' di bawah judul node ini

    def to_dict(self) -> dict:
        d = {
            "tipe": self.tipe,
            "kode": self.kode,
            "label": self.label,
            "uraian": self.uraian,
            "jumlah_tercantum": self.jumlah_tercantum,
            "halaman": self.halaman,
        }
        if self.volume_suboutput is not None:
            d["volume_suboutput"] = self.volume_suboutput
        if self.satuan_suboutput:
            d["satuan_suboutput"] = self.satuan_suboutput
        if self.harga_satuan is not None:
            d["harga_satuan"] = self.harga_satuan
        if self.jenis_komponen:
            d["jenis_komponen"] = self.jenis_komponen
        if self.keterangan:
            d["keterangan"] = self.keterangan
        if self.lokasi:
            d["lokasi"] = self.lokasi
        if self.tipe == "akun" and self.kode:
            d["jenis_belanja"] = JENIS_BELANJA.get(self.kode[:2])
        d["items"] = self.items
        d["children"] = [c.to_dict() for c in self.children]
        return d


# --------------------------------------------------------------------------------------
# Parser utama
# --------------------------------------------------------------------------------------

_QUOTE_PREFIX = "'\u2018\u2019`\u00b4\"\u201c\u201d\u200b\u200c\u200d\ufeff\u00a0"


def _strip_code_prefix(toks: list[str]) -> list[str]:
    """Buang tanda kutip/karakter tak terlihat di depan kode ('4485.EBA.962 -> 4485.EBA.962).
    Umum pada ekspor Excel (awalan ' agar kode terbaca teks). Hanya bila sisanya berawalan angka
    dan tokennya memang berbentuk kode (angka/titik/huruf kapital), supaya kutipan di uraian aman."""
    if not toks:
        return toks
    t = toks[0]
    # akun BAS yang ditulis bertitik ('524.111 Belanja Perjalanan Dinas Biasa') -> '524111'. Hanya bila
    # diikuti uraian huruf, supaya nominal '524.111' di awal baris tidak ikut diubah.
    if re.fullmatch(r"5\d{2}\.\d{3}", t) and len(toks) > 1 and re.match(r"[A-Za-z]", toks[1]):
        return [t.replace(".", "")] + toks[1:]
    # akun yang menempel ke uraian ('521211Belanja Bahan', '524.111Belanja ...') -> ['521211', 'Belanja', ...]
    mg = re.fullmatch(r"(5\d{2})\.?(\d{3})([A-Za-z].*)", t)
    if mg:
        return [mg.group(1) + mg.group(2), mg.group(3)] + toks[1:]
    c = t.lstrip(_QUOTE_PREFIX)
    if c != t and c and c[0].isdigit() and re.fullmatch(r"\d[\d.A-Z]*", c):
        return [c] + toks[1:]
    return toks


def _node_amount(n: "Node") -> Optional[float]:
    if n.jumlah_tercantum is not None:
        return float(n.jumlah_tercantum)
    if not n.items and not n.children:
        return None
    s = 0.0
    for it in n.items:
        if it.get("jumlah") is None:
            if it.get("harga_satuan") is None:
                continue            # baris teks/rincian tanpa nominal ('Business standard 1 user,') = 0
            return None
        s += float(it["jumlah"])
    for c in n.children:
        a = _node_amount(c)
        if a is None:
            return None
        s += a
    return s


def _leaf_sum(n: "Node") -> float:
    """Jumlah nominal item-item ASLI di bawah n (rekursif), tanpa memakai subtotal tercantum anak-anaknya:
    judul rincian bernominal ('- Pengukuran QoS ... 1.506.655.000' = nominal akun) tidak membuat akun 'lengkap'."""
    return sum(float(it.get("jumlah") or 0) for it in n.items) + sum(_leaf_sum(c) for c in n.children)


def _akun_open(a: Optional["Node"]) -> bool:
    """Akun bernominal yang item-item aslinya belum mencapai nominal tercantum (rinciannya masih berlanjut)."""
    if a is None or a.tipe != "akun" or a.jumlah_tercantum is None:
        return False
    t = float(a.jumlah_tercantum)
    return _leaf_sum(a) < t - _tol(t)


def _tol(target: float) -> float:
    """Toleransi pembulatan dokumen (mis. selisih Rp100 pada 1,36 M dari item ber-desimal)."""
    return max(1.0, abs(target) * 1e-6)


_LIFT_ROOTS: list = []          # daftar node teratas RAB yang sedang diproses (diisi RabParser.finish)


def _lift_confirmed(n: "Node", siblings: list["Node"], i: int, moved_amts: list[float],
                    first_moved: "Node", n_target: float) -> bool:
    """Anak yang akan diangkat keluar dari n hanya dipindah bila aritmetika di level atas membenarkannya:
      (a) induk n bernominal dan isinya TEPAT cocok sesudah pemindahan, atau
      (b) anak pertama yang dipindah adalah judul kelompok bernominal T dan saudara-saudara sesudah n
          (satu atau lebih) berjumlah TEPAT T  -> judul yang tercetak di atas akunnya ('Perawatan Gedung
          Kantor 435.225.000' lalu '523111 ... 435.225.000'), atau
      (c) yang dipindah adalah node sementara (judul tersamar) bernominal.
    Tanpa bukti itu (mis. rincian ulang satu paket 'Pengembangan SIMS Generasi Baru' yang dipecah lagi),
    struktur dibiarkan seperti urutan dokumen."""
    p = n.parent
    if p is not None and p.jumlah_tercantum is not None:
        tot = 0.0
        for c in p.children:
            a = n_target if c is n else _node_amount(c)
            if a is None:
                tot = None
                break
            tot += a
        if tot is not None:
            for it in p.items:
                tot += float(it.get("jumlah") or 0)
            pt = float(p.jumlah_tercantum)
            if abs(tot + sum(moved_amts) - pt) <= _tol(pt):
                return True
    if first_moved.placeholder:
        return True
    if first_moved.tipe == "kelompok" and first_moved.jumlah_tercantum is not None:
        t = float(first_moved.jumlah_tercantum)
        # saudara sesudah n, lalu sesudah tiap leluhurnya (pengangkatan bisa lebih dari satu tingkat)
        x, seq = n, siblings[i + 1:]
        while True:
            s = 0.0
            for c in seq:
                a = _node_amount(c)
                if a is None:
                    break
                s += a
                if abs(s - t) <= _tol(t):
                    return True
                if s > t + _tol(t):
                    break
            x = x.parent
            if x is None:
                break
            lst = x.parent.children if x.parent is not None else _LIFT_ROOTS
            seq = lst[lst.index(x) + 1:] if x in lst else []
        # judul kelompok yang sudah berisi akun-akunnya sendiri (isinya cocok dengan nominalnya)
        own = _node_amount(Node("kelompok", None, "", items=first_moved.items, children=first_moved.children)) \
            if (first_moved.items or first_moved.children) else None
        if own is not None and abs(own - t) <= _tol(t):
            return True
    return False


def _lift_overflow(children: list["Node"]) -> None:
    """Anak yang 'kebablasan' masuk ke node sebelumnya (mis. kelompok '* Gaji dan Tunjangan PPPK' tergantung
    di bawah akun 512411 milik grup PNS). Bila item + k anak pertama node TEPAT sama dengan jumlah tercantumnya
    sedangkan seluruh isinya melebihi, anak ke-k dst. diangkat menjadi saudara sesudah node itu."""
    for c in children:
        _lift_overflow(c.children)
    # item yang kebablasan (mis. daftar 'BBSDMP Medan 1 Paket 421.923.000' milik komponen tertangkap sebagai
    # item kelompok '# Bandung'): bila k item pertama TEPAT = jumlah tercantum, item ke-k dst. naik ke induk.
    for n in children:
        if n.jumlah_tercantum is None or n.children or not n.items or n.parent is None:
            continue
        js = [it.get("jumlah") for it in n.items]
        target = float(n.jumlah_tercantum)
        if None in js or abs(sum(js) - target) <= _tol(target):
            continue
        run, cut = 0.0, None
        for k in range(len(js)):
            run += float(js[k])
            if abs(run - target) <= _tol(target):
                cut = k + 1
        if cut is not None and cut < len(js):
            # hanya bila induknya ikut cocok sesudah item diangkat; kalau tidak, ini salah hitung di
            # dokumen (akun 25.000.000 berisi ATK 25.000.000 + Souvenir 22.500.000) -> biarkan.
            p = n.parent
            if p.jumlah_tercantum is not None:
                pc = _node_amount(Node(p.tipe, None, "", items=p.items, children=p.children))
                lifted = sum(float(x) for x in js[cut:])
                if pc is None or abs(pc + lifted - float(p.jumlah_tercantum)) > _tol(float(p.jumlah_tercantum)):
                    continue
            n.parent.items.extend(n.items[cut:])
            n.items = n.items[:cut]
    # diulang sampai stabil: pengangkatan dari node sesudahnya (mis. baris tersamar di akun 521213) bisa
    # menjadi bukti aritmetika bagi pengangkatan node sebelumnya (judul 'B ...' di akun 524119).
    for _round in range(4):
        if not _lift_pass(children):
            break
    _nest_groups(children)          # kelompok yang baru terangkat langsung menelan isinya sebelum level atas dicek


def _lift_pass(children: list["Node"]) -> bool:
    lifted_any = False
    i = 0
    while i < len(children):
        n = children[i]
        if n.jumlah_tercantum is not None and n.children:
            target = float(n.jumlah_tercantum)
            base = 0.0
            ok = True
            for it in n.items:
                if it.get("jumlah") is None:
                    if it.get("harga_satuan") is not None:
                        ok = False          # nominal ada tapi jumlah tak terbaca: jangan ubah apa pun
                        break
                    continue                # baris rincian tanpa nominal ('- ATK 1 Paket') = 0
                base += float(it["jumlah"])
            amts = [_node_amount(c) for c in n.children]
            if ok and None not in amts and abs(base + sum(amts) - target) > _tol(target):
                cut, run = None, base
                for k in range(len(amts)):
                    if abs(run - target) <= _tol(target) and (k > 0 or n.items):
                        cut = k                     # pilih k terbesar yang cocok
                    c = n.children[k]
                    # anak kelompok tanpa nominal yang menampung item milik level atas ('# Bandung' + daftar
                    # 'BBSDMP ... 1 Paket'): potong di dalam item-nya bila prefiksnya menggenapi target.
                    if cut is None and run + amts[k] > target + _tol(target) and c.jumlah_tercantum is None \
                            and not c.children and c.items and n.parent is not None:
                        r2 = run
                        for m_ in range(len(c.items)):
                            r2 += float(c.items[m_]["jumlah"])
                            if abs(r2 - target) <= _tol(target) and m_ + 1 < len(c.items):
                                n.parent.items.extend(c.items[m_ + 1:])
                                c.items = c.items[:m_ + 1]
                                amts[k] = _node_amount(c)
                                cut = k + 1
                                break
                        if cut is not None:
                            break
                    run += amts[k]
                # kelebihan yang TEPAT sama dengan jumlah node = rincian ulang (elaborasi) nominal yang sama
                # ('Kebutuhan Tenaga Ahli' = 1.080.000.000 di bawah akun 1.080.000.000), bukan isi node lain.
                if cut is not None and cut < len(amts) and abs(sum(amts[cut:]) - target) <= _tol(target):
                    cut = None
                if cut is not None and cut < len(amts) and not _lift_confirmed(n, children, i, amts[cut:],
                                                                                n.children[cut], target):
                    cut = None              # tidak ada bukti aritmetika di level atas: biarkan (mis. rincian/elaborasi)
                if cut is not None and cut < len(amts):
                    moved = n.children[cut:]
                    n.children = n.children[:cut]
                    for x in moved:
                        x.parent = n.parent
                    children[i + 1:i + 1] = moved
                    lifted_any = True
        i += 1
    return lifted_any


def _computed_items_to_titles(nodes: list["Node"]) -> None:
    """'Perawatan Kendaraan Dinas Roda 4 45.000.000' terbaca sebagai item (volume 4 x harga 45 jt, jumlah
    dihitung 180 jt) padahal judul kelompok bernominal 45 jt yang dicetak di atas akunnya. Bila item terakhir
    sebuah node berjumlah HASIL HITUNG, tanpa rincian, dan tanpa item itu isi node TEPAT sama dengan jumlah
    tercantumnya -> jadikan kelompok anak (judul + angka kecil, nominal = 'harga'), lalu _lift_overflow /
    _nest_groups yang menaruhnya sesuai aritmetika."""
    for n in nodes:
        _computed_items_to_titles(n.children)
        if not n.items or n.jumlah_tercantum is None:
            continue
        target = float(n.jumlah_tercantum)
        for k_ in range(len(n.items) - 1, -1, -1):
            it = n.items[k_]
            if it.get("jumlah_sumber") != "dihitung" or it.get("rincian_perhitungan") \
                    or it.get("harga_satuan") is None:
                continue
            before = n.items[:k_]
            if any(x.get("jumlah") is None for x in before) or n.children and k_ < len(n.items) - 1:
                continue
            s = sum(float(x["jumlah"]) for x in before) + sum((_node_amount(c) or 0.0) for c in n.children)
            if abs(s - target) > _tol(target):
                continue
            # item sesudahnya ('Biaya pemeliharaan kendaraan operasional roda 2 ... 35.000.000') = isi judul ini
            after = n.items[k_ + 1:]
            n.items = before
            v = it.get("volume")
            vtxt = str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)
            k = Node("kelompok", None, clean_text(f"{it.get('uraian') or ''} {vtxt}"), label=it.get("label"),
                     jumlah_tercantum=it["harga_satuan"], halaman=it.get("halaman"), parent=n, items=after)
            n.children.append(k)
            break


PLACEHOLDER_URAIAN = "[uraian tersamar/tidak terbaca]"
_LABEL_ONLY_RE = re.compile(r"\(?(?:\d{1,2}|[A-Za-z]|[IVX]{1,4})[.)]")
_PROGRAM_ROW_RE = re.compile(r"^(?:\d{3}\.)?([A-Z]{2})\s+Program\s+(\S.*?)(?:\s+(?:Rp\.?\s*)?\d{1,3}(?:[.,]\d{3})+)?$")
_URL_TOK_RE = re.compile(r"\S*(?:https?:|htps?:|drive\.google|docs\.google|bit\.ly|www\.)\S*", re.I)
# '... 81.000 9.720.000SBM 2026' -> teks s.d. nominal | keterangan
_GLUED_KET_RE = re.compile(r"^(.*(?:^|\s)\d{1,3}(?:[.,]\d{3})+(?:,\d{2})?)([A-Za-z]{2}[^\d]*(?:(?:19|20)\d{2}[^\d]*)?)$")
_HEAD_LABEL_RE = re.compile(r"^(?:[A-Z]|[IVX]{1,4}|\d{1,2})[.)]?\s+[A-Z]")
_ROMAN_HEAD_RE = re.compile(r"^(?:I{1,3}|IV|VI{0,3}|IX|X)[.)]?\s+[A-Z][a-z]")


def _prune_placeholders(children: list["Node"]) -> int:
    """Buang node placeholder yang tidak menampung apa pun (ternyata hanya pengulangan subtotal)."""
    n = 0
    for c in children:
        n += _prune_placeholders(c.children)
    keep = [c for c in children if not (c.placeholder and not c.items and not c.children)]
    n += len(children) - len(keep)
    children[:] = keep
    return n


def _nest_groups(children: list["Node"]) -> None:
    """Kelompok bertingkat. Parser hanya mengenal satu tingkat kelompok, sehingga judul kelompok induk
    ('Jakarta 20.900.000' -> '*Golongan IV 5.700.000', '*Golongan III 15.200.000'; atau
    '* Gaji dan Tunjangan PNS 31.890.347.000' -> akun-akun) menjadi saudara yang KOSONG dari isinya.
    Kelompok kosong bernominal T menelan saudara-saudara berikutnya bila jumlah mereka TEPAT T
    (aritmetika dokumen sendiri yang menentukan; tanpa kecocokan persis, struktur tidak diubah)."""
    for c in children:
        _nest_groups(c.children)
    # dari kanan ke kiri: rantai judul bernominal sama ('[tersamar] 1.742.160.000' -> '[MASKED] 1.742.160.000'
    # -> 'Sewa Kendaraan ... 1.742.160.000' + item) tersusun dari yang paling dalam lebih dulu.
    i = len(children) - 1
    while i >= 0:
        k = children[i]
        own = None
        # baris RO berkode 3 digit ('956 Layanan Kepatuhan Internal ...', '994 Layanan Perkantoran') terbaca
        # sebagai komponen KOSONG yang sejajar dengan komponen-komponennya: jadikan RO bila jumlah komponen
        # sesudahnya TEPAT sama.
        if k.tipe == "komponen" and k.jumlah_tercantum and not k.items and not k.children:
            target, s, j = float(k.jumlah_tercantum), 0.0, i + 1
            while j < len(children) and children[j].tipe == "komponen" and s < target - _tol(target):
                a = _node_amount(children[j])
                if a is None:
                    break
                s += a
                j += 1
            if j > i + 1 and abs(s - target) <= _tol(target):
                moved = children[i + 1:j]
                for x in moved:
                    x.parent = k
                k.children, k.tipe = moved, "ro"
                del children[i + 1:j]
                i -= 1
                continue
        # KRO kosong diikuti saudara bernominal sama (dokumen mencetak dua baris KRO berturut-turut,
        # mis. '7450.SCA PELATIHAN BIDANG ...' lalu '7450.SCA PELATIHAN DIGITAL TALENT ...').
        if k.tipe == "kro" and k.jumlah_tercantum and not k.items and not k.children:
            target, s, j = float(k.jumlah_tercantum), 0.0, i + 1
            while j < len(children) and s < target - _tol(target):
                a = _node_amount(children[j])
                if a is None:
                    break
                s += a
                j += 1
            if j > i + 1 and abs(s - target) <= _tol(target):
                moved = children[i + 1:j]
                for x in moved:
                    x.parent = k
                k.children = moved
                del children[i + 1:j]
                i -= 1
                continue
        if k.tipe == "kelompok" and k.jumlah_tercantum:
            own = 0.0 if not k.items and not k.children else _node_amount(
                Node(k.tipe, None, "", items=k.items, children=k.children))
        if own is not None and own < float(k.jumlah_tercantum) - _tol(float(k.jumlah_tercantum)):
            target, s, j = float(k.jumlah_tercantum), own, i + 1
            while j < len(children) and s < target - _tol(target):
                c = children[j]
                a = _node_amount(c)
                if a is None:
                    if c.tipe == "kelompok" and c.jumlah_tercantum is None and not c.items and not c.children:
                        a = 0.0             # baris judul/teks tanpa nominal ('mySPECTRA_portal (2 License) ...')
                    else:
                        break
                s += a
                j += 1
            if j > i + 1 and abs(s - target) <= _tol(target):
                moved = children[i + 1:j]
                for x in moved:
                    x.parent = k
                if own == 0 and k.parent is None and all(x.tipe == "komponen" for x in moved):
                    k.tipe = "ro"       # judul RO tanpa kode di atas komponen-komponennya
                k.children = k.children + moved
                del children[i + 1:j]
                _nest_groups(k.children)
        i -= 1


def tahun_dari_nama_file(name: str) -> Optional[int]:
    """Tahun anggaran dari nama file: 'TA 2026' / 'TA_2027' diutamakan, lalu tahun 4 digit yang berdiri
    sendiri ('RAB Layanan Umum 2027'). Tanggal menempel ('23052025', 'Rev100826') tidak dihitung."""
    stem = re.sub(r"\.(pdf|xlsx|xlsm|txt|md)$", "", str(name or "").split("#")[0], flags=re.I)
    m = re.search(r"(?<![A-Za-z])T\.?A\.?[\s_.-]*(20\d{2})(?!\d)", stem, re.I)
    if m:
        return int(m.group(1))
    ys = [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", stem)]
    return max(ys, key=ys.count) if ys else None      # seri -> yang muncul lebih dulu


class RabParser:
    def __init__(self, source: str = ""):
        self.source = source
        self.meta: dict[str, Any] = {}
        self.masked: list[str] = []
        self.roots: list[Node] = []
        self.current: Optional[Node] = None
        self.orphans: list[dict] = []
        self.warnings: list[str] = []
        self.state = "pre"             # pre | table | post
        self._hdr_seen = False         # kepala kolom tabel sudah lewat (masih di state 'pre')
        self.pending: list[str] = []   # teks tanpa angka (kandidat uraian/kelompok)
        self._jenis: Optional[str] = None      # nilai kolom Jenis Komponen yang belum ditempelkan
        self.pending_page: Optional[int] = None
        self.last_was_node = False
        self.waiting_meta: list[str] = []
        self.total_pdf: Optional[int | float] = None
        self.alokasi_header: Optional[int | float] = None
        self.ttd: dict[str, Any] = {}
        self.footer_next_jabatan = False
        self.meta_tersamar: list[str] = []
        self.last_meta_key: Optional[str] = None
        self._label_x0: Optional[float] = None
        self._cont = 0
        self._cur_text = ""
        self._prev_text = ""
        self._nip_wait = False
        self.ikk: list[str] = []
        self._ikk_blk: list[str] = []               # IKK tanpa nomor: nilai baris label + baris lanjutannya
        self._ikk_cont_x0: Optional[float] = None   # x kolom baris lanjutan IKK
        self._tahun_hdr: Optional[int] = None       # 'PERHITUNGAN TAHUN 2026' di kepala tabel
        self._ikk_open = False
        self._ttd_x0: Optional[float] = None

    # ---- helpers ---------------------------------------------------------------
    def warn(self, msg: str, line: Optional[Line] = None):
        loc = f" [hal {line.page}]" if line is not None else ""
        self.warnings.append(msg + loc)

    def push(self, node: Node):
        if node.x0 is None:
            node.x0 = getattr(self, "_line_x0", None)
        if node.tipe == "kelompok" and node.label and node.label.isupper() and node.x0 is not None:
            a = self.current
            while a is not None and a.tipe == "kelompok":
                a = a.parent
            if a is not None and a.tipe == "akun" and a.x0 is not None and node.x0 > a.x0 + 15:
                node.in_akun = True     # 'I Biaya Langsung Personel' / 'A Tenaga Ahli' menjorok di bawah akun
            elif a is not None and _akun_open(a):
                # tidak menjorok, tapi akun belum lengkap dan hurufnya bukan lanjutan subkomponen:
                # Romawi 'I Biaya Langsung Personel' / huruf berulang 'B Biaya Pelaporan' (sesudah subkomponen B)
                # = rincian jasa konsultan di dalam akun (BIF.002/003, akun 522131)
                prev_sub = getattr(self, "_last_sub", None)
                if node.label in ("I", "V", "X") and not (prev_sub and ord(prev_sub) == ord(node.label) - 1) \
                        or (prev_sub and len(prev_sub) == 1 and ord(node.label) <= ord(prev_sub)):
                    node.in_akun = True
        lvl = node_level(node)
        cur = self.current
        while cur is not None and node_level(cur) >= lvl:
            cur = cur.parent
        if node.tipe == "kelompok" and node.jumlah_tercantum is None and cur is not None and cur.tipe == "akun":
            cur = self._rincian_host(node, cur)
        node.parent = cur
        (cur.children if cur is not None else self.roots).append(node)
        self.current = node
        self.last_was_node = True
        if self._jenis and not node.jenis_komponen:
            node.jenis_komponen, self._jenis = self._jenis, None

    _ROMAN_RE = re.compile(r"^(I{1,3}|IV|VI{0,3}|IX|X)\.?\s+(\S.*)$")

    def _rincian_host(self, node: "Node", akun: "Node") -> "Node":
        """Rincian jasa (konsultan/lainnya) bertingkat tanpa nominal di dalam akun (CCL.001):
            '- Pemeliharaan datamart' / 'Pendampingan Perbaikan Dokumen ISO 27001'   (judul rincian)
              'I Biaya Langsung Personel'                                            (Romawi)
                'A Tenaga Ahli'  -> item 'Full Stack Developer ...'                  (huruf, lebih menjorok)
              'II Biaya Non Personel' -> item 'Biaya Laporan ...'
        Romawi masuk ke judul rincian terakhir akun itu (kelompok tanpa label/nominal/item); huruf kapital yang
        lebih menjorok masuk ke kelompok Romawi terakhir. Selain itu tetap anak akun."""
        def roman(n):
            if n.label in ("I", "V", "X"):
                return n.label
            mr = self._ROMAN_RE.match(n.uraian or "") if not n.label else None
            return mr.group(1) if mr else None

        last = akun.children[-1] if akun.children else None
        is_head = last is not None and last.tipe == "kelompok" and not last.label and roman(last) is None \
            and last.jumlah_tercantum is None and not last.items
        r = roman(node)
        if r is not None:
            if not is_head:
                return akun
            if not node.label:                       # 'II Biaya Non Personel' -> label 'II'
                node.label, node.uraian = r, self._ROMAN_RE.match(node.uraian).group(2)
            node.in_akun = True
            return last
        if node.label and len(node.label) == 1 and node.label.isupper() and node.x0 is not None:
            chain = last
            host = None
            while chain is not None and chain.tipe == "kelompok":
                if roman(chain) is not None and chain.jumlah_tercantum is None and chain.x0 is not None \
                        and chain.x0 < node.x0 - 3:
                    host = chain
                chain = chain.children[-1] if chain.children else None
            if host is not None:
                node.in_akun = True
                return host
        return akun

    def context(self) -> dict:
        ctx: dict[str, Any] = {"kro": None, "ro": None, "komponen": None, "subkomponen": None,
                               "akun_kode": None, "akun_uraian": None, "kelompok": None}
        n = self.current
        while n is not None:
            if n.tipe == "akun":
                ctx["akun_kode"], ctx["akun_uraian"] = n.kode, n.uraian
            elif n.tipe == "kelompok":
                nm = f"{n.label} {n.uraian}".strip() if n.label and n.label.isupper() else n.uraian
                ctx["kelompok"] = f"{nm} > {ctx['kelompok']}" if ctx["kelompok"] else nm
            elif n.tipe in ctx:
                ctx[n.tipe] = f"{n.kode} {n.uraian}".strip() if n.kode else n.uraian
            n = n.parent
        return ctx

    @staticmethod
    def _extract_label(uraian: str):
        m = LABEL_RE.match(uraian)
        return (m.group(1), re.sub(r"^[-\u2022\u2013]\s*", "", m.group(2))) if m else (None, uraian)

    @staticmethod
    def _extract_group_label(uraian: str):
        """Label kelompok: huruf kecil (a, b) ATAU huruf kapital (A, B) pada judul kelompok."""
        m = LABEL_RE.match(uraian) or UPPER_LABEL_RE.match(uraian)
        return (m.group(1), m.group(2)) if m else (None, uraian)

    def _join_pending(self) -> str:
        """Gabung teks tertunda. Huruf label sendirian di kolom kode yang tercetak rata-tengah di antara
        baris-baris judul ('Penyusunan ... Tahun 2008' / 'A' / 'tentang ...') dipindah ke depan sebagai label."""
        p = list(self.pending)
        if len(p) >= 2 and not re.match(r"^[A-Za-z][.)]?\s", p[0]):
            k = [i for i, t in enumerate(p) if re.fullmatch(r"[A-Z][.)]?", t.strip())]
            if len(k) == 1 and k[0] > 0:
                p.insert(0, p.pop(k[0]))
        return clean_text(" ".join(p))

    def flush_pending(self, line: Optional[Line] = None, as_continuation: bool = True):
        if not self.pending:
            return
        text = " ".join(self.pending)
        self.pending = []
        if self.state == "pre":
            self.meta.setdefault("baris_pra_tabel", []).append(text)
            return
        ct = getattr(self, "_center_title", None)
        if as_continuation and ct is not None and ct[0] is self.current and not self.current.items:
            self.current.uraian = clean_text(f"{self.current.uraian} {text}")     # sisa judul rata-tengah
            self._center_title = None
            return
        if as_continuation and self.last_was_node and self.current is not None:
            self.current.uraian = clean_text(f"{self.current.uraian} {text}")
            if self._jenis and not self.current.jenis_komponen:
                self.current.jenis_komponen = self._jenis
        else:
            self.warn(f"Teks tanpa angka tidak terklasifikasi: '{text}'", line)

    # ---- metadata & footer -----------------------------------------------------
    def _set_meta(self, key: str, val: str, line: Optional[Line] = None) -> bool:
        val = val.strip()
        if val.startswith(":"):
            val = val[1:].strip()
        if not val:
            return False
        if key == "satuan_ukur" and re.fullmatch(r"[-\u2013\u2014.\s]*", val):
            return False                      # placeholder kosong ("-") -> biarkan fallback dari Volume
        if key == "alokasi_dana":
            v = parse_amount(val)
            self.meta["alokasi_dana_teks"] = val
            if v is not None:
                self.meta["alokasi_dana"] = v
            elif MASK_ANY_RE.search(val):
                self.meta_tersamar.append(key)
            else:
                self.warn(f"Alokasi Dana tidak bisa diurai menjadi angka: '{val}'")
            return True
        if key == "indikator_kinerja_kegiatan":
            mi = IKK_NUM_RE.match(val)
            if mi:
                self.ikk.append(clean_text(mi.group(1)))
                self._ikk_open = True
                self.last_meta_key, self._cont = key, 0
                return True
        if key == "volume":
            # '( 6)' = format angka Excel berkurung (bukan negatif; 'Alokasi Dana : (Rp 1,008,780,000)' sama)
            pv = parse_volume(val)
            mpar = re.match(r"^\(\s*(\d[\d.,]*)\s*\)\s*(.*)$", pv["teks"]) if pv["angka"] is None else None
            if mpar:
                pv = {**parse_volume(f"{mpar.group(1)} {mpar.group(2)}"), "teks": pv["teks"]}
            self.meta["volume"] = pv["angka"] if pv["angka"] is not None else val
            self.meta["volume_teks"] = pv["teks"]
            if pv["satuan"]:
                self.meta["volume_satuan_inline"] = pv["satuan"]
        else:
            self.meta[key] = val
        if key == "indikator_kinerja_kegiatan":
            self._ikk_blk = [val]        # butir pertama; baris lanjutan ditambahkan oleh _try_meta
            self._ikk_cont_x0 = None
        if key == "kro" and isinstance(self.meta.get("kegiatan"), str):
            self.meta["kegiatan"] = re.sub(r"\s*Klasifikasi$", "", self.meta["kegiatan"])
        if MASK_ANY_RE.search(val) and key not in self.meta_tersamar:
            self.meta_tersamar.append(key)
        self.last_meta_key, self._cont = key, 0
        self._label_x0 = line.x0 if line is not None else None
        return True

    def _is_meta_cont(self, text: str, toks: list[str], line: Line) -> bool:
        """Baris lanjutan (wrap) dari nilai header sebelumnya."""
        # Indikator Kinerja Kegiatan bisa memuat banyak butir (satu baris atau lebih per butir): tidak dibatasi
        # 2 baris, asal tetap sekolom dengan baris lanjutan pertamanya (kolom nilai)
        ikk = self.last_meta_key == "indikator_kinerja_kegiatan" and bool(getattr(self, "_ikk_blk", None))
        if self.last_meta_key not in META_CONT_KEYS or self._cont >= (12 if ikk else 2) or not toks:
            return False
        if ikk and self._cont >= 2 and (line.x0 is None or self._ikk_cont_x0 is None
                                        or abs(line.x0 - self._ikk_cont_x0) > 25):
            return False
        if any(rx.match(text) for _, rx, req in META_RES if not req) or TTD_RE.match(text):
            return False
        first = toks[0]
        if any(r.match(first) for r in (KRO_RE, RO_RE, COMP_FULL_RE, SUBCODE_RE, AKUN_RE)):
            return False
        if self._label_x0 is not None and line.x0 is not None and line.x0 <= self._label_x0 + 5:
            return False                       # sejajar/di kiri label: bukan lanjutan nilai
        if re.search(r"\bRp\b|\d[\d.,]{5,}", text):
            return False
        return True

    def _try_meta(self, text: str, toks: list[str], line: Optional[Line] = None) -> bool:
        if (KRO_RE.match(toks[0]) or RO_RE.match(toks[0])):
            return False
        # label & nilai kop yang sudah dipisah berdasarkan posisi (loader PDF): label menentukan kunci,
        # nilai diambil utuh dari kolom nilai (bagian label yang meluber/tersamar tidak ikut jadi nilai)
        kv = line.kv if line is not None and self.state == "pre" else None
        key = _meta_label_key(kv[0]) if kv else None
        if key and kv[1]:
            self._set_meta(key, kv[1], line)
            if key in self.waiting_meta:
                self.waiting_meta.remove(key)
            return True
        if key:
            # label kosong menurut geometri (nilai di atas & bawahnya milik label lain): jangan menunggu nilai
            if key in self.waiting_meta:
                self.waiting_meta.remove(key)
            self.last_meta_key, self._cont = None, 0
            return True
        # placeholder masking yang menjadi baris wrap butir IKK ('... pada Tahun 2027 di' / '[MASKED DATA]'):
        # sekolom dengan baris lanjutan IKK sebelumnya -> bagian butir itu, bukan teks tersamar lepas
        if MASK_RE.fullmatch(text) and self.state == "pre" and line is not None and line.x0 is not None \
                and self.last_meta_key == "indikator_kinerja_kegiatan" and getattr(self, "_ikk_blk", None) \
                and getattr(self, "_ikk_cont_x0", None) is not None and abs(line.x0 - self._ikk_cont_x0) <= 3 \
                and self._is_meta_cont(text, toks, line):
            self.masked.append(text)
            self._append_ikk(text)
            return True
        if MASK_RE.fullmatch(text):
            self.masked.append(text)
            up = text.upper()
            # nilai tersamar untuk label yang menunggu (label sebaris sebelumnya tanpa nilai)
            if self.waiting_meta and self.state == "pre":
                key = self.waiting_meta.pop(0)
                self.meta[key] = text
                if key not in self.meta_tersamar:
                    self.meta_tersamar.append(key)
                return True
            if "KEMENTERIAN" in up:
                self.meta["kementerian"] = text
            elif "PEJABAT" in up:
                self.ttd["pejabat"] = text
            elif "JABATAN" in up and self.state != "pre":
                self.ttd["jabatan"] = text
                self.footer_next_jabatan = False
            elif "ESELON" in up or "SATKER" in up:
                if "unit_eselon2_satker" not in self.meta:
                    self.meta["unit_eselon2_satker"] = text
                else:
                    self.ttd["unit"] = text
            elif "SASARAN" in up:
                self.meta["sasaran_kegiatan"] = text
            elif "INDIKATOR" in up:
                self.meta["indikator_kinerja_kegiatan"] = text
            elif "NIP" in up:
                pass                             # ditangani _try_nip
            else:
                self.ttd.setdefault("masked_lain", []).append(text)
            for k in ("kementerian", "unit_eselon2_satker", "sasaran_kegiatan", "indikator_kinerja_kegiatan"):
                if k in self.meta and self.meta[k] == text and k not in self.meta_tersamar:
                    self.meta_tersamar.append(k)
                if k in self.waiting_meta and k in self.meta:
                    self.waiting_meta.remove(k)
            return True
        for key, rx, req in META_RES:
            m = rx.match(text)
            if not m:
                continue
            has_colon, val = m.group(1), m.group(2).strip()
            if req and not has_colon:
                # 'Volume RO 1' / 'Volume 12 Layanan' di atas tabel: label volume tanpa titik dua diikuti angka
                # 'Kegiatan Pengelolaan Komunikasi Publik ...' (label tanpa titik dua) di kop, sebelum tabel
                keg_ok = key == "kegiatan" and self.state == "pre" and "kegiatan" not in self.meta \
                    and re.match(r"^[A-Z][a-z]", val) is not None and not self._hdr_seen
                if not (key == "volume" and self.state == "pre" and re.match(r"^\d", val)) and not keg_ok:
                    continue
            if key in META_PRE_ONLY and self.state != "pre":
                if key in self.meta:             # header berulang di halaman lanjutan: telan saja
                    self.last_meta_key, self._cont = key, 0
                    self._label_x0 = line.x0 if line is not None else None
                    return True
                continue
            if self.state == "table" and key in ("kegiatan", "volume") and not has_colon:
                continue
            if val.startswith(":"):
                val = val[1:].strip()
            if val:
                self._set_meta(key, val, line)
                if key in self.waiting_meta:
                    self.waiting_meta.remove(key)        # label ini sudah bernilai: jangan menunggu nilai lain
            else:
                if key not in META_NO_WAIT:
                    self.waiting_meta.append(key)
                self.last_meta_key, self._cont = key, 0
                self._label_x0 = line.x0 if line is not None else None
            return True
        # nilai label yang terpisah baris (label di satu baris, isi di baris berikutnya)
        if self.waiting_meta and self.state == "pre" and is_header_row(text) and len(toks) >= 2:
            self.waiting_meta.clear()          # judul tabel sudah tercapai: label kosong (mis. 'Indikator KRO (Opsional)') memang kosong
        if self.waiting_meta and self.state == "pre" and not is_header_row(text):
            key = self.waiting_meta.pop(0)
            self._set_meta(key, text, line)
            return True
        # lanjutan nilai yang terbungkus (wrap)
        if line is not None and self.state in ("pre", "table") and self._is_meta_cont(text, toks, line):
            if self.state == "pre" and self.last_meta_key == "indikator_kinerja_kegiatan" \
                    and getattr(self, "_ikk_blk", None):
                if self._ikk_cont_x0 is None:
                    self._ikk_cont_x0 = line.x0
                self._append_ikk(text)
                return True
            self._cont += 1
            if self.state == "pre" and self.last_meta_key in self.meta \
                    and isinstance(self.meta[self.last_meta_key], str):
                self.meta[self.last_meta_key] = clean_text(self.meta[self.last_meta_key] + " " + text)
            return True
        return False

    def _append_ikk(self, text: str) -> None:
        """Baris lanjutan blok Indikator Kinerja Kegiatan: simpan per baris (dipecah per butir di finish())
        dan tetap sambungkan ke teks gabungan (hasil bila ternyata hanya satu butir)."""
        k = "indikator_kinerja_kegiatan"
        self._cont += 1
        self._ikk_blk.append(text)
        if isinstance(self.meta.get(k), str):
            self.meta[k] = clean_text(self.meta[k] + " " + text)
        if MASK_ANY_RE.search(text) and k not in self.meta_tersamar:
            self.meta_tersamar.append(k)

    # ---- NIP -------------------------------------------------------------------
    def _try_nip(self, text: str) -> bool:
        """NIP sering tersamar; yang dicari: ada tidaknya LABEL 'NIP' di blok penanda tangan."""
        if self._nip_wait and self.state in ("table", "post"):
            self._nip_wait = False
            st, nd = classify_nip_value(text)
            if st in ("tersamar", "terbaca_valid", "terbaca_tidak_standar") and len(text) < 60 \
                    and not NIP_LABEL_RE.search(text):
                nip = self.ttd.get("nip", {})
                nip.update(nilai_status=st, jumlah_digit=nd, teks_baris=redact_digits(text),
                           nilai=nip_value(text, st == "tersamar"))
                return True
        if self.state == "pre":
            return False
        placeholder = bool(MASK_RE.fullmatch(text)) and bool(NIP_LABEL_RE.search(text))
        m = NIP_LABEL_RE.search(text)
        if not m:
            return False
        if not placeholder:
            if m.start() > 3 and self.state != "post":
                return False
            rest = text[m.end():]
            st, nd = classify_nip_value(rest)
            if self.state == "table" and not (st != "kosong" or not rest.strip(" :.")):
                return False                     # di tengah tabel: hanya bila bentuknya benar-benar baris NIP
            if self.state == "table" and re.search(r"[A-Za-z]{3,}", MASK_ANY_RE.sub("", rest)):
                return False
        else:
            st, nd, rest = "tersamar", 0, ""
        prev = self._prev_text
        nama = None
        if prev and not TTD_RE.match(prev) and prev != self.ttd.get("jabatan") and len(prev) < 90 \
                and not NIP_LABEL_RE.search(prev):
            nama = prev
        self.ttd["nip"] = {
            "ada_label": True,
            "sumber": "placeholder" if placeholder else "teks",
            "nilai_status": st, "jumlah_digit": nd,
            "teks_baris": redact_digits(text),
            "nilai": clean_text(text) if placeholder else nip_value(rest),
        }
        if nama and "nama" not in self.ttd:
            self.ttd["nama"] = nama
        self.state = "post" if self.state == "table" and self.total_pdf is not None else self.state
        self._nip_wait = (st == "kosong" and not placeholder)
        return True

    def _try_footer(self, text: str, line: Optional[Line] = None) -> bool:
        m = TTD_RE.match(text)
        if m and self.state in ("table", "post") or (m and not self.ttd):
            self.ttd.update({"tempat": m.group(1).strip(), "tanggal": m.group(2).strip()})
            ym = re.search(r"(\d{4})$", m.group(2))
            if ym:
                self.ttd["tahun"] = int(ym.group(1))
            self.footer_next_jabatan = True
            self._ttd_x0 = line.x0 if line is not None else None
            self.state = "post"          # RAB tanpa baris 'JUMLAH TOTAL': tanggal/pejabat menutup tabel
            self.last_was_node = False
            return True
        if self.state == "post":
            x0 = line.x0 if line is not None else None
            if x0 is not None and self._ttd_x0 is not None and x0 < self._ttd_x0 - 80:
                self.ttd.setdefault("paraf_tabel", []).append(text)     # tabel 'Jabatan | Paraf' di kiri
                return True
            if self.footer_next_jabatan:
                self.ttd["jabatan"] = text
                self.footer_next_jabatan = False
            elif "nama" not in self.ttd and not NIP_LABEL_RE.search(text):
                self.ttd["nama"] = text
            else:
                self.ttd.setdefault("catatan_lain", []).append(text)
            return True
        return False

    # ---- entry point per baris ---------------------------------------------------
    def feed(self, line: Line):
        """Bungkus _feed: menempelkan nilai kolom Jenis Komponen (Line.jenis, hasil pembacaan posisi kolom
        PDF) ke node yang dibuat baris ini, ke item yang ditambahkan baris ini, ke node terakhir (sel Jenis
        milik baris lanjutan/wrap), atau -- bila baris ini teks tertunda -- ke node yang dibuat berikutnya."""
        self._line_x0 = line.x0
        # kolom KETERANGAN menempel ke jumlah ('9.720.000SBM 2026'): potong, simpan sebagai keterangan item
        mk = _GLUED_KET_RE.match(line.text)
        ket = line.keterangan
        if mk or ket:
            if mk:
                ket = clean_text(f"{mk.group(2)} {ket or ''}")
            # tautan (Data Dukung yang tercetak di dalam/menimpa sel keterangan) tidak disimpan
            ket = clean_text(_URL_TOK_RE.sub(" ", ket or "")) or None
            line = Line(**{**line.__dict__, "text": mk.group(1) if mk else line.text, "keterangan": None})
            n0, cur0 = getattr(self, "_n_items", 0), self.current
            self.feed(line)
            if not ket:
                return
            if getattr(self, "_n_items", 0) > n0 and self._last_item is not None:
                self._last_item["keterangan"] = ket
            elif self.current is not None and self.current is not cur0 and self.state == "table":
                self.current.keterangan = ket                # baris judul akun/kelompok
            elif self._last_item is not None and self.state == "table":
                # sambungan teks keterangan (sel terbungkus) -> item terakhir
                self._last_item["keterangan"] = clean_text(f"{self._last_item.get('keterangan') or ''} {ket}")
            return
        jen = line.jenis
        if jen:
            t = clean_text(line.text)
            if not t or is_header_row(t) or any(rx.search(t) for rx in NOISE_RES):
                jen = None
            elif line.text_with_jenis and _NO_URAIAN_RE.fullmatch(t) and _amount_only_tail(t):
                # baris tanpa uraian ('1 Layanan 2.017.894.000': judul sel tertimpa/hilang) -> jangan sampai
                # menjadi item palsu; urai seperti semula dengan kata Jenis tetap di teks.
                line = Line(**{**line.__dict__, "text": line.text_with_jenis, "jenis": None})
                jen = None
        if not jen:
            if self._jenis and not self.pending:
                self._jenis = None            # tidak ada teks tertunda yang menunggu node: kedaluwarsa
            self._feed(line)
            return
        self._jenis = jen
        cur = self.current
        n_items = len(cur.items) if cur is not None else 0
        self._feed(line)
        if self._jenis is None:               # sudah dipakai node baru (push) / lanjutan uraian
            return
        if self.current is cur and cur is not None and len(cur.items) > n_items:
            for it in cur.items[n_items:]:
                it.setdefault("jenis_komponen", jen)
            self._jenis = None
        elif self.pending:
            pass                              # teks tertunda: tempel ke node yang dibuat berikutnya
        else:
            if self.current is not None and not self.current.jenis_komponen and self.state == "table":
                self.current.jenis_komponen = jen
            self._jenis = None

    _WRAP_RUMUS_RE = re.compile(
        r"(\d+(?:[.,]\d+)?\s*[A-Za-z/]+\.?(?:\s*[xX×]\s*\d+(?:[.,]\d+)?\s*[A-Za-z/]+\.?)+)(?:\s+(\d+(?:[.,]\d+)?))?")

    def _try_wrapped_rincian(self, text: str, line: Line) -> bool:
        """Sel rincian yang tercetak turun satu baris dari item-nya ('13 ORG X 1 HR X 6 KGT 78' di bawah
        '- Paket Meeting Fullday Eselon 2 OK 692.000 53.976.000'): baris ini HANYA berisi rumus (+ volume),
        letaknya di kolom rincian (bukan kolom kode/uraian). Ditempelkan ke item sebelumnya; volume hanya
        diisi bila item belum punya volume dan volume x harga = jumlah."""
        if self.state != "table" or self.current is None or not self.current.items or self.pending:
            return False
        if line.x0 is None or line.pw is None or line.x0 / line.pw < 0.25:
            return False
        mw = self._WRAP_RUMUS_RE.fullmatch(text)
        if not mw:
            return False
        it = self.current.items[-1]
        rumus, vtxt = mw.group(1), mw.group(2)
        if not it.get("rumus_uraian"):
            it["rumus_uraian"] = rumus
        old = it.get("volume")
        h, j = it.get("harga_satuan"), it.get("jumlah")
        tol = max(1, abs(j) * 1e-6) if j is not None else 1
        old_bad = old is None or (h is not None and j is not None and abs(old * h - j) > tol)
        if vtxt and old_bad:
            v = parse_num(vtxt)
            if v is not None and h is not None and j is not None and abs(v * h - j) <= tol:
                if old is not None:
                    # volume lama 'dicuri' dari ujung uraian ('Paket Meeting Fullday Eselon 2') -> kembalikan
                    ot = str(int(old)) if float(old).is_integer() else str(old)
                    if not str(it.get("uraian") or "").endswith(ot):
                        it["uraian"] = clean_text(f"{it.get('uraian') or ''} {ot}")
                it["volume"] = v
                it["volume_sumber"] = "pdf"
                sat = it.get("satuan")
                it["rincian_perhitungan"] = f"{vtxt} {sat}".strip() if sat else vtxt
                it["catatan_parser"] = [c for c in it.get("catatan_parser", []) if "volume tidak terbaca" not in c] \
                    + ["rincian & volume tercetak satu baris di bawah item"]
        return True

    def _resolve_tahun(self):
        """Tahun anggaran, urut keandalan: judul ('TA/TAHUN ANGGARAN 2026') > kepala tabel
        ('PERHITUNGAN TAHUN 2026') > nama file > 'Tahun 2026' di Indikator Kinerja > tanggal TTD + 1."""
        if "tahun_anggaran" in self.meta:
            self.meta["tahun_anggaran_sumber"] = "judul"
            return
        th, src = self._tahun_hdr, "kepala_tabel"
        if th is None:
            th, src = tahun_dari_nama_file(self.source), "nama_file"
        if th is None:
            ys = [int(y) for t in self.ikk for y in re.findall(r"\btahun\s+(20\d{2})\b", t, re.I)]
            if ys:
                th, src = max(sorted(set(ys), reverse=True), key=ys.count), "indikator_kinerja"
        if th is None and self.ttd.get("tahun"):
            # RAB disusun/ditandatangani pada tahun SEBELUM tahun anggarannya (TTD 2025 -> TA 2026)
            th, src = self.ttd["tahun"] + 1, "tanggal_ttd"
            self.warn(f"Tahun anggaran tidak tercantum di dokumen maupun nama file; diperkirakan {th} "
                      f"(tahun tanda tangan {self.ttd['tahun']} + 1) - periksa manual.")
        if th is None:
            self.warn("Tahun anggaran tidak ditemukan.")
            return
        self.meta["tahun_anggaran"] = th
        self.meta["tahun_anggaran_sumber"] = src

    def _feed(self, line: Line):
        text = clean_text(line.text)
        if not text:
            return
        toks = normalize_tokens(text.split())
        toks = _strip_code_prefix(toks)
        text = " ".join(toks)
        self._prev_text, self._cur_text = self._cur_text, text

        mt = re.fullmatch(r"TA\.?\s*(\d{4})", text, re.I)
        if mt:
            self.meta["tahun_anggaran"] = int(mt.group(1))
            return
        if self.state == "pre":
            # 'KELUARAN (OUTPUT) KEGIATAN T.A. 2026' | 'RINCIAN ANGGARAN BIAYA TA 2026' | 'RIN ARAN 2026'
            # + 'RINCIAN ANGGARAN BELANJA TAHUN 2026'
            my = re.search(r"\b(?:T\.?\s?A\.?|TAHUN\s+ANGGARAN|ANGGARAN\s+(?:BELANJA|BIAYA)\s+TAHUN|ARAN)"
                           r"\s*(20\d{2})\b", text, re.I)
            if my and "tahun_anggaran" not in self.meta:
                self.meta["tahun_anggaran"] = int(my.group(1))
            if re.fullmatch(r"(?:T\.?\s?A\.?|TAHUN\s+ANGGARAN)\s*20\d{2}", text, re.I):
                return                               # sudah masuk tahun_anggaran
        mh = re.search(r"PERHITUNGAN\s+TAHUN\s+(20\d{2})\b", text, re.I)
        if mh and self._tahun_hdr is None:
            self._tahun_hdr = int(mh.group(1))
        if is_header_row(text) or any(rx.search(text) for rx in NOISE_RES):
            if self.state == "pre" and is_header_row(text):
                self._hdr_seen = True
                # label kop yang masih kosong saat kepala tabel tercapai ('Kegiatan :' / 'Sasaran Kegiatan :')
                # memang kosong: jangan diisi teks kepala tabel ('PROGRAM,' / 'KEG., PROGRAM/KEGIATAN/...')
                # dan jangan menelan baris tabel sebagai lanjutan nilainya (CCL.001)
                self.waiting_meta.clear()
                if self.last_meta_key in META_CONT_KEYS:
                    self.last_meta_key = None
            m = re.search(r"Rp\s?([\d.,]+)", text)
            if m and self.alokasi_header is None and re.match(r"^RINCIAN\s+ANGGARAN", text, re.I):
                self.alokasi_header = parse_amount(text[m.start():])
            return

        # judul unit di kop ('BAGIAN KEUANGAN [MASKED DATA]'): KAPITAL, sebelum label metadata mana pun
        if self.state == "pre" and "judul_unit" not in self.meta and not self.ikk \
                and not any(k in self.meta for k in ("kementerian", "unit_eselon2_satker", "kegiatan")):
            bare = MASK_ANY_RE.sub(" ", text).strip()
            if len(bare) >= 3 and bare == bare.upper() and re.search(r"[A-Z]{3}", bare) \
                    and not re.search(r"\d", bare) and not any(rx.match(text) for _, rx, _r in META_RES):
                self.meta["judul_unit"] = text
                if MASK_ANY_RE.search(text):
                    self.meta_tersamar.append("judul_unit")
                return

        # 'SUB TOTAL I + II 200.000.000' / 'SUBTOTAL 1.000.000' = pengulangan subtotal rincian -> lewati
        if self.state == "table" and re.match(r"^SUB\s*-?\s*TOTAL\b", text, re.I) and _THOUSANDS_RE.search(text):
            self.flush_pending(line, as_continuation=False)
            self.warn(f"Baris '{text}' dianggap pengulangan subtotal (tidak dimasukkan)", line)
            return
        # 'TOTAL 504.498.000' di tengah tabel = pengulangan subtotal (bukan total dokumen) bila tidak sama
        # dengan alokasi dana -> lewati baris itu saja, tabel berlanjut.
        if self.state == "table" and _BARE_TOTAL_RE.match(text):
            _, tv, _ = split_amount_tail([t for t in toks if t.lower() not in ("rp", "rp.")])
            alok = self.meta.get("alokasi_dana") or self.alokasi_header
            if alok and tv is not None and abs(float(tv) - float(alok)) > _tol(float(alok)):
                self.flush_pending(line, as_continuation=False)
                self.warn(f"Baris 'TOTAL {fmt_idr(tv)}' di tengah tabel (≠ alokasi) dianggap pengulangan subtotal", line)
                return
        # JUMLAH TOTAL (kadang huruf terpisah: 'J U M L A H  T O T A L')
        if re.sub(r"\s+", "", text).upper().startswith("JUMLAHTOTAL") or (
                self.state == "table" and _BARE_TOTAL_RE.match(text)):     # 'TOTAL 2.449.072.000'
            self.flush_pending(line, as_continuation=False)
            _, total, _ = split_amount_tail([t for t in toks if t.lower() not in ("rp", "rp.")])
            self.total_pdf = total
            self.state = "post"
            self.last_was_node = False
            return

        # Rp.. di luar tabel -> alokasi header
        if self.state != "table":
            m = re.search(r"Rp\s?([\d.,]+)", text)
            # baris pertama tabel berkode ('051 Sistem informasi ... Rp 478.058.000') bukan alokasi kop
            struct_row = re.match(r"^(?:\d{4}\.[A-Z]{3}(?:\.\d{3})?|0\d{2}|5\d{5})\s+[A-Za-z]", text)
            if m and not struct_row and not re.match(r"^(Alokasi\s+Dana)", text, re.I):
                v = parse_amount(text[m.start():])
                if v is not None and self.alokasi_header is None:
                    self.alokasi_header = v
                    return

        # 'Lokasi : Jawa Barat' -> atribut node (judul komponen/subkomponen/kelompok di atasnya), BUKAN metadata.
        # Sebelum tabel (mis. di kop) -> dipasang ke node akar pertama saat finish().
        mlok = _LOKASI_RE.match(text)
        if mlok and self.state != "post":
            val = clean_text(mlok.group(1))
            if self.state == "table":
                self.flush_pending(line, as_continuation=True)
                n = self.current
                while n is not None and (n.tipe == "akun" or n.items):   # judul terakhir yang belum berisi
                    n = n.parent
                if n is not None:
                    n.lokasi = val if not n.lokasi else f"{n.lokasi}; {val}"
                    self.last_was_node = False
                    return
            self._lokasi_pre = val if not getattr(self, "_lokasi_pre", None) else f"{self._lokasi_pre}; {val}"
            return

        if self.state == "pre" and self._try_ikk(text, line):
            return
        if self._try_nip(text):
            return
        if self._try_meta(text, toks, line):
            return
        if self.state == "post" or TTD_RE.match(text):
            if self._try_footer(text, line):
                return

        if self._try_wrapped_rincian(text, line):
            return

        # baris Program ('059.WA Program Dukungan Manajemen 1.125.000.000'), juga yang diulang di kepala
        # halaman lanjutan -> metadata, bukan node.
        mp = _PROGRAM_ROW_RE.match(text)
        if mp:
            self.meta.setdefault("kode_program", mp.group(1))
            self.meta.setdefault("program", clean_text(mp.group(2)))
            return

        # ---- baris struktur ------------------------------------------------------
        first, rest = toks[0], toks[1:]
        ik = line.in_kode_col()
        # batas kolom Kode dari kepala tabel bisa terlalu sempit: kode komponen yang nyata tercetak
        # melewatinya ('051' x0=114 > batas 110, Rentan) -> huruf subkomponen sedikit di kanannya ('A' x0=120)
        # masih di kolom Kode.
        cx = getattr(self, "_komp_code_x", None)
        ik_sub = ik or (cx is not None and line.x0 is not None and line.x0 <= cx + 8)
        ntype = None
        first = self._fix_ro_code(first, rest)
        if RO_RE.match(first):
            ntype = "ro"
        elif KRO_RE.match(first):
            ntype = "kro"
        elif COMP_FULL_RE.match(first):
            ntype = "komponen"          # '4485.EBA.994.001 Belanja Pegawai 2.731.525.000' (kode RO + komponen)
            first = first.rsplit(".", 1)[-1]
        elif KOMP_RE.match(first) and ik and rest and all(NUMISH.fullmatch(t) for t in rest) \
                and _amount_only(rest):
            ntype = "komponen"          # '051 660.000.000' / '002 1 1,346,597,000' (tanpa uraian)
        elif SUBCODE_RE.match(first) and rest and rest[0][:1].isalpha():
            ntype = "subkomponen"       # '001.0A Gaji dan Tunjangan PNS 1.685.281.000'
        elif AKUN_RE.match(first) and ik and rest and _amount_only(rest) and self.state == "table":
            ntype = "akun"              # '521115 64.440.000' (judul akun terpotong/tersamar, hanya kode + nominal)
        elif SUB_RE.match(first) and ik_sub and rest and _amount_only(rest) and self.state == "table" \
                and self._sub_ok(first, line):
            ntype = "subkomponen"       # 'C 707.760.000' (judul sub berada di sel lain, tidak sebaris)
        elif SUB_RE.match(first) and ik_sub and len(rest) >= 2 and MASK_RE.match(" ".join(rest[:-1])) \
                and _amount_only(rest[-1:]) and self.state == "table" and self._sub_ok(first, line):
            ntype = "subkomponen"       # 'B [MASKED DATA PEJABAT] 522.125.000' (judul sub tersamar)
        elif rest and rest[0][:1].isalpha():
            if AKUN_RE.match(first):
                ntype = "akun"
            elif re.fullmatch(r"\d{5}", first) and ik and rest[0].lower().startswith("belanja") \
                    and self.state == "table":
                ntype = "akun"
                self.warn(f"Kode akun 5 digit '{first}' (kemungkinan salah ketik di dokumen); diperlakukan sebagai akun", line)
            elif KOMP_RE.match(first) and ik:
                ntype = "komponen"
            elif SUB_RE.match(first) and ik_sub and self._sub_ok(first, line):
                ntype = "subkomponen"

        # judul komponen membungkus kodenya ('Persiapan Fasilitasi ... Kelompok' / '051 206.000.000' / 'Rentan'):
        # kode 0xx + satu nominal ber-ribuan, judulnya = teks tertunda di atasnya.
        #   juga dengan volume sub output: '051 3 KEGIATAN 362.620.000' / '052 3 KEGIATAN #REF! 843.179.000'
        rest_k = [t for t in rest if t.upper() not in ("#REF!", "#N/A", "#VALUE!", "#DIV/0!")]
        if ntype is None and (self.state == "table" or self._hdr_seen) and self.pending \
                and re.fullmatch(r"0\d{2}", first) \
                and rest_k and _AMT_FULL_RE.fullmatch(rest_k[-1]) \
                and re.fullmatch(r"(\d{1,4} [A-Za-z/]+(?: [A-Za-z/]+)?)?", " ".join(rest_k[:-1])):
            ct = getattr(self, "_center_title", None)
            if ct is not None and len(self.pending) > 1:
                # judul rata-tengah sebelumnya: baris di bawah nominalnya sebanyak baris di atasnya
                k = min(ct[1], len(self.pending) - 1)
                ct[0].uraian = clean_text(" ".join([ct[0].uraian] + self.pending[:k]))
                self.pending = self.pending[k:]
            self.state = "table"
            title = clean_text(" ".join(self.pending))
            self.pending = []
            if len(rest_k) < len(rest):
                self.warn(f"Sel galat Excel ({', '.join(t for t in rest if t not in rest_k)}) pada baris komponen {first} diabaikan", line)
            self._add_node("komponen", first, rest_k[-1:], line)
            self.current.uraian = title
            if len(rest_k) >= 3:                  # '3 KEGIATAN' = volume + satuan sub output
                self.current.volume_suboutput = parse_num(rest_k[0])
                self.current.satuan_suboutput = clean_text(" ".join(rest_k[1:-1]))
            self.last_was_node = True
            return

        if ntype:
            self.flush_pending(line, as_continuation=True)
            self.state = "table"
            self._add_node(ntype, first, rest, line)
            return

        # baris pertama tabel berkode tak baku ('FAE Pengawasan ... Rp3,168,327,000', '10 Pengukuran ...
        # Rp1,487,940,000') langsung sesudah kepala kolom: buka tabel; letaknya diatur _nest_groups.
        if self.state == "pre" and self._hdr_seen and re.match(r"^(\d{2,3}|[A-Z]{3})\s+[A-Za-z]", text) \
                and _AMT_FULL_RE.fullmatch(re.sub(r"^Rp\.?", "", toks[-1], flags=re.I)):
            self.state = "table"
            self._add_content(toks, line)
            cur = self.current
            if cur is not None and cur.tipe == "kelompok" and cur.uraian.startswith(toks[0] + " "):
                cur.kode, cur.uraian = toks[0], cur.uraian[len(toks[0]) + 1:]
            return

        # sesudah kepala kolom (masih 'pre'): judul sel rata-tengah ('Pengelolaan Komunikasi ... Koperasi,' /
        # '- 15.300.011.000' / 'UMKM') -> teks ditahan sebagai pending, baris nominal-saja membuka tabel.
        if self.state == "pre" and self._hdr_seen:
            body = [t for t in toks if t not in ("-", "\u2013")]
            hdr_words = sum(1 for w in re.split(r"[\s/,()]+", text.lower()) if w.rstrip(".") in _HDR_WORDS)
            leftover = [w for w in re.split(r"[\s/,]+", text.lower())
                        if w and w.rstrip(".") not in _HDR_WORDS and not re.fullmatch(r"\(?\d{1,2}\)?|[()]", w)]
            if hdr_words >= 2 or not leftover:
                return                    # sisa kepala kolom ('KEG., PROGRAM / KEGIATAN / OUTPUT / AKUN TOTAL',
                                          # 'OUTPUT/ 1 2 ( 3) ( 4) ( 5)')
            if body and not any(_THOUSANDS_RE.search(t) for t in body) and ":" not in text \
                    and not any(rx.match(text) for _, rx, _r in META_RES):
                self.pending.append(clean_text(text))
                self.pending_page = line.page
                return
            if body and self.pending and all(_AMT_FULL_RE.fullmatch(re.sub(r"^Rp\.?", "", t, flags=re.I))
                                             or t.lower() in ("rp", "rp.") for t in body):
                self.state = "table"
                self._add_content(toks, line)
                return

        if self.state == "pre":
            # sebelum tabel dimulai: hanya simpan baris yang benar-benar membawa informasi baru.
            if self._pre_redundant(text):
                return
            self.meta.setdefault("baris_pra_tabel", []).append(text)
            return

        self._add_content(toks, line)

    def _sub_ok(self, label: str, line: Line) -> bool:
        """Huruf di depan baris benar label subkomponen? Bukan bila:
          - baris menjorok jauh ke kanan dari akun yang sedang terbuka ('I Biaya Langsung Personel' /
            'A Tenaga Ahli' di dalam akun 522131) -> itu kelompok di dalam akun;
          - angka Romawi I/V/X yang tidak melanjutkan urutan huruf subkomponen sebelumnya (B -> 'I')."""
        n = self.current
        while n is not None and n.tipe not in ("akun", "subkomponen", "komponen", "ro", "kro"):
            n = n.parent
        prev_sub = getattr(self, "_last_sub", None)
        if n is not None and n.tipe == "akun" and n.x0 is not None and line.x0 is not None \
                and line.x0 > n.x0 + 15 and not self._sub_same_col(label, line, n, prev_sub):
            return False
        # huruf yang berulang/mundur tanpa komponen baru ('B Biaya Pelaporan' sesudah subkomponen B) bukan
        # subkomponen baru, melainkan bagian rincian (mis. rincian jasa konsultan)
        if prev_sub and len(prev_sub) == 1 and ord(label) <= ord(prev_sub):
            # ... hanya bila akun yang sedang terbuka BELUM terpenuhi nominalnya (rinciannya masih berlanjut).
            # Huruf dokumen yang salah urut ('E BPT' sesudah 'I MANADO') tetap subkomponen bila akunnya lengkap.
            a = self.current
            while a is not None and a.tipe != "akun":
                a = a.parent
            if a is not None and a.jumlah_tercantum is not None:
                got = _node_amount(Node("akun", None, "", items=a.items, children=a.children)) or 0.0
                if got < float(a.jumlah_tercantum) - _tol(float(a.jumlah_tercantum)) or _akun_open(a):
                    return False                # 'B Biaya Pelaporan' di dalam rincian akun, sesudah subkomponen B/C
        if label in ("I", "V", "X"):
            prev = getattr(self, "_last_sub", None)
            if prev is None or ord(prev) != ord(label) - 1:
                return False
        return True

    def _fix_ro_code(self, first: str, rest: list[str]) -> str:
        """Kode RO di sel tabel yang terpotong/terbungkus, dipulihkan dari 'Kode : 7450.EBC.996' di kop:
          - terpotong      : '7450.EBC.99 LAYANAN PENDIDIKAN ...'           -> '7450.EBC.996' (RO)
          - terbungkus     : '7450.SCA PELATIHAN DIGITAL ...' (KRO yang sama sudah terbuka, belum ada RO)
                             -> '7450.SCA.002' (RO); sisa '.002' tercetak di baris berikutnya dan bertumpuk
                             dengan kode komponen: '.005211 PERENCANAAN ...' = '.002' + '051' -> '051'."""
        kh = self.meta.get("kode_header") or ""
        wrap = getattr(self, "_ro_wrap", None)
        self._ro_wrap = None
        if not RO_RE.match(kh) or not rest or not rest[0][:1].isalpha():
            return first
        if wrap and not any(rx.match(first) for rx in (KRO_RE, RO_RE, KOMP_RE, AKUN_RE, SUB_RE)):
            from itertools import combinations
            cands: dict[str, str] = {}          # kode komponen -> sisa kode RO yang bertumpuk
            for pos in combinations(range(len(first)), len(first) - 3):
                suf = "".join(first[i] for i in pos)
                rem = "".join(ch for i, ch in enumerate(first) if i not in pos)
                if re.fullmatch(r"\.\d{3}", suf) and re.fullmatch(r"0\d{2}", rem):
                    cands.setdefault(rem, suf)
            # banyak kemungkinan ('.005211' = '.021'+'051' / '.051'+'021' / ...): pilih komponen di kop
            # ('Komponen : 051-Persiapan dan Evaluasi ... 052-Pelaksanaan ...') yang belum terpakai dan
            # judulnya paling mirip dengan uraian baris ini; tanpa kop -> sisa RO yang sama dengan kode kop
            used = {n.kode for n in self._walk() if n.tipe == "komponen"}
            kop = str(self.meta.get("komponen") or "")
            words = set(re.findall(r"[a-z]{3,}", " ".join(rest).lower()))
            best, best_sc = [], 0
            for km in re.finditer(r"\b(0\d{2})\s*[-.:]?\s*(.*?)(?=\s*\b0\d{2}\s*[-.:]|$)", kop):
                c = km.group(1)
                if c not in cands or c in used:
                    continue
                sc = len(words & set(re.findall(r"[a-z]{3,}", km.group(2).lower())))
                if sc > best_sc:
                    best, best_sc = [c], sc
                elif sc == best_sc and sc:
                    best.append(c)
            pick = best or [c for c in cands if cands[c] == wrap]
            if len(pick) != 1:
                return first
            ro, a = None, self.current
            while a is not None and ro is None:
                ro = a if a.tipe == "ro" else None
                a = a.parent
            if ro is not None and ro.kode and cands[pick[0]] != wrap:
                ro.kode = ro.kode[: -len(wrap)] + cands[pick[0]]     # kode RO yang TERCETAK di tabel
            return pick[0]
        if first != kh and first.count(".") == 2 and kh.startswith(first) and len(first) > len(kh) - 3:
            return kh
        if KRO_RE.match(first) and kh.startswith(first + "."):
            has_ro, a = False, self.current
            kro_open = False
            while a is not None:
                has_ro |= a.tipe == "ro"
                kro_open |= a.tipe == "kro" and a.kode == first
                a = a.parent
            if kro_open and not has_ro and not any(n.tipe == "ro" for n in self._walk()):
                self._ro_wrap = kh[len(first):]
                return kh
        return first

    def _sub_same_col(self, label: str, line: Line, akun: Node, prev_sub: Optional[str]) -> bool:
        """Huruf yang menjorok dari akun tetap subkomponen bila tata letak dokumen memang menaruh huruf
        subkomponen lebih menjorok daripada kode akun (EBC.996: 'A ...' x0=76, '524119' x0=57), yaitu:
        sekolom dengan subkomponen sebelumnya, hurufnya melanjutkan urutan, dan akun terbuka sudah lengkap."""
        if not prev_sub or len(prev_sub) != 1 or ord(label) <= ord(prev_sub):
            return False
        s = akun.parent
        while s is not None and s.tipe != "subkomponen":
            s = s.parent
        if s is None or s.label != prev_sub or s.x0 is None or abs(line.x0 - s.x0) > 3:
            return False
        if akun.jumlah_tercantum is None:
            return False
        got = _node_amount(Node("akun", None, "", items=akun.items, children=akun.children)) or 0.0
        return abs(got - float(akun.jumlah_tercantum)) <= _tol(float(akun.jumlah_tercantum))

    def _try_ikk(self, text: str, line: Line) -> bool:
        """Indikator kinerja bernomor (termasuk yang terpusat vertikal di atas labelnya) dan lanjutan wrap-nya."""
        mi = IKK_NUM_RE.match(text)
        if mi and self.last_meta_key in _AFTER_IKK_KEYS:
            self._ikk_open = False
            return False                 # '2. Capaian target PNBP ...' di bawah 'Indikator Rincian Output : 1.Jumlah ...'
        if mi and not any(rx.match(text) for _, rx, _r in META_RES):
            self.ikk.append(clean_text(mi.group(1)))
            self._ikk_open = True
            return True
        if self._ikk_open and self.ikk and line.x0 is not None and line.x0 > 150 \
                and not any(rx.match(text) for _, rx, _r in META_RES) and not re.search(r"\bRp\b", text):
            self.ikk[-1] = clean_text(self.ikk[-1] + " " + text)
            return True
        self._ikk_open = False
        return False

    @staticmethod
    def _norm_key(t: str) -> str:
        t = MASK_ANY_RE.sub(" ", t)
        return re.sub(r"[^0-9a-z]", "", t.lower())

    def _pre_redundant(self, text: str) -> bool:
        """Baris pra-tabel yang tidak perlu disimpan:
          - hanya placeholder masking ('[MASKED DATA UK ESELON 1 6]')
          - baris kegiatan di dalam tabel ('4485 <nama kegiatan> 1') -> kode kegiatan masuk metadata
          - teks yang sudah tercakup nilai metadata (kegiatan/KRO/kementerian/unit) atau sekadar ulangan
            nama kegiatan yang terpotong ke baris lain."""
        key = self._norm_key(text)
        if not key:
            return True
        mp = re.match(r"^([A-Z]{2})\s+Program\s+(\S.*)$", text)          # 'WA Program Dukungan Manajemen'
        if mp:
            self.meta.setdefault("kode_program", mp.group(1))
            self.meta.setdefault("program", mp.group(2))
            return True
        mk = re.match(r"^(\d{4})\s+\S.*\s\d{1,3}(?:[.,]\d{3})+$", text)  # '4489 <kegiatan> 9.644.197.000'
        if mk:
            self.meta.setdefault("kode_kegiatan", mk.group(1))
            return True
        m = re.match(r"^(\d{4})\s+(.*?)(?:\s+\d{1,3})?$", text)
        if m and not self._norm_key(m.group(2)):          # '4485 [MASKED ...] 1'
            self.meta.setdefault("kode_kegiatan", m.group(1))
            return True
        if m and re.fullmatch(r"\d{4}", text.split()[0]) and self._norm_key(m.group(2)) in self._meta_keys():
            self.meta.setdefault("kode_kegiatan", m.group(1))
            return True
        return len(key) >= 8 and key in self._meta_keys()

    def _meta_keys(self) -> str:
        return "|".join(self._norm_key(str(v)) for k, v in self.meta.items()
                        if k in ("kementerian", "unit_eselon2_satker", "kegiatan", "kro", "ro", "satuan_ukur")
                        and isinstance(v, str))

    def _add_node(self, ntype: str, kode: str, rest: list[str], line: Line):
        jenis = None
        vol_so = None
        mj = JENIS_RE.search(" ".join(rest))
        if mj:
            jenis = mj.group(1).capitalize()
            rest = rest[:-1]
        rest = [t for t in rest if t.lower() not in ("rp", "rp.")]
        head, declared, rawtok = split_amount_tail(rest)
        # Akun yang memuat detail item pada baris yang sama:
        #   '511111 Belanja Gaji Pokok PNS 1 tahun 684.332.000 684.332.000'
        # -> akun + satu item (kalau tidak, nominalnya tidak pernah masuk ke total hitung).
        inline = None
        if ntype == "akun":
            h2, n2, _, note2, ri2 = resolve_tail(rest)
            ok = False
            if len(n2) >= 3 and abs(n2[-3] * n2[-2] - n2[-1]) < 0.5:
                ok = True                                            # volume x harga = jumlah
                if ri2 is None:
                    ri2 = find_rincian_for(h2, vol=n2[-3])
            elif len(n2) == 2:
                if ri2 is None:
                    ri2 = find_rincian_for(h2, harga=n2[0], jumlah=n2[1])
                prod = rincian_info(" ".join(h2[ri2:]))[1] if ri2 is not None else None
                ok = (prod is not None and abs(prod * n2[0] - n2[1]) < 0.5) or n2[0] == n2[1]
            if ok:
                cut = ri2 if ri2 is not None else len(h2)
                inline = (clean_text(" ".join(h2[:cut])), " ".join(h2[cut:]) or None, n2, note2)
                head, declared = h2[:cut], n2[-1]
        # node berformat item: volume, harga satuan, jumlah (volume x harga = jumlah) -> simpan ketiganya
        harga_node = None
        if ntype in ("ro", "komponen", "subkomponen") and declared is not None and not inline:
            h3, n3, _r3 = split_tail(rest, 3)
            if len(n3) == 3 and 0 < n3[0] < 100000 and n3[1] >= 1000 and abs(n3[0] * n3[1] - n3[2]) < 0.5:
                head, declared, vol_so, harga_node = h3, n3[2], n3[0], n3[1]
        # angka kecil tanpa pemisah ribuan pada KRO/RO/Komponen = volume sub output, bukan jumlah
        if declared is not None and ntype in ("kro", "ro", "komponen") and declared < 1000 and head:
            vol_so, declared = declared, None
        elif declared is not None and ntype in ("subkomponen", "akun") and 0 <= declared < 1000 and head \
                and rawtok and "." not in rawtok and "," not in rawtok:
            head = head + [rawtok]          # 'Pemeliharaan Kendaraan Roda 4' : '4' bukan nominal
            declared = None
        uraian = clean_text(" ".join(head))
        self._center_title = None
        node = Node(ntype, kode if ntype != "subkomponen" else None, uraian,
                    label=kode if ntype == "subkomponen" else None,
                    jumlah_tercantum=declared, volume_suboutput=vol_so, harga_satuan=harga_node,
                    jenis_komponen=jenis, halaman=line.page, x0=line.x0)
        if ntype == "subkomponen":
            self._last_sub = kode
        elif ntype in ("komponen", "ro", "kro"):
            self._last_sub = None
        if ntype == "komponen" and KOMP_RE.match(kode or "") and line.x0 is not None \
                and line.kode_limit is not None and line.x0 >= line.kode_limit:
            self._komp_code_x = max(getattr(self, "_komp_code_x", None) or 0.0, line.x0)
        self.push(node)
        if getattr(line, "title_pre", ""):
            node.uraian = clean_text(f"{line.title_pre} {node.uraian}")
        if inline:
            utxt, rinc, nums, note2 = inline
            item = self._make_item(utxt, None, rinc, nums, line)
            if note2:
                item.setdefault("catatan_parser", []).append(note2)
                self.warn(f"{utxt}: {note2}", line)
            node.items.append(item)
            self.last_was_node = False

    def _add_content(self, toks: list[str], line: Line):
        if len(toks) > 1 and toks[0] in ("-", "\u2022", "\u2013"):   # bullet '- Kertas HVS ...'
            toks = toks[1:]
        toks = [t for t in toks if t.lower() not in ("rp", "rp.")]     # template 'Rp 1.000.000'
        if not toks:
            return
        head, nums, raw, fix_note, ri = resolve_tail(toks)
        # judul yang diakhiri tahun + subtotal ('B. Penyusunan ... E-SKP Tahun 2026 111.830.000'):
        # '2026' bagian judul, bukan harga satuan.
        if len(nums) == 2 and len(raw) == 2 and ri is None and re.fullmatch(r"(19|20)\d{2}", raw[0]) \
                and head and head[-1].rstrip(".:").upper() in ("TAHUN", "TA", "T.A", "THN") \
                and nums[1] >= 1000 and nums[1] != nums[0]:
            head, nums, raw = head + [raw[0]], nums[1:], raw[1:]
        if ri is None:
            if len(nums) >= 3:
                ri = find_rincian_for(head, vol=nums[-3])
            elif len(nums) == 2:
                ri = find_rincian_for(head, harga=nums[0], jumlah=nums[1])
            else:
                ri = find_rincian(head)
        # 'Tahun 2026 Triwulanan' bukan rincian '2026 Triwulanan' (volume + satuan): tahun bagian judul
        if ri is not None and 0 < ri < len(head) and re.fullmatch(r"(19|20)\d{2}", head[ri]) \
                and head[ri - 1].rstrip(".:").upper() in ("TAHUN", "TA", "T.A", "THN"):
            ri = None
        if ri is None:
            uraian_tok, rinc_tok = head, None
        else:
            uraian_tok, rinc_tok = head[:ri], head[ri:]
        uraian = clean_text(" ".join(uraian_tok))
        rinc = " ".join(rinc_tok) if rinc_tok else None

        # buang rumus yang diulang di uraian ('... 2 Rim x 6 Keg =') bila sudah diwakili rincian/volume
        if uraian and nums:
            _refs = [rincian_info(rinc)[1]] if rinc else []
            if len(nums) >= 3:
                _refs.append(nums[-3])
            _u, _rumus, _beda = strip_rumus_uraian(uraian, [r for r in _refs if r is not None])
            if _beda:
                self.warn(f"Rumus di uraian tidak sama dengan rincian/volume, uraian dibiarkan: '{uraian}'", line)
            uraian = _u
            self._rumus_dibuang = _rumus
            if _rumus and not rinc:
                rinc = re.sub(r"\s*=\s*$", "", _rumus)     # rincian kosong: pakai rumus dari uraian

        # ---- sel tabel yang diratakan tengah-vertikal: satu baris tabel pecah jadi 2-3 baris PDF ----
        #   '1 Ketua Tim ... (Lead Auditor 2' / 'Rp 27.500.000 Rp 55.000.000' / '27001:2022) 1 OH x 2 Bln'
        #   '2' / 'Rp 18.870.000 Rp 37.740.000' / '2 Tenaga Ahli ISO 27001 (...) 1 OH x 2 Bln'
        # Hanya dirakit bila volume x harga = jumlah.
        open_item, self._num_item = getattr(self, "_num_item", None), None
        pend_vol, self._pend_vol = getattr(self, "_pend_vol", None), None
        if open_item is not None and not nums and (rinc or uraian):
            if uraian:
                open_item["uraian"] = clean_text(f"{open_item['uraian'] or ''} {uraian}")
            if rinc and not open_item.get("rincian_perhitungan"):
                f_, p_, s_ = rincian_info(rinc)
                open_item.update(rincian_perhitungan=rinc, faktor=f_, satuan=s_)
            return
        if not uraian and rinc is None and not self.pending and len(nums) == 1 and len(raw) == 1 \
                and re.fullmatch(r"\d{1,3}", raw[0]) and self.state == "table":
            self._pend_vol = nums[0]          # volume sendirian di kolom volume
            return
        if not uraian and rinc is None and not self.pending and len(nums) == 2 and nums[0] >= 1000:
            cur = self.current
            vol, host, utxt = None, cur, ""
            if pend_vol and abs(pend_vol * nums[0] - nums[1]) < 0.5:
                vol = pend_vol
            elif cur is not None and cur.tipe == "kelompok" and not cur.items and not cur.children \
                    and cur.jumlah_tercantum is not None and 0 < cur.jumlah_tercantum < 1000 \
                    and float(cur.jumlah_tercantum).is_integer() \
                    and abs(cur.jumlah_tercantum * nums[0] - nums[1]) < 0.5:
                # kelompok '1 Ketua Tim ... 2' sebenarnya item: angka kecilnya = volume
                vol, host = cur.jumlah_tercantum, cur.parent
                pv = getattr(cur, "_prev", None)
                anc = pv.parent if pv is not None else None          # judul boleh bersarang (_rincian_host)
                while anc is not None and anc is not cur.parent and anc.tipe == "kelompok":
                    anc = anc.parent
                if pv is not None and pv is not cur and pv.tipe == "kelompok" and anc is cur.parent \
                        and pv.jumlah_tercantum is None and not pv.children:
                    host = pv                   # item milik kelompok judul sebelumnya ('I Biaya Langsung Personel')
                utxt = clean_text(f"{cur.label + ' ' if cur.label else ''}{cur.uraian}")
                sibs = cur.parent.children if cur.parent is not None else self.roots
                sibs.remove(cur)
                self.current = host
            if vol is not None and self.current is not None:
                item = self._make_item(utxt, None, None, [vol, nums[0], nums[1]], line)
                self.current.items.append(item)
                self._num_item = item
                self.last_was_node = False
                return

        # tanpa angka sama sekali -> kandidat uraian/kelompok
        if not nums and rinc is None:
            # judul berlabel berurutan ('I Biaya Langsung Personel' lalu 'A Tenaga Ahli'): judul sebelumnya
            # adalah kelompok tersendiri, bukan awal kalimat yang terbungkus.
            if self.pending and self.state == "table" and _HEAD_LABEL_RE.match(uraian) \
                    and (_HEAD_LABEL_RE.match(self.pending[0]) or _ROMAN_HEAD_RE.match(uraian)):
                lab, gtxt = self._extract_group_label(clean_text(" ".join(self.pending)))
                self.pending = []
                self.push(Node("kelompok", None, gtxt, label=lab, halaman=line.page))
            self.pending.append(uraian)
            self.pending_page = line.page
            return

        # judul kelompok bernomor satu kata ('7 Lampung 142.114.000') terbaca sebagai rincian '7 Lampung':
        # baris ada di kolom uraian (kiri), bukan kolom rincian, dan hanya membawa satu nominal.
        if not uraian and rinc and len(nums) == 1 and nums[0] >= 1000 and line.x0 is not None and line.pw \
                and line.x0 / line.pw < 0.2 and re.fullmatch(r"\d{1,2}\s+[A-Za-z](?!.*\bx\b).*", rinc, re.I):
            uraian, rinc = clean_text(rinc), None

        # baris hanya angka -> uraian dari pending
        group_text = None
        n_above = len(self.pending) if not uraian else 0
        if uraian and self.pending and len(nums) == 1 and rinc is None and _LABEL_ONLY_RE.fullmatch(uraian):
            # '3) 214.341.000' dengan judul 'Pembahasan dan Pertemuan berkala ...' di atas & bawahnya
            n_above = len(self.pending)
            uraian = clean_text(f"{uraian} {' '.join(self.pending)}")
            self.pending = []
        if not uraian:
            uraian = self._join_pending()
            self.pending = []
            if not uraian and len(nums) == 1 and rinc is None:
                # angka sendirian di kolom kanan = pengulangan subtotal kelompok/akun, bukan item baru
                cur = self.current
                if cur is not None and cur.jumlah_tercantum is None:
                    cur.jumlah_tercantum = nums[0]
                elif nums[0] >= 1000 and self.state == "table":
                    # judul baris tersamar ('[MASKED ...]' terbuang) -> node sementara; posisinya ditentukan
                    # aritmetika di finish() (_nest_groups/_lift_overflow). Yang tetap kosong dibuang lagi.
                    self.push(Node("kelompok", None, PLACEHOLDER_URAIAN, jumlah_tercantum=nums[0],
                                   halaman=line.page, placeholder=True))
                else:
                    self.warn(f"Angka subtotal tanpa uraian ({fmt_idr(nums[0])}) diabaikan", line)
                self.last_was_node = False
                return
        elif self.pending:
            group_text = self._join_pending()
            self.pending = []

        if group_text:
            lab, gtxt = self._extract_group_label(group_text)
            self.push(Node("kelompok", None, gtxt, label=lab, halaman=line.page))
        self.last_was_node = False

        # nomor standar/regulasi di ujung judul ('Dokumen ISO 27001', 'SNI 8799') bukan nominal
        if len(nums) == 1 and rinc is None and uraian and not group_text and re.fullmatch(r"\d{3,6}", toks[-1]) \
                and len(toks) >= 2 and toks[-2].upper() in ("ISO", "SNI", "IEC", "UU", "PP", "PERMEN", "NO", "NO.",
                                                            "NOMOR", "TAHUN"):
            self.pending.append(clean_text(f"{uraian} {toks[-1]}"))
            self.pending_page = line.page
            return

        # tahun di ujung judul ('- Tracer Study 2025') bukan subtotal: perlakukan sebagai teks
        if len(nums) == 1 and rinc is None and uraian and not group_text and re.fullmatch(r"(19|20)\d{2}", toks[-1]) \
                and nums[0] == int(toks[-1]):
            self.pending.append(clean_text(f"{uraian} {toks[-1]}"))
            self.pending_page = line.page
            return

        # satuan volume RO/KRO/komponen yang terbungkus ke baris berikutnya bersama nominalnya
        # ('4492.EBA.001 Layanan Umum' / 'Layanan 962.362.000' di kolom volume): bukan kelompok.
        cur = self.current
        if len(nums) == 1 and rinc is None and uraian and not group_text and nums[0] >= 1000 \
                and cur is not None and cur.tipe in ("kro", "ro", "komponen") and cur.jumlah_tercantum is None \
                and not cur.items and not cur.children and len(uraian.split()) <= 2 \
                and re.fullmatch(r"[A-Za-z/]+(?: [A-Za-z/]+)?", uraian) \
                and line.x0 is not None and line.pw and line.x0 / line.pw > 0.3:
            cur.jumlah_tercantum = nums[0]
            cur.satuan_suboutput = cur.satuan_suboutput or uraian
            self.last_was_node = False
            return

        # judul komponen/subkomponen terbungkus, nominalnya di baris lanjutan:
        #   'C Audit Surveillance ISO 27001:2022 ... atas Pengelolaan Data Industri dan'
        #   'Pelayanan Pelaporan Penyelenggara Pos, ... Portal Pelaporan Pos dan Informatika Rp 278.058.000'
        # Hanya bila judulnya jelas terpotong (berakhir kata sambung / koma) dan barisnya menjorok.
        if len(nums) == 1 and rinc is None and uraian and not group_text and nums[0] >= 1000 \
                and cur is not None and cur.tipe in ("komponen", "subkomponen") and cur.jumlah_tercantum is None \
                and not cur.items and not cur.children \
                and re.search(r"(?:\b(?:dan|atau|serta|untuk|pada|di|ke|dari|dengan|yang|atas|bagi|melalui)|[,&/-])$",
                              cur.uraian or "", re.I) \
                and line.x0 is not None and cur.x0 is not None and line.x0 > cur.x0 + 5:
            cur.uraian = clean_text(f"{cur.uraian} {uraian}")
            cur.jumlah_tercantum = nums[0]
            return

        # subtotal kelompok: teks + 1 angka, tanpa rincian
        if len(nums) == 1 and rinc is None and uraian:
            lab, utxt = self._extract_group_label(uraian)
            prev = self.current
            self.push(Node("kelompok", None, utxt, label=lab, jumlah_tercantum=nums[0],
                           halaman=line.page))
            self.current._prev = prev           # dipakai bila 'kelompok' ini ternyata item (angkanya volume)
            self._center_title = (self.current, n_above) if n_above else None
            self.last_was_node = False
            return

        _rumus_item, self._rumus_dibuang = getattr(self, "_rumus_dibuang", None), None
        lab, utxt = self._extract_label(uraian)
        if lab and not group_text:
            # huruf (a, b, c...) pada baris item menandai awal kelompok baru (mis. satu perjalanan)
            self.push(Node("kelompok", None, utxt, label=lab, halaman=line.page))
            self.last_was_node = False
        item = self._make_item(utxt, lab, rinc, nums, line)
        if _rumus_item:
            item["rumus_uraian"] = re.sub(r"\s*=\s*$", "", _rumus_item)   # rumus yang dibuang dari uraian (arsip)
        if fix_note:
            item.setdefault("catatan_parser", []).append(fix_note)
            self.warn(f"{utxt}: {fix_note}", line)
        if self.current is None:
            self.orphans.append(item)
            self.warn(f"Item tanpa induk hierarki: '{utxt}'", line)
        else:
            self.current.items.append(item)

    def _make_item(self, uraian, label, rinc, nums, line: Line) -> dict:
        faktor, produk, satuan = rincian_info(rinc)
        vol = harga = jumlah = None
        jumlah_src = "pdf"
        vol_src: Optional[str] = "pdf"     # pdf | rincian | asumsi | None  (asal nilai volume)
        notes: list[str] = []
        if len(nums) >= 3:
            vol, harga, jumlah = nums[-3:]
        elif len(nums) == 2:
            a, b = nums
            if produk is not None and abs(produk * a - b) < 0.5:
                vol, harga, jumlah = produk, a, b               # volume tertulis di rincian ('12 bln')
                vol_src = "rincian"
            elif produk is not None and a > b and abs(produk * b - a) < 0.5:
                vol, harga, jumlah = produk, b, a               # kolom Jumlah & Harga Satuan tertukar di dokumen
                vol_src = "rincian"
                notes.append("kolom jumlah dan harga satuan tertukar posisi di dokumen; dikoreksi dari aritmetika")
            elif produk is not None and a < 1000 and abs(produk * a * 1000 - b) < 0.5:
                vol, harga, jumlah = produk, a * 1000, b        # '350 OK 390 136.500.000': harga dalam ribuan
                vol_src = "rincian"
                notes.append(f"harga satuan tercetak '{a:g}' (dalam ribuan); dibaca {fmt_idr(a * 1000)} "
                             f"karena volume x harga = jumlah")
            elif a == b:
                vol, harga, jumlah = 1, a, b                    # kolom harga satuan berisi total
                vol_src = "asumsi"
                notes.append("harga satuan sama dengan jumlah; volume dianggap 1")
            elif a < 1000 and float(a).is_integer():
                vol, harga = a, b                               # (volume, harga) tanpa jumlah
                jumlah = vol * harga
                jumlah_src = "dihitung"
                notes.append("jumlah tidak terbaca; dihitung dari volume x harga")
            else:                                               # (harga, jumlah) tanpa kolom volume
                harga, jumlah = a, b
                vol, vol_src = produk, ("rincian" if produk is not None else None)
                notes.append("kolom volume tidak terbaca; harga satuan dan jumlah diambil dari dua angka terakhir")
        elif len(nums) == 1:
            mv = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+\S", rinc or "")
            v_rinc = parse_num(mv.group(1)) if mv else None
            if rinc and v_rinc and nums[0] >= 1000 and nums[0] > v_rinc:
                # 'BBSDMP Medan 1 Paket 421.923.000': volume di rincian, satu nominal = jumlah (harga kosong)
                vol, jumlah, jumlah_src = v_rinc, nums[0], "pdf"
                vol_src = "rincian"
                if vol and float(nums[0] / vol).is_integer():
                    harga = nums[0] / vol if vol != 1 else nums[0]
                notes.append("hanya satu nominal setelah rincian; dianggap jumlah (harga = jumlah / volume)")
            elif rinc:
                vol = nums[0]
                notes.append("hanya satu angka terbaca setelah rincian")
            else:
                jumlah = nums[0]
                jumlah_src = "pdf"
                notes.append("hanya satu angka terbaca")
        if vol is None and produk is not None:
            vol, vol_src = produk, "rincian"
        item = {
            "label": label,
            "uraian": uraian,
            "rincian_perhitungan": rinc,
            "faktor": faktor,
            "volume": vol,
            "volume_sumber": vol_src,
            "satuan": satuan,
            "harga_satuan": harga,
            "jumlah": jumlah,
            "jumlah_sumber": jumlah_src,
            "halaman": line.page,
        }
        if notes:
            item["catatan_parser"] = notes
            self.warn(f"{uraian or '(tanpa uraian)'}: " + "; ".join(notes), line)
        item["_ctx"] = self.context()
        self._n_items = getattr(self, "_n_items", 0) + 1
        self._last_item = item
        return item

    # ---- finalisasi --------------------------------------------------------------
    def finish(self) -> dict:
        self.flush_pending(None, as_continuation=False)
        if getattr(self, "_lokasi_pre", None) and self.roots and not self.roots[0].lokasi:
            self.roots[0].lokasi = self._lokasi_pre         # 'Lokasi : Pusat' di kop -> node akar tabel
        _computed_items_to_titles(self.roots)
        _nest_groups(self.roots)
        global _LIFT_ROOTS
        _LIFT_ROOTS = self.roots
        _lift_overflow(self.roots)
        _nest_groups(self.roots)
        if _prune_placeholders(self.roots):
            self.warnings.append("Angka subtotal tanpa uraian yang tidak menampung baris apa pun diabaikan "
                                 "(pengulangan subtotal)")
        for r in self._walk():
            if r.placeholder:
                self.warn(f"Baris bernominal {fmt_idr(r.jumlah_tercantum)} tanpa judul terbaca (tersamar/terpotong); "
                          f"dipasang sebagai kelompok '{PLACEHOLDER_URAIAN}' sesuai aritmetika dokumen")

        # kode kegiatan / KRO / RO dari hierarki
        for r in self._walk():
            if r.tipe == "kro" and r.kode:
                a, b = r.kode.split(".")
                self.meta.setdefault("kode_kegiatan", a)
                self.meta.setdefault("kode_kro", b)
            if r.tipe == "ro" and r.kode:
                self.meta.setdefault("kode_ro", r.kode.split(".")[-1])
                self.meta.setdefault("kode_ro_lengkap", r.kode)
        # KRO & RO WAJIB dari header di atas tabel. Nilai dari baris tabel dipisah ke *_tabel
        # (hanya pembanding), tidak pernah mengisi 'kro' / 'ro'.
        for _t in self._walk():
            if _t.tipe == "kro" and "kro_tabel" not in self.meta:
                self.meta["kro_tabel"] = f"{_t.kode or ''} {_t.uraian or ''}".strip()
            if _t.tipe == "ro" and "ro_tabel" not in self.meta:
                self.meta["ro_tabel"] = f"{_t.kode or ''} {_t.uraian or ''}".strip()
        for _k in ("kro", "ro"):
            _v = self.meta.get(_k)
            if not _v:
                self.warn(f"Header '{_k.upper()}' tidak ditemukan di atas tabel; field '{_k}' dikosongkan "
                          f"(nilai tabel hanya di '{_k}_tabel').")
            elif isinstance(_v, str):
                _mk = re.search(r"\b(\d{4})\.([A-Z]{3})(?:\.(\d{3}))?\b", _v)
                if _mk:
                    self.meta[f"kode_{_k}_header"] = _mk.group(0)
        if self.meta.get("volume") is None:
            self.warn("Header 'Volume' tidak ditemukan / tidak terbaca di atas tabel.")
        if not self.meta.get("satuan_ukur") and self.meta.get("volume_satuan_inline"):
            self.meta["satuan_ukur"] = self.meta["volume_satuan_inline"]
            self.meta["satuan_ukur_sumber"] = "menempel_di_volume"
        if self.ikk:
            self.meta["indikator_kinerja_kegiatan"] = self.ikk
        elif len(self._ikk_blk) > 1 and isinstance(self.meta.get("indikator_kinerja_kegiatan"), str):
            butir = _split_ikk_block(self._ikk_blk)
            if len(butir) > 1:                  # beberapa indikator -> daftar (satu butir tetap teks gabungan)
                self.meta["indikator_kinerja_kegiatan"] = butir
        self._resolve_tahun()
        alok = self._alok()
        if alok is not None:
            self.meta["alokasi_dana"] = alok
        else:
            self.meta.pop("alokasi_dana", None)
        if self.alokasi_header is not None:
            self.meta.setdefault("alokasi_header", self.alokasi_header)
        if self.masked:
            self.meta["placeholder_masking"] = self.masked
        if self.meta_tersamar:
            self.meta["field_tersamar"] = sorted(set(self.meta_tersamar))
        if isinstance(self.meta.get("kegiatan"), str):
            self.meta["kegiatan"] = re.sub(r"\s*Klasifikasi(?:\s+Rincian\s+Output)?$", "", self.meta["kegiatan"]).strip()

        items_flat = []
        for n in self._walk():
            for it in n.items:
                items_flat.append(it)
        items_flat += self.orphans
        flat = []
        for no, it in enumerate(items_flat, 1):
            ctx = it.pop("_ctx", {})
            ctx["kro_tabel"], ctx["ro_tabel"] = ctx.get("kro"), ctx.get("ro")
            ctx["kro"], ctx["ro"] = self.meta.get("kro"), self.meta.get("ro")      # header, bukan tabel
            row = {"no": no, **ctx, **it}
            flat.append(row)
        # item di pohon: buang _ctx
        for n in self._walk():
            for it in n.items:
                it.pop("_ctx", None)

        return {
            "source_file": self.source,
            "parser_version": PARSER_VERSION,
            "metadata": self.meta,
            "penandatangan": self.ttd,
            "total": {
                "jumlah_total_tercantum": self.total_pdf,
                "alokasi_dana": self.meta.get("alokasi_dana"),
            },
            "struktur": [r.to_dict() for r in self.roots],
            "items": flat,
            "peringatan": self.warnings,
        }

    def _alok(self):
        """Alokasi dana sebagai angka (None bila tidak ada / bukan angka)."""
        for cand in (self.meta.get("alokasi_dana"), self.alokasi_header):
            if isinstance(cand, (int, float)) and not isinstance(cand, bool):
                return cand
        return None

    def _walk(self):
        stack = list(reversed(self.roots))
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))

# --------------------------------------------------------------------------------------
# Deteksi tanda tangan (PDF)
#   Tiga sinyal pada area di bawah jabatan penandatangan (setelah 'Jakarta, Juni 2025' / 'Sekretaris'):
#     1) gambar tertanam (tanda tangan hasil scan/stempel) di dalam area
#     2) objek vektor non-horizontal (kurva/garis miring) di dalam area -> tanda tangan digital/gambar tangan
#     3) rasio ink (piksel gelap) pada hasil render area, di luar teks
# --------------------------------------------------------------------------------------

def _rows_from_words(words: list[dict], y_tol: float = 3.0) -> list[dict]:
    rows: list[dict] = []
    for w in sorted(words, key=lambda w: (w["top"] + w["bottom"]) / 2):
        cy = (w["top"] + w["bottom"]) / 2
        if rows and abs(cy - rows[-1]["cy"]) <= y_tol:
            r = rows[-1]
            r["words"].append(w)
            r["cy"] = sum((x["top"] + x["bottom"]) / 2 for x in r["words"]) / len(r["words"])
        else:
            rows.append({"cy": cy, "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["x0"])
        r["text"] = clean_text(" ".join(w["text"] for w in r["words"]))
        r["x0"] = min(w["x0"] for w in r["words"])
        r["x1"] = max(w["x1"] for w in r["words"])
        r["top"] = min(w["top"] for w in r["words"])
        r["bottom"] = max(w["bottom"] for w in r["words"])
    return rows


def _overlap(a, b) -> float:
    """Luas irisan dua bbox (x0, top, x1, bottom)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


_DATE_ANCHOR_RE = re.compile(
    r"(?:[A-Za-z .]+|\[\s*MASK[^\]]*\]),\s*(?:(?:\d{1,2}|\[\s*MASK[^\]]*\])\s+)?(?:[A-Za-z]+|\[\s*MASK[^\]]*\])\s+"
    r"(?:\d{4}|\[\s*MASK[^\]]*\])$", re.I)


def _row_text_ok_anchor(t: str) -> bool:
    return bool(_DATE_ANCHOR_RE.search(t)) and not re.search(r"\d[\d.]*\s+\d", t)


def _find_anchor_index(rows: list[dict]) -> Optional[int]:
    """Baris 'Jakarta, 5 Juni 2026' (kota/tanggal boleh tersamar). Ambil yang TERAKHIR di halaman."""
    hit = [i for i, r in enumerate(rows) if _row_text_ok_anchor(r["text"])]
    return hit[-1] if hit else None


def _nip_in_row(row: dict) -> Optional[dict]:
    """Cari label NIP pada satu baris berbasis kata+posisi. Nilai NIP (tersamar/tidak) hanya dinilai statusnya."""
    words = row["words"]
    text, spans = "", []
    for w in words:
        spans.append((len(text), len(text) + len(w["text"]), w))
        text += w["text"] + " "
    m = NIP_LABEL_RE.search(text)
    if not m:
        return None
    # label harus di dekat awal kolomnya: kalimat biasa yang kebetulan memuat 'nip' diabaikan
    idx = [i for i, (st, en, _) in enumerate(spans) if en > m.start() and st < m.end()]
    if not idx:
        return None
    first, last = idx[0], idx[-1]
    lw = words[first:last + 1]
    placeholder = any(MASK_RE.fullmatch(w["text"]) for w in lw)
    tail = spans[last][2]["text"][max(0, m.end() - spans[last][0]):] if m.end() - spans[last][0] < len(spans[last][2]["text"]) else ""
    vals, prev = [], words[last]
    for w in words[last + 1:]:
        if w["x0"] - prev["x1"] > 45:
            break
        vals.append(w["text"])
        prev = w
    rest = " ".join(([tail] if tail else []) + vals)
    return {"row": row, "label_x0": lw[0]["x0"], "label_x1": lw[-1]["x1"], "value_x1": prev["x1"],
            "rest": rest, "placeholder": placeholder or bool(MASK_RE.fullmatch(rest.strip()) and "NIP" in rest.upper())}


def _dark(c) -> bool:
    if c is None:
        return False
    if isinstance(c, (int, float)):
        return c < 0.35
    try:
        v = [float(x) for x in c]
    except Exception:
        return False
    if not v:
        return False
    if len(v) == 4:                       # CMYK
        return v[3] > 0.6 or sum(v[:3]) > 2.4
    return sum(v) / len(v) < 0.35


def _is_white(c) -> bool:
    if c is None:
        return True
    if isinstance(c, (int, float)):
        return c >= 0.95
    try:
        v = [float(x) for x in c]
    except Exception:
        return False
    if not v:
        return True
    if len(v) == 4:                       # CMYK: putih = semua komponen ~0
        return max(v) <= 0.05
    return min(v) >= 0.95


def _visible_vector(o: dict) -> bool:
    """Objek vektor yang benar-benar tergambar: bergaris/berisi warna non-putih."""
    stroked = o.get("stroke", True) and not _is_white(o.get("stroking_color"))
    filled = bool(o.get("fill")) and not _is_white(o.get("non_stroking_color"))
    return bool(stroked or filled)


def _visual_mask_evidence(page, nip: dict) -> list[str]:
    """Nilai NIP bisa ditutup kotak hitam/gambar (redaksi visual) alih-alih teks placeholder."""
    ev = []
    row = nip["row"]
    reg = (nip["label_x1"], row["top"] - 2, min(page.width, nip["label_x1"] + 260), row["bottom"] + 2)
    for r in page.rects:
        rb = (r["x0"], r["top"], r["x1"], r["bottom"])
        h, w = rb[3] - rb[1], rb[2] - rb[0]
        if h < 30 and w > 12 and _overlap(rb, reg) > 0 and r.get("fill") and _dark(r.get("non_stroking_color")):
            ev.append("kotak gelap (redaksi visual) menutup nilai NIP")
            break
    for im in page.images:
        ib = (im["x0"], im["top"], im["x1"], im["bottom"])
        if (ib[3] - ib[1]) < 40 and _overlap(ib, reg) > 0:
            ev.append("gambar kecil di posisi nilai NIP")
            break
    return ev


def _ocr_nip(page, bbox) -> Optional[bool]:
    """Cadangan untuk halaman hasil scan (tanpa lapisan teks). None bila OCR tidak tersedia."""
    try:
        import pytesseract
        img = page.crop(bbox, strict=False).to_image(resolution=220).original.convert("L")
        txt = pytesseract.image_to_string(img, lang="ind+eng", config="--psm 6")
    except Exception:
        return None
    return bool(NIP_LABEL_RE.search(txt))


def _locate_signature(page, rows: list[dict]) -> dict:
    """Blok penandatangan pada satu halaman: jangkar tanggal, baris NIP, bbox blok, dan bbox ruang tanda tangan."""
    out: dict[str, Any] = {"anchor_i": None, "anchor": None, "nip": None, "block": None, "gap": None, "jabatan": None}
    ai = _find_anchor_index(rows)
    nips = []
    for i, r in enumerate(rows):
        n = _nip_in_row(r)
        if n:
            n["i"] = i
            nips.append(n)
    chosen = None
    if ai is not None:
        a = rows[ai]
        below = [n for n in nips if n["row"]["top"] >= a["top"] - 1 and n["row"]["top"] - a["bottom"] < 260]
        if below:
            ac = (a["x0"] + a["x1"]) / 2
            chosen = min(below, key=lambda n: (abs((n["label_x0"]) - a["x0"]) > 80, n["row"]["top"] - a["bottom"],
                                                abs(n["label_x0"] - ac)))
        out["anchor_i"], out["anchor"] = ai, a["text"]
    elif nips:
        chosen = nips[-1]
    out["nip"] = chosen

    if ai is None and chosen is None:
        # jangkar cadangan: baris jabatan di separuh bawah halaman
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["top"] > page.height * 0.45 and JABATAN_RE.search(rows[i]["text"]):
                out["anchor_i"], out["anchor"] = i, rows[i]["text"]
                out["jabatan"] = True
                ai = i
                break
    if ai is None and chosen is None:
        # jangkar cadangan 2: tanggal & NIP tidak terbaca (tersamar), tetapi nama pejabat tersamar ('[MASKED ... PEJABAT]')
        # tampak di separuh bawah halaman. Baris tepat di atasnya (jabatan) menjadi awal blok; ruang tanda tangan
        # = celah antara jabatan dan nama.
        for i in range(len(rows) - 1, -1, -1):
            r = rows[i]
            if r["top"] > page.height * 0.3 and MASK_RE.fullmatch(r["text"].strip()) and "PEJABAT" in r["text"].upper():
                j = i - 1 if i > 0 and r["top"] - rows[i - 1]["bottom"] < 120 else i
                out["anchor_i"], out["anchor"] = j, rows[j]["text"]
                out["jabatan"] = True
                ai = j
                break
    if ai is None and chosen is None:
        return out

    if ai is not None:
        a = rows[ai]
        end_i = chosen["i"] if chosen else min(len(rows) - 1, ai + 4)
        seq = rows[ai:end_i + 1]
        left = min(a["x0"], chosen["row"]["x0"] if chosen else a["x0"]) if chosen is None else min(a["x0"], chosen["label_x0"])
        right = max(a["x1"], chosen["value_x1"] if chosen else a["x1"])
        top_obj = a["bottom"] + 1
    else:
        end_i = chosen["i"]
        seq = [r for r in rows[max(0, end_i - 6):end_i + 1] if chosen["row"]["top"] - r["bottom"] < 150]
        left, right = chosen["label_x0"], chosen["value_x1"]
        top_obj = seq[0]["top"] if seq else chosen["row"]["top"] - 100
    bot_obj = (chosen["row"]["bottom"] + 1) if chosen else min(page.height, seq[-1]["bottom"] + 90)
    bl, br = max(0, left - 30), min(page.width, right + 30)
    out["block"] = (bl, top_obj, br, bot_obj)

    # ruang tanda tangan = celah vertikal terbesar antar baris dalam blok
    best, gap_box = 0.0, None
    out["gaps"] = []
    for r1, r2 in zip(seq, seq[1:]):
        g = r2["top"] - r1["bottom"]
        if g >= 6:
            out["gaps"].append((bl, r1["bottom"] + 1, br, r2["top"] - 1))
        if g > best:
            best, gap_box = g, (bl, r1["bottom"] + 1, br, r2["top"] - 1)
    if gap_box is not None and best >= 15:
        out["gap"] = gap_box
    elif chosen is None:
        a = rows[ai]
        nxt = rows[ai + 2] if ai + 2 < len(rows) else None
        top = (rows[ai + 1]["bottom"] + 1) if ai + 1 < len(rows) else a["bottom"] + 2
        bottom = nxt["top"] - 1 if nxt and nxt["top"] - top > 15 else min(page.height, top + 90)
        out["gap"] = (bl, top, br, bottom)
    else:
        out["gap"] = (bl, top_obj, br, max(top_obj + 2, chosen["row"]["top"] - 1))
    return out


_SIG_INTRO_RE = re.compile(r"^(?:Menyetujui|Mengetahui|Disetujui|Diperiksa|Dibuat\s+oleh)\b[ ,:.]*$", re.I)
_SIG_FOOT_RE = re.compile(r"^(?:Catatan\s*:|- UU ITE|\"Informasi Elektronik|- Dokumen ini telah)", re.I)
_MASK_FRAG_RE = re.compile(r"^\[M[A-Z]{0,5}\s+(?=\[\s*MASK)")       # '[MA [MASKED DATA]': fragmen tertimpa
_NAMA_HINT_RE = re.compile(r"PEJABAT|NAMA", re.I)


def _sig_cell_text(rows: list[dict]) -> Optional[str]:
    """Teks sel blok tanda tangan apa adanya; fragmen mask yang tertimpa ('[MA ') di depan placeholder dibuang."""
    if not rows:
        return None
    return clean_text(" ".join(_MASK_FRAG_RE.sub("", r["text"]) for r in rows)) or None


def _signature_text_block(page, rows: list[dict], loc: dict) -> Optional[dict]:
    """Isi teks blok penandatangan berdasarkan POSISI: [tanggal] -> jabatan (1+ baris) -> ruang tanda tangan
    (celah vertikal) -> nama -> NIP. Teks tidak diperbaiki; placeholder mask diambil apa adanya."""
    if not rows:
        return None
    ai = loc.get("anchor_i")
    n = loc.get("nip")
    ni = n["i"] if n else None
    date_row = None
    if ai is not None and _row_text_ok_anchor(rows[ai]["text"]) and "," in rows[ai]["text"]:
        date_row = rows[ai]
    if ai is not None:
        start = ai + 1 if date_row is not None else ai
        ref = rows[ai]
    elif ni is not None:
        start, ref = max(0, ni - 8), n["row"]
    else:
        # tanpa tanggal & NIP: baris pendek di bawah baris tabel (lebar) terakhir halaman
        wide = [i for i, r in enumerate(rows) if r["x1"] - r["x0"] > 250 and not _SIG_FOOT_RE.match(r["text"])]
        start = (wide[-1] + 1) if wide else 0
        cand = [r for r in rows[start:] if not _SIG_FOOT_RE.match(r["text"]) and r["x1"] - r["x0"] < 250
                and r["x0"] > page.width * 0.3]
        if not cand:
            return None
        ref = cand[0]
    end = ni if ni is not None else len(rows)
    rc = (ref["x0"] + ref["x1"]) / 2
    seq, prev_bot = [], (date_row["bottom"] if date_row is not None else None)
    for i in range(start, end):
        r = rows[i]
        if _SIG_FOOT_RE.match(r["text"]) or NIP_LABEL_RE.search(r["text"]):
            break
        if r["x1"] - r["x0"] > 300 or abs((r["x0"] + r["x1"]) / 2 - rc) > 130:
            if ni is None and ai is None and seq:
                break
            continue                                   # tabel paraf di kiri / baris tabel lebar
        if ni is None and seq and r["top"] - seq[-1]["bottom"] > 160:
            break
        seq.append(r)
    if ni is None and ai is not None and date_row is None and not seq:
        return None
    out: dict[str, Any] = {"tanggal_teks": date_row["text"] if date_row is not None else None,
                           "pengantar": None, "jabatan": None, "nama": None}
    while seq and _SIG_INTRO_RE.match(seq[0]["text"]):
        out["pengantar"] = clean_text(((out["pengantar"] or "") + " " + seq[0]["text"]))
        prev_bot = seq[0]["bottom"]
        seq = seq[1:]
    if not seq:
        return out
    nip_top = n["row"]["top"] if n else None
    gaps = [(seq[k + 1]["top"] - seq[k]["bottom"], k) for k in range(len(seq) - 1)]
    big = max(gaps) if gaps else None
    if big and big[0] >= 15:
        out["jabatan"] = _sig_cell_text(seq[:big[1] + 1])
        out["nama"] = _sig_cell_text(seq[big[1] + 1:])
    elif len(seq) >= 2:
        if nip_top is not None and nip_top - seq[-1]["bottom"] < 15:
            out["jabatan"], out["nama"] = _sig_cell_text(seq[:-1]), _sig_cell_text(seq[-1:])
        else:
            out["jabatan"] = _sig_cell_text(seq)
    else:
        r = seq[0]
        above = (r["top"] - prev_bot) if prev_bot is not None else None
        below = (nip_top - r["bottom"]) if nip_top is not None else None
        if below is not None and below < 15 and not (above is not None and above < 15):
            out["nama"] = _sig_cell_text(seq)
        elif above is not None and above < 15:
            out["jabatan"] = _sig_cell_text(seq)
        elif _NAMA_HINT_RE.search(r["text"]) and not JABATAN_RE.search(r["text"]):
            out["nama"] = _sig_cell_text(seq)
        else:
            out["jabatan"] = _sig_cell_text(seq)
    return out


def _find_signature_region(page, rows: list[dict]):
    """Kompatibilitas lama: (bbox ruang tanda tangan, teks jangkar) atau (None, None)."""
    loc = _locate_signature(page, rows)
    return (loc["gap"], loc["anchor"]) if loc["gap"] else (None, None)


def detect_signature(pdf_path: str | Path, resolution: int = 150, ink_threshold: float = 0.004,
                     dark_level: int = 170, debug_dir: Optional[str | Path] = None,
                     pages: Optional[tuple[int, int]] = None) -> dict:
    """
    Deteksi blok penanggung jawab: tanda tangan + NIP.
      - Tanda tangan: gambar tertanam, objek vektor non-horizontal, rasio ink (di ruang antara jabatan dan nama).
      - NIP: nilainya hampir selalu tersamar, jadi yang diperiksa adalah ADA TIDAKNYA LABEL 'NIP' pada baris
        di bawah nama dalam blok tanda tangan (teks -> placeholder '[MASKED..]' -> OCR bila halaman scan).
        Nilai hanya diklasifikasikan (tersamar / terbaca valid 18 digit / kosong), tidak disalin.
    Return: {detected, confidence, halaman, bbox, anchor, evidence, ink_ratio, method, blok_bbox, nip:{...}}
    """
    import pdfplumber

    res: dict[str, Any] = {"detected": False, "confidence": None, "halaman": None, "bbox": None,
                           "anchor": None, "evidence": [], "ink_ratio": None,
                           "method": "gambar+vektor+ink", "blok_bbox": None,
                           "nip": {"ada_label": False, "sumber": None, "nilai_status": None, "halaman": None,
                                   "bbox": None, "confidence": None, "evidence": []}}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            lo, hi = pages if pages else (1, len(pdf.pages))
            hi = min(hi, len(pdf.pages))
            found = None                                   # (pno, page, rows, loc)
            fallback_nip = None
            for pno in range(hi, lo - 1, -1):              # dari halaman terakhir
                page = pdf.pages[pno - 1]
                rows = _rows_from_words(page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False))
                loc = _locate_signature(page, rows)
                if loc["anchor_i"] is not None or loc["nip"] is not None:
                    found = (pno, page, rows, loc)
                    break
            fallback = False
            if found is None:
                fallback = True
                page = pdf.pages[hi - 1]
                bbox = (0, page.height * 0.70, page.width, page.height)
                found = (hi, page, [], {"anchor": None, "nip": None, "block": bbox, "gap": bbox})
                res["evidence"].append("anchor tanggal/jabatan/NIP tidak ditemukan; memakai 30% bawah halaman terakhir")
            pno, page, rows, loc = found
            try:
                trows = rows or _rows_from_words(page.extract_words(x_tolerance=2, y_tolerance=2,
                                                                    keep_blank_chars=False))
                res["blok_teks"] = _signature_text_block(page, trows, {"anchor_i": loc.get("anchor_i"),
                                                                       "nip": loc.get("nip")})
            except Exception as e:
                res["evidence"].append(f"pembacaan teks blok tanda tangan gagal: {e}")
            gap, block, anchor = loc["gap"], loc["block"], loc["anchor"]
            res.update(halaman=pno, bbox=[round(v, 1) for v in gap], anchor=anchor,
                       blok_bbox=[round(v, 1) for v in block])
            page_area = page.width * page.height

            # ---------- NIP ----------
            nip = res["nip"]
            n = loc["nip"]
            if n is None and loc["anchor"] is not None and pno < hi:        # blok terbelah ke halaman berikut
                nxt = pdf.pages[pno]
                nrows = _rows_from_words(nxt.extract_words(x_tolerance=2, y_tolerance=2))
                for r in nrows[:10]:
                    nn = _nip_in_row(r)
                    if nn:
                        n, page_n = nn, nxt
                        nip["halaman"] = pno + 1
                        break
            else:
                page_n = page
            if n is not None:
                row = n["row"]
                mph = re.search(r"\[\s*MASK[^\]]*NIP[^\]]*\]", row["text"], re.I)      # placeholder utuh '[MASKED DATA-NIP-22]'
                if mph:
                    n["placeholder"] = True
                st, nd = classify_nip_value(n["rest"])
                if n["placeholder"]:
                    st = "tersamar"
                nilai = clean_text(mph.group(0)) if mph else nip_value(n["rest"], n["placeholder"])
                nip.update(ada_label=True, sumber="placeholder" if n["placeholder"] else "teks",
                           nilai_status=st, jumlah_digit=nd, nilai=nilai, halaman=nip["halaman"] or pno,
                           bbox=[round(v, 1) for v in (n["label_x0"], row["top"], max(n["value_x1"], n["label_x1"]), row["bottom"])],
                           confidence="sedang" if n["placeholder"] else "tinggi", teks_baris=redact_digits(row["text"]))
                nip["evidence"].append("label NIP pada lapisan teks" if not n["placeholder"]
                                       else "label NIP di dalam placeholder masking")
                if st == "kosong":
                    ev = _visual_mask_evidence(page_n, n)
                    if ev:
                        nip["nilai_status"] = "tersamar"
                        nip["evidence"] += ev
                loc_i = n.get("i")
                if loc_i and loc_i > 0 and not res.get("nama_ada"):
                    above = rows[loc_i - 1]
                    if not NIP_LABEL_RE.search(above["text"]) and row["top"] - above["bottom"] < 20:
                        res["nama_ada"] = True
                        res["nama_teks"] = above["text"][:80]
            else:
                # halaman scan / label tertanam sebagai gambar: OCR terbatas pada separuh bawah blok
                if not rows or not any(len(r["text"]) > 3 for r in rows):
                    box = (0, page.height * 0.55, page.width, page.height)
                else:
                    box = (block[0], block[1], block[2], min(page.height, block[3] + 70))
                o = _ocr_nip(page, box)
                if o:
                    nip.update(ada_label=True, sumber="ocr", nilai_status="tidak_diketahui", halaman=pno,
                               bbox=[round(v, 1) for v in box], confidence="rendah")
                    nip["evidence"].append("label NIP terbaca lewat OCR")
                elif o is None:
                    nip["evidence"].append("label NIP tidak ditemukan di lapisan teks (OCR tidak tersedia)")
                else:
                    nip["evidence"].append("label NIP tidak ditemukan di lapisan teks maupun OCR")

            # ---------- tanda tangan ----------
            for im in page.images:
                ib = (im["x0"], im["top"], im["x1"], im["bottom"])
                area = max(1e-6, (ib[2] - ib[0]) * (ib[3] - ib[1]))
                if area > 0.6 * page_area:
                    continue
                if _overlap(ib, block) / area >= 0.3 and _overlap(ib, block) > 0 \
                        and not (nip["bbox"] and _overlap(ib, tuple(nip["bbox"])) / area >= 0.5 and (ib[3] - ib[1]) < 20):
                    res["evidence"].append(f"gambar tertanam {ib[2]-ib[0]:.0f}x{ib[3]-ib[1]:.0f}pt di blok tanda tangan")
            nvec = 0
            nvec_white = 0
            for c in page.curves:
                if _overlap((c["x0"], c["top"], c["x1"], c["bottom"]), block) > 0:
                    if _visible_vector(c):
                        nvec += 1
                    else:
                        nvec_white += 1
            for ln in page.lines:
                if abs(ln["bottom"] - ln["top"]) > 1.5 and _overlap((ln["x0"], ln["top"], ln["x1"], ln["bottom"]), block) > 0:
                    if _visible_vector(ln):
                        nvec += 1
                    else:
                        nvec_white += 1
            if nvec:
                res["evidence"].append(f"{nvec} objek vektor (kurva/garis miring) di blok tanda tangan")
            if nvec_white:
                res["evidence"].append(f"abaikan: {nvec_white} objek vektor putih/tak terlihat di blok (bukan tanda tangan)")
            try:
                cands = list(loc.get("gaps") or []) or [gap]
                if gap not in cands:
                    cands.append(gap)
                best_ink, best_box = -1.0, gap
                for gb in cands:
                    img = page.crop(gb, strict=False).to_image(resolution=resolution).original.convert("L")
                    hist = img.histogram()
                    ink = sum(hist[:dark_level]) / (sum(hist) or 1)
                    if ink > best_ink:
                        best_ink, best_box = ink, gb
                res["ink_ratio"] = round(best_ink, 5)
                res["bbox"] = [round(v, 1) for v in best_box]
                if res["ink_ratio"] >= ink_threshold:
                    res["evidence"].append(f"ink ratio {res['ink_ratio']:.4f} >= {ink_threshold} (celah antar-baris di blok)")
                if debug_dir:
                    d = Path(debug_dir)
                    d.mkdir(parents=True, exist_ok=True)
                    page.crop(block, strict=False).to_image(resolution=resolution).save(
                        d / f"{Path(pdf_path).stem}_blok_p{pno}.png")
            except Exception as e:
                res["evidence"].append(f"render ink dilewati: {e}")

            has_img = any(e.startswith("gambar") for e in res["evidence"])
            has_vec = nvec > 0
            ink_hit = (res["ink_ratio"] or 0) >= ink_threshold
            ink_half = (res["ink_ratio"] or 0) >= ink_threshold / 2
            # vektor terlihat HARUS didukung tinta di hasil render; vektor tanpa tinta = tidak dihitung
            res["detected"] = bool(has_img or ink_hit or (has_vec and ink_half))
            # Tanpa anchor (tanggal/jabatan/NIP) yang menandai blok tanda tangan, 30% bawah halaman biasanya
            # masih berisi tabel/garis/teks -> tinta di situ BUKAN bukti tanda tangan. Pada halaman yang punya
            # lapisan teks, hanya gambar tertanam atau objek vektor yang boleh dihitung; tinta saja tidak.
            if fallback and page.chars and not (has_img or has_vec):
                if res["detected"]:
                    res["evidence"].append("tinta di 30% bawah halaman tanpa anchor blok tanda tangan (kemungkinan "
                                           "tabel/teks): tidak dihitung sebagai tanda tangan")
                res["detected"] = False
                ink_hit = ink_half = False
            if has_vec and not (has_img or ink_hit or ink_half):
                res["evidence"].append("objek vektor tidak menghasilkan tinta pada render: tidak dihitung sebagai tanda tangan")
            strong = has_img or has_vec
            res["confidence"] = ("tinggi" if strong and ink_hit else "sedang" if res["detected"] else None)
    except Exception as e:
        res["error"] = str(e)
    return res


# --------------------------------------------------------------------------------------
# API & CLI
# --------------------------------------------------------------------------------------

#   Dokumen gabungan (beberapa RAB dalam satu PDF)
#     Awal sebuah RAB = baris judul ('RINCIAN ANGGARAN BIAYA/BELANJA', 'RIN ARAN' bila judul rusak)
#     di awal halaman DAN blok identitas (Kementerian/Lembaga + Satker/Unit Eselon) tepat sesudahnya.
#     Header yang diulang di tiap halaman RAB yang sama tidak dianggap RAB baru: itu dikenali dari
#     identitas (satker + alokasi) yang sama dan RAB sebelumnya belum 'ditutup' (JUMLAH TOTAL / tanggal TTD).
#     Halaman sebelum RAB pertama (mis. 'LIST RAB ...') dipisah sebagai 'pengantar'.

TITLE_RE = re.compile(r"RINCIAN\s+(?:ANGGARAN|BELANJA)|RIN\s+ARAN", re.I)
IDENT_K_RE = re.compile(r"Kementerian|Lembaga", re.I)
IDENT_S_RE = re.compile(r"Satker|Sakter|Unit\s+Eselon", re.I)


def _ident_key(lines: list[Line], i: int) -> tuple:
    win = [clean_text(x.text).lower() for x in lines[i:i + 14]]
    return tuple(t for t in win if IDENT_S_RE.search(t) or t.startswith("alokasi"))


def _is_closed(seg: list[Line]) -> bool:
    for x in seg:
        if re.sub(r"\s+", "", x.text).upper().startswith("JUMLAHTOTAL") or TTD_RE.match(clean_text(x.text)):
            return True
    return False


def find_rab_starts(lines: list[Line]) -> list[int]:
    """Cari awal setiap RAB tanpa merusak parser hierarki.

    Prinsipnya page-aware: RAB gabungan pada dokumen ini dimulai pada halaman baru,
    sedangkan halaman lanjutan dapat mengulang header tabel. Header tabel saja TIDAK
    boleh dianggap sebagai RAB baru. Sinyal utama adalah judul + blok metadata RAB
    di bagian atas halaman.
    """
    page_first: dict[int, int] = {}
    page_lines: dict[int, list[tuple[int, Line]]] = {}
    for i, l in enumerate(lines):
        page_first.setdefault(l.page, i)
        page_lines.setdefault(l.page, []).append((i, l))

    starts: list[int] = []
    for page in sorted(page_lines):
        entries = page_lines[page]
        # Hanya bagian atas halaman yang diperiksa. Ini penting agar header tabel
        # yang muncul di tengah halaman lanjutan tidak membuka RAB baru.
        top = entries[:18]
        texts = [clean_text(l.text) for _, l in top]
        title_pos = next((j for j, t in enumerate(texts) if TITLE_RE.search(t)), None)
        if title_pos is None:
            continue

        # Metadata khas halaman pertama RAB. Variasi label antar template diterima.
        meta_window = texts[title_pos:title_pos + 16]
        has_kementerian = any(IDENT_K_RE.search(t) for t in meta_window)
        has_satker = any(IDENT_S_RE.search(t) for t in meta_window)
        has_alloc = any(re.search(r"Alokasi\s+Dana", t, re.I) for t in meta_window)
        has_volume = any(re.search(r"\bVolume\b", t, re.I) for t in meta_window)
        has_satuan = any(re.search(r"Satuan\s+Ukur", t, re.I) for t in meta_window)

        # Judul harus berada dekat bagian atas. Untuk template yang sangat berantakan
        # cukup 2 sinyal metadata; template normal diharapkan punya >=3.
        score = sum([has_kementerian, has_satker, has_alloc, has_volume, has_satuan])
        if title_pos <= 3 and score >= 3:
            starts.append(entries[title_pos][0])

    # Defensive fallback untuk template lama: bila page-aware terlalu ketat, gunakan
    # logika lama tetapi tetap hanya menerima judul di awal halaman.
    if not starts:
        first_idx = {p: min(i for i, l in enumerate(lines) if l.page == p) for p in page_lines}
        cand = []
        for i, l in enumerate(lines):
            if not TITLE_RE.search(l.text):
                continue
            if i - first_idx[l.page] > 3:
                continue
            win = [x.text for x in lines[i:i + 16]]
            if any(IDENT_K_RE.search(t) for t in win) and any(IDENT_S_RE.search(t) for t in win):
                cand.append(i)
        starts = cand

    return sorted(dict.fromkeys(starts))


AMT_RE = re.compile(r"\d{1,3}(?:\.\d{3})+")


def _merge_amount_tokens(tokens: list[str]) -> list[str]:
    """Normalisasi token angka lama.

    Penggabungan berbasis isi sengaja tidak lagi dilakukan di sini karena tidak tahu
    posisi kolom. Pemecahan token PDF sekarang ditangani di ``load_pdf_lines``
    menggunakan koordinat x. Fungsi ini dipertahankan untuk kompatibilitas pemanggil lama.
    """
    return list(tokens)


_DAFTAR_KOLOM = ("pagu_belanja_barang", "pagu_belanja_pegawai", "total_pagu",
                 "rasional_belanja_barang", "rasional_belanja_pegawai", "total_rasional", "kekurangan")


def parse_daftar(cover: list[Line]) -> Optional[dict]:
    """
    Tabel 'LIST RAB' pada halaman pengantar: satu baris per satker (No, Satker, 7 kolom nominal).
    Nama kolom hanya dipasang bila aritmetikanya cocok (barang + pegawai = total; rasional barang +
    pegawai = total rasional; total rasional - total pagu = kekurangan), jadi tidak sekadar ditebak.
    """
    rows, total = [], None
    for l in cover:
        toks = l.text.split()
        if not toks:
            continue
        is_total = toks[0].lower() == "total"
        if not (is_total or re.fullmatch(r"\d{1,3}", toks[0])):
            continue
        body = _merge_amount_tokens(toks[1:])
        amts = [parse_num(t) for t in body if AMT_RE.fullmatch(t)]
        if len(amts) < 3:
            continue
        row: dict[str, Any] = {"no": None if is_total else int(toks[0]),
                               "satker": " ".join(t for t in body if not AMT_RE.fullmatch(t)) or None,
                               "angka": amts}
        if (len(amts) == 7 and amts[0] + amts[1] == amts[2] and amts[3] + amts[4] == amts[5]
                and amts[5] - amts[2] == amts[6]):
            row.update(dict(zip(_DAFTAR_KOLOM, amts)))
        if is_total:
            total = row
        else:
            rows.append(row)
    return {"baris": rows, "total": total} if len(rows) >= 2 else None


def match_daftar(res: dict) -> None:
    """Cocokkan tiap RAB dengan baris daftar pengantar (alokasi = total pagu). Hasil: res['rab'][i]['daftar']."""
    daftar = (res.get("pengantar") or {}).get("daftar")
    if not daftar:
        return
    rows = [r for r in daftar["baris"] if r.get("total_pagu") is not None]
    used: set[int] = set()
    todo = []
    for r in res["rab"]:
        ac = r["total"].get("alokasi_dana")
        hit = [x for x in rows if x["total_pagu"] == ac and x["no"] not in used]
        if len(hit) == 1:
            used.add(hit[0]["no"])
            r["daftar"] = {"no": hit[0]["no"], "satker": hit[0]["satker"], "kecocokan": "persis",
                           "total_pagu": hit[0]["total_pagu"], "total_rasional": hit[0]["total_rasional"]}
        else:
            todo.append(r)
    for r in todo:                                  # sisa: terdekat dalam 5% -> 'perkiraan'
        ac = r["total"].get("alokasi_dana") or 0
        left = [x for x in rows if x["no"] not in used and ac and abs(x["total_pagu"] - ac) / ac <= 0.05]
        if left:
            x = min(left, key=lambda z: abs(z["total_pagu"] - ac))
            used.add(x["no"])
            r["daftar"] = {"no": x["no"], "satker": x["satker"], "kecocokan": "perkiraan",
                           "total_pagu": x["total_pagu"], "total_rasional": x["total_rasional"]}
    sisa = [x for x in rows if x["no"] not in used]
    daftar["tanpa_rab"] = [{"no": x["no"], "satker": x["satker"], "total_pagu": x["total_pagu"]} for x in sisa]
    for x in sisa:
        res["peringatan"].append(
            f"Daftar pengantar No. {x['no']} ({x['satker']}, total pagu {fmt_idr(x['total_pagu'])}) "
            "tidak punya RAB di dokumen ini")
    for r in res["rab"]:
        if not r.get("daftar"):
            res["peringatan"].append(
                f"RAB {r['no_rab']} (alokasi {fmt_idr(r['total'].get('alokasi_dana'))}) "
                "tidak cocok dengan baris mana pun di daftar pengantar")


_NODE_FIRST = (KOMP_RE, SUBCODE_RE, SUB_RE, AKUN_RE)


def _is_node_row(l: Line, code_x: float) -> bool:
    toks = l.text.split()
    if not toks or l.x0 is None or l.x0 >= code_x:
        return False
    f = _strip_code_prefix(toks)[0]
    return any(rx.match(f) for rx in _NODE_FIRST)


def _is_text_only(l: Line, code_x: float) -> bool:
    """Baris teks murni di kolom uraian: tidak ada angka, bukan header/noise/placeholder/meta."""
    if l.x0 is None or l.x0 < code_x or l.top is None:
        return False
    toks = l.text.split()
    if not toks or any(is_num(t) or NUMISH.fullmatch(t) for t in toks):
        return False
    if is_header_row(l.text) or any(rx.search(l.text) for rx in NOISE_RES):
        return False
    if MASK_ANY_RE.search(l.text) or any(rx.match(l.text) for _, rx, _ in META_RES):
        return False
    return True


def _adjacent(a: Line, b: Line) -> bool:
    """b persis di bawah a (halaman sama, jarak <= ~2 tinggi huruf)."""
    if a.page != b.page or a.top is None or b.top is None:
        return False
    h = max(6.0, (a.bottom or a.top) - a.top)
    return 0 < b.top - a.top <= 2.0 * h


def attach_centered_titles(lines: list[Line]) -> list[Line]:
    """
    Sel Uraian Komponen/Subkomponen/Akun sering rata-TENGAH secara vertikal: baris kode ('A  285.924.000')
    berada di tengah sel, sehingga judul panjang yang membungkus terbelah menjadi baris DI ATAS dan
    DI BAWAH baris kode. Tanpa penanganan, baris atas ikut menempel ke node sebelumnya.

    Baris teks di atas baris kode dipindahkan ke node tersebut bila ada bukti kuat:
      R1  baris kode tidak punya judul sendiri (hanya kode + nominal)  -> judul PASTI di sekitarnya;
      R2  baris kode punya judul, jumlah baris di atas == di bawah (simetris, sel rata-tengah) DAN
          baris tepat di atas 'penuh' (kata pertama baris kode tidak muat lagi di kolom uraian ->
          kalimatnya memang lanjut ke baris kode).
    Hanya PDF (butuh geometri); kasus lain tidak disentuh.
    """
    geo = [l for l in lines if l.src == "pdf" and l.top is not None]
    if not geo or len(geo) < 0.8 * len(lines):
        return lines
    akun_x = sorted(l.x0 for l in geo if AKUN_RE.match(_strip_code_prefix(l.text.split())[0]))
    if not akun_x:
        return lines
    code_x = akun_x[len(akun_x) // 2] + 40.0          # kolom Kode berakhir sekitar 40 pt setelah kode akun
    items_x = sorted(l.x0 for l in geo if l.x0 >= code_x)
    if not items_x:
        return lines
    uraian_x = items_x[len(items_x) // 10]            # x awal kolom uraian (persentil rendah, kebal outlier)
    code_x = min(code_x, uraian_x - 4.0) if uraian_x - 4.0 > akun_x[0] else code_x
    # batas kanan kolom uraian: hanya dari baris teks yang SEJAJAR kolom uraian (blok tanda tangan/footer
    # berada di x lain dan akan mencemari batas ini)
    texts = [l for l in geo if _is_text_only(l, code_x) and abs(l.x0 - uraian_x) <= 3.0]
    col_right = max((l.x1 for l in texts if l.x1), default=None)

    drop: set[int] = set()
    for i, node in enumerate(lines):
        if i in drop or not _is_node_row(node, code_x):
            continue
        toks = _strip_code_prefix(node.text.split())
        title_toks = [t for t in toks[1:] if not (is_num(t) or NUMISH.fullmatch(t))]
        empty_title = not any(re.search(r"[A-Za-z]", t) for t in title_toks)

        pre = []
        j = i - 1
        nxt = node
        while j >= 0 and j not in drop and _is_text_only(lines[j], code_x) and _adjacent(lines[j], nxt):
            pre.insert(0, j)
            nxt = lines[j]
            j -= 1
        if not pre:
            continue
        post = 0
        j, prv = i + 1, node
        while j < len(lines) and _is_text_only(lines[j], code_x) and _adjacent(prv, lines[j]):
            post += 1
            prv = lines[j]
            j += 1

        take = 0
        if empty_title:
            take = len(pre) if post == 0 else min(len(pre), post + 1)
        elif post >= 1:
            last = lines[pre[-1]]
            first_word = title_toks[0] if title_toks else ""
            full = True
            if col_right and last.x1 and last.text:
                cw = (last.x1 - last.x0) / max(1, len(last.text))      # lebar rata-rata huruf
                full = last.x1 + (len(first_word) + 1) * cw > col_right
            if full:
                take = min(len(pre), post)
        if take:
            chosen = pre[-take:]
            node.title_pre = clean_text(" ".join(lines[k].text for k in chosen))
            drop.update(chosen)
    return [l for k, l in enumerate(lines) if k not in drop] if drop else lines


def parse_one(lines: list[Line], source: str = "") -> dict:
    """Urai SATU RAB. Gaya angka (titik/koma ribuan) dideteksi dari isi RAB itu sendiri."""
    set_number_style(detect_number_style(l.text for l in lines))
    set_content_join(any(l.src == "text" for l in lines))
    lines = attach_centered_titles(lines)
    try:
        p = RabParser(source)
        for ln in lines:
            p.feed(ln)
        return p.finish()
    finally:
        set_number_style(False)
        set_content_join(True)


def _rab_label(res: dict) -> str:
    m = res.get("metadata", {})
    bits = [m.get("unit_eselon2_satker"), m.get("kegiatan")]
    return " · ".join(str(b) for b in bits if b)


def _ringkas(res: dict) -> dict:
    t = res["total"]
    sg = res.get("penandatangan", {}).get("tanda_tangan")
    return {
        "no_rab": res["no_rab"], "halaman_awal": res["halaman_awal"], "halaman_akhir": res["halaman_akhir"],
        "satker": res["metadata"].get("unit_eselon2_satker"),
        "kegiatan": res["metadata"].get("kegiatan"),
        "kro": res["metadata"].get("kro"),
        "alokasi_dana": t.get("alokasi_dana"),
        "jumlah_total_tercantum": t.get("jumlah_total_tercantum"),
        "jumlah_item": len(res["items"]),
        "peringatan": len(res["peringatan"]),
        "tanda_tangan": (None if sg is None else bool(sg.get("detected"))),
        "tanda_tangan_elektronik": (res.get("penandatangan", {}).get("tanda_tangan_elektronik") or {}).get("ada_pernyataan"),
        "daftar_no": (res.get("daftar") or {}).get("no"),
        "daftar_kecocokan": (res.get("daftar") or {}).get("kecocokan"),
    }


def parse_lines(lines: list[Line], source: str = "", split: bool = True) -> dict:
    starts = find_rab_starts(lines) if split else []
    if len(starts) < 2:                       # RAB tunggal: keluaran sama seperti versi sebelumnya
        return parse_one(lines, source)

    cover = lines[:starts[0]]
    bounds = starts + [len(lines)]
    rabs = []
    for no, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
        seg = lines[a:b]
        res = parse_one(seg, source)
        res["no_rab"] = no
        res["halaman_awal"], res["halaman_akhir"] = seg[0].page, seg[-1].page
        res["judul"] = f"RAB {no}"
        per_page: dict[int, int] = {}
        for l in seg:
            per_page[l.page] = per_page.get(l.page, 0) + 1
        gap = [p for p in range(seg[0].page, seg[-1].page + 1) if p not in per_page]
        if gap:
            res["peringatan"].append(
                f"Halaman {', '.join(map(str, gap))} tidak mengandung teks (kosong/gambar/hasil scan)")
        sparse = [p for p, n in per_page.items() if n <= 4 and p != seg[0].page]
        if sparse:
            res["peringatan"].append(
                f"Halaman {', '.join(map(str, sparse))} hanya memuat <= 4 baris teks: kemungkinan sisa kolom "
                "yang terpotong dari cetakan spreadsheet; angka pada halaman itu tidak ikut terhitung")
        rabs.append(res)

    out: dict[str, Any] = {
        "source_file": source,
        "parser_version": PARSER_VERSION,
        "multi_rab": True,
        "jumlah_rab": len(rabs),
    }
    expected = None
    if cover:
        out["pengantar"] = {
            "halaman_awal": cover[0].page, "halaman_akhir": cover[-1].page,
            "judul": cover[0].text,
            "baris": [{"halaman": l.page, "teks": l.text} for l in cover],
        }
        daftar = parse_daftar(cover)
        if daftar:
            out["pengantar"]["daftar"] = daftar
            expected = len(daftar.get("baris", [])) or None

    out["rab"] = rabs
    out["peringatan"] = [f"RAB {r['no_rab']}: {w}" for r in rabs for w in r["peringatan"]]
    if expected is not None:
        out["jumlah_rab_di_daftar"] = expected
        if expected != len(rabs):
            out["peringatan"].append(
                f"Daftar pengantar memuat {expected} RAB/satker, tetapi hanya {len(rabs)} "
                "blok RAB detail yang terdeteksi. RAB yang tidak memiliki blok detail tidak dibuat-buat."
            )
    return out


# --------------------------------------------------------------------------------------
# Tanda tangan elektronik (TTE): PERNYATAAN di teks dokumen, terpisah dari tanda tangan visual.
# 'Dokumen ini telah ditandatangani secara elektronik menggunakan sertifikat elektronik yang diterbitkan BSrE'.
# Ini hanya bukti bahwa dokumen MENGKLAIM bertanda tangan elektronik; keabsahan sertifikat tidak diverifikasi.
# Kalimat UU ITE ('Informasi Elektronik dan/atau Dokumen Elektronik ... alat bukti hukum yang sah') bukan TTE.
# --------------------------------------------------------------------------------------
TTE_RE = re.compile(
    r"(?:telah\s+|sudah\s+)?ditandatangani\s+secara\s+(?:elektronik|digital)"
    r"|tanda\s*tangan\s+(?:elektronik|digital)"
    r"|ditandatangani\s+(?:dengan|menggunakan)\s+sertifikat\s+elektronik"
    r"|(?:dengan|menggunakan)\s+sertifikat\s+elektronik", re.I)
TTE_PENERBIT_RE = re.compile(r"BSrE|Balai\s+Sertifikasi\s+Elektronik|\bBSSN\b|\bPrivy\b|\bPERURI\b|\bVIDA\b", re.I)
TTE_NEG_RE = re.compile(r"\b(?:belum|tidak|bukan)\s+(?:\w+\s+){0,2}$", re.I)       # 'belum ditandatangani secara ...'


def detect_tte(pdf_path: str | Path, pages: Optional[tuple[int, int]] = None) -> dict:
    """-> {ada_pernyataan, penerbit, halaman[], kalimat, jumlah_halaman_bertulis}. Tidak menilai keabsahan."""
    import pdfplumber
    out: dict[str, Any] = {"ada_pernyataan": False, "penerbit": None, "halaman": [], "kalimat": None}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            lo, hi = pages if pages else (1, len(pdf.pages))
            for pno in range(lo, min(hi, len(pdf.pages)) + 1):
                txt = clean_text(pdf.pages[pno - 1].extract_text(x_tolerance=1.2) or "")
                for m in TTE_RE.finditer(txt):
                    if TTE_NEG_RE.search(txt[max(0, m.start() - 25):m.start()]):
                        continue
                    out["ada_pernyataan"] = True
                    out["halaman"].append(pno)
                    if out["kalimat"] is None:
                        a = max(txt.rfind("-", 0, m.start()), txt.rfind(". ", 0, m.start()), -1) + 1
                        out["kalimat"] = clean_text(txt[a:m.end() + 110])[:200]
                    pm = TTE_PENERBIT_RE.search(txt[m.start():m.end() + 160])
                    if pm and not out["penerbit"]:
                        out["penerbit"] = pm.group(0)
                    break
    except Exception as e:
        out["error"] = str(e)
    return out


def _attach_signature(res: dict, sig: dict, tte: Optional[dict] = None) -> None:
    sig = dict(sig)
    if tte is not None:
        res["penandatangan"]["tanda_tangan_elektronik"] = tte
    nip = sig.pop("nip", None)
    blok = sig.pop("blok_teks", None)
    ttd = res["penandatangan"]
    if blok:
        # posisi di blok tanda tangan menentukan peran: atas = jabatan, bawah (setelah ruang ttd) = nama
        ttd["jabatan"], ttd["nama"] = blok.get("jabatan"), blok.get("nama")
        ttd.pop("pejabat", None)
        ttd["jabatan_nama_sumber"] = "posisi_blok"
        if blok.get("pengantar"):
            ttd["pengantar"] = blok["pengantar"]
        for k in ("jabatan", "nama"):
            if ttd[k] is None:
                del ttd[k]
            elif ttd.get("masked_lain") and ttd[k] in ttd["masked_lain"]:
                ttd["masked_lain"].remove(ttd[k])
        if "masked_lain" in ttd and not ttd["masked_lain"]:
            del ttd["masked_lain"]
    ttd["tanda_tangan"] = sig
    if nip and nip.get("ada_label"):
        old = ttd.get("nip") or {}
        new = {k: v for k, v in nip.items() if v is not None}
        if old.get("nilai_status") in ("tersamar", "terbaca_valid", "terbaca_tidak_standar") \
                and new.get("nilai_status") in (None, "kosong"):
            new.pop("nilai_status", None)
            new.pop("jumlah_digit", None)
        if old.get("nilai"):
            new.pop("nilai", None)                      # nilai dari teks dokumen didahulukan
        ttd["nip"] = {**old, **new}
    elif nip is not None and "nip" not in ttd:
        ttd["nip"] = nip                      # ada_label False + bukti pencarian
    if sig.get("nama_ada") and "nama" not in ttd and not blok:
        ttd["nama"] = sig.get("nama_teks")


def parse_rab(path: str | Path, detect_sig: bool = True, sig_debug_dir=None, split: bool = True) -> dict:
    path = Path(path)
    res = parse_lines(load_lines(path), path.name, split=split)
    is_pdf = path.suffix.lower() == ".pdf"
    if res.get("multi_rab"):
        for r in res["rab"]:
            if detect_sig and is_pdf:
                _pg = (r["halaman_awal"], r["halaman_akhir"])
                _attach_signature(r, detect_signature(path, debug_dir=sig_debug_dir, pages=_pg),
                                  detect_tte(path, pages=_pg))
        match_daftar(res)
        res["ringkasan"] = [_ringkas(r) for r in res["rab"]]
    elif detect_sig and is_pdf:
        _attach_signature(res, detect_signature(path, debug_dir=sig_debug_dir), detect_tte(path))
    return res


# --------------------------------------------------------------------------------------
# JSON ringkas (default). Hanya mengubah SERIALISASI; hasil parse internal (CSV, ringkasan) tetap penuh.
#   - 'items' datar dibuang (isinya sudah ada di pohon 'struktur'; pakai --flat bila perlu)
#   - nilai kosong (null / [] / {}) tidak ditulis
#   - field turunan/duplikat dibuang: faktor, halaman per item, jenis_belanja, *_sumber bernilai default
# --------------------------------------------------------------------------------------
_ITEM_DROP = {"faktor", "halaman"}


def _prune(o):
    if isinstance(o, dict):
        d = {k: _prune(v) for k, v in o.items()}
        return {k: v for k, v in d.items() if v is not None and v != [] and v != {} and v != ""}
    if isinstance(o, list):
        return [_prune(x) for x in o]
    return o


def _compact_item(it: dict) -> dict:
    d = {k: v for k, v in it.items() if k not in _ITEM_DROP}
    if d.get("volume_sumber") == "rincian":
        d.pop("volume_sumber")
    if d.get("jumlah_sumber") == "pdf":
        d.pop("jumlah_sumber")
    return d


def _compact_node(n: dict) -> dict:
    d = {k: v for k, v in n.items() if k not in ("items", "children", "halaman", "jenis_belanja")}
    if n.get("items"):
        d["items"] = [_compact_item(i) for i in n["items"]]
    if n.get("children"):
        d["children"] = [_compact_node(c) for c in n["children"]]
    return d


def _find_ro(nodes):
    for n in nodes or []:
        if n.get("tipe") == "ro":
            return f"{n.get('kode') or ''} {n.get('uraian') or ''}".strip()
        sub = _find_ro(n.get("children"))
        if sub:
            return sub
    return None


_MASKED_TEXT_RE = re.compile(r"MASK|MMAASSKK|^\[\s*MA?\b|\[\s*MA\s*$|^\[?\s*MASK", re.I)   # MMAASSKK: huruf terdobel


def _is_masked_text(v) -> bool:
    """'[MASKED DATA PEJABAT]', '[MA [MASKED DATA PEJABAT]', '[MA' (fragmen tertimpa), '*****', 'XXXX'."""
    v = str(v or "").strip()
    return bool(v) and bool(_MASKED_TEXT_RE.search(v) or MASK_ANY_RE.search(v))


def _compact_meta(r: dict, source: Optional[str] = None) -> dict:
    """Satu blok metadata yang memuat semua butir checklist format (III.A-III.L) + tahun + nama file."""
    m, t = r.get("metadata", {}) or {}, r.get("penandatangan", {}) or {}
    sg, nip = t.get("tanda_tangan") or {}, t.get("nip") or {}
    nama, jab = t.get("nama") or t.get("pejabat"), t.get("jabatan")
    nilai = nip.get("nilai")
    nip_mask = nip.get("nilai_status") == "tersamar" or "MASK" in str(nip.get("teks_baris") or nilai or "").upper()
    nip_ada = bool(nip.get("ada_label")) and (bool(nilai) or nip.get("nilai_status") not in (None, "kosong") or nip_mask)
    nip_digits = re.sub(r"\D", "", nilai or "") if nilai and not nip_mask else ""
    masked = set(m.get("field_tersamar") or []) - {"judul_unit"}
    ada_ttd = bool((sg or {}).get("detected")) or bool((t.get("tanda_tangan_elektronik") or {}).get("ada_pernyataan"))
    # Nama/jabatan ditulis apa adanya (placeholder mask seperti '[MASKED DATA PEJABAT]' juga), dan
    # dicatat di field_tersamar bila tersamar.
    if not jab and ada_ttd and t.get("masked_lain") and t.get("jabatan_nama_sumber") != "posisi_blok":
        jab = t["masked_lain"][0]                             # blok tanda tangan memuat teks tersamar tanpa label jabatan
    for key, val in (("nama_penanggung_jawab", nama), ("jabatan_penanggung_jawab", jab)):
        if val and _is_masked_text(val):
            masked.add(key)
    if nip_ada and nip_mask:
        masked.add("nip")
    out = {
        "nama_file": r.get("source_file") or source,
        "tahun_anggaran": m.get("tahun_anggaran"),
        "kementerian": m.get("kementerian"),
        "unit_eselon2_satker": m.get("unit_eselon2_satker"),
        "program": m.get("program"),
        "kegiatan": m.get("kegiatan"),
        "sasaran_kegiatan": m.get("sasaran_kegiatan"),
        "indikator_kinerja_kegiatan": m.get("indikator_kinerja_kegiatan"),
        "kro": m.get("kro"),
        "kode_header": m.get("kode_header"),
        "komponen": m.get("komponen"),
        "ro": m.get("ro"),                       # header saja; tanpa fallback ke tabel
        "kro_tabel": m.get("kro_tabel"),
        "ro_tabel": m.get("ro_tabel"),
        "volume": m.get("volume"),
        "volume_teks": m.get("volume_teks"),
        "satuan_ukur": m.get("satuan_ukur"),
        "alokasi_dana": m.get("alokasi_dana"),
        "penanggung_jawab": {
            "nama": nama, "jabatan": jab,
            "tanda_tangan": sg.get("detected") if sg else None,     # None = deteksi tidak dijalankan
            "tanda_tangan_keyakinan": sg.get("confidence") if sg else None,   # tinggi | sedang (ink saja -> cek manual)
            "tanda_tangan_elektronik": (t.get("tanda_tangan_elektronik") or {}).get("ada_pernyataan")
                                       if t.get("tanda_tangan_elektronik") is not None else None,   # pernyataan TTE di teks
            "tte_penerbit": (t.get("tanda_tangan_elektronik") or {}).get("penerbit"),
            "nip": nilai if nilai else (nip.get("teks_baris") if nip_mask else None),   # nomor, atau teks tersamar
            "nip_ada": nip_ada if nip else False,
            # NIP baru = 18 digit; NIP tersamar tetap dianggap valid (isinya tidak bisa diperiksa)
            "nip_valid": True if (nip_ada and nip_mask) else ((len(nip_digits) == 18) if nip_digits else None),
        },
        "jumlah_total_tercantum": (r.get("total") or {}).get("jumlah_total_tercantum"),
        "field_tersamar": sorted(masked),
    }
    return out


def compact_rab(r: dict, flat: bool = False, source: Optional[str] = None) -> dict:
    out = {"parser_version": r.get("parser_version"), "metadata": _compact_meta(r, source)}
    if "struktur" in r:
        out["struktur"] = [_compact_node(n) for n in r["struktur"]]
    if flat and "items" in r:
        out["items"] = [_compact_item(i) for i in r["items"]]
    out = _prune(out)
    out["peringatan"] = r.get("peringatan", [])          # kunci selalu ada, walau kosong
    return out


def compact_result(res: dict, flat: bool = False) -> dict:
    if res.get("multi_rab"):
        out = {k: v for k, v in res.items() if k != "rab"}
        out["rab"] = [compact_rab(r, flat, f"{res.get('source_file')}#RAB{r.get('no_rab')}") for r in res["rab"]]
        return _prune(out)
    return compact_rab(res, flat)


_EXTRA_KEYS = ("no_rab", "judul", "halaman_awal", "halaman_akhir", "daftar")


def to_single_schema(r: dict, source: str, jumlah_rab: int) -> dict:
    """Satu RAB hasil dokumen gabungan -> struktur JSON IDENTIK dengan RAB tunggal (mis. rab0353.json).
    Info posisi dalam dokumen gabungan dipindah ke metadata.dokumen_gabungan."""
    one = {k: v for k, v in r.items() if k not in _EXTRA_KEYS}
    one["source_file"] = f"{source}#RAB{r['no_rab']}"
    meta = dict(one.get("metadata", {}))
    meta["dokumen_gabungan"] = {"file": source, "no_rab": r["no_rab"], "dari_total_rab": jumlah_rab,
                                "halaman_awal": r["halaman_awal"], "halaman_akhir": r["halaman_akhir"]}
    if r.get("daftar"):
        meta["dokumen_gabungan"]["daftar_pengantar"] = r["daftar"]
    one["metadata"] = meta
    order = ["source_file", "parser_version", "metadata", "penandatangan", "total",
             "struktur", "items", "peringatan"]
    return {k: one[k] for k in order if k in one}


def write_split(res: dict, out_dir: Path, stem: str, full: bool = False, flat: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if not res.get("multi_rab"):
        pth = out_dir / f"{stem}.json"
        pth.write_text(json.dumps(res if full else compact_result(res, flat), ensure_ascii=False,
                                  indent=2), encoding="utf-8")
        return [pth]
    for r in res["rab"]:
        pth = out_dir / f"{stem}_rab{r['no_rab']:02d}.json"
        one = to_single_schema(r, res["source_file"], res["jumlah_rab"])
        pth.write_text(json.dumps(one if full else compact_rab(one, flat), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        paths.append(pth)
    idx = {k: v for k, v in res.items() if k != "rab"}
    pth = out_dir / f"{stem}_ringkasan.json"
    pth.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths + [pth]


def write_csv(result: dict, out: Path):
    cols = ["no", "kro", "ro", "kro_tabel", "ro_tabel", "komponen", "subkomponen", "akun_kode", "akun_uraian", "kelompok",
            "label", "uraian", "rincian_perhitungan", "volume", "satuan", "harga_satuan",
            "jumlah", "jumlah_sumber", "halaman"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        if result.get("multi_rab"):
            w = csv.DictWriter(f, fieldnames=["rab"] + cols, extrasaction="ignore")
            w.writeheader()
            for r in result["rab"]:
                for it in r["items"]:
                    w.writerow({"rab": r["no_rab"], **it})
            return
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in result["items"]:
            w.writerow(it)


def summarize(res: dict) -> str:
    if res.get("multi_rab"):
        s = [f"Versi parser: {PARSER_VERSION}", f"File        : {res['source_file']}",
             f"Jumlah RAB  : {res['jumlah_rab']}"
             + (f" (+ pengantar hal {res['pengantar']['halaman_awal']}-{res['pengantar']['halaman_akhir']})"
                if res.get("pengantar") else "")]
        for r in res["ringkasan"]:
            s.append(f"  RAB {r['no_rab']:>2} hal {r['halaman_awal']}-{r['halaman_akhir']:<3} "
                     f"item={r['jumlah_item']:<4} alokasi={fmt_idr(r['alokasi_dana'])} "
                     f"total={fmt_idr(r['jumlah_total_tercantum'])}")
        return "\n".join(s)
    t = res["total"]
    s = [f"Versi parser: {PARSER_VERSION}",
         f"File        : {res['source_file']}",
         f"Item        : {len(res['items'])}",
         f"Total (PDF) : {fmt_idr(t['jumlah_total_tercantum'])}",
         f"Peringatan  : {len(res['peringatan'])}"]
    sg = res.get("penandatangan", {}).get("tanda_tangan")
    if sg is not None:
        s.append("Tanda tangan: " + ("TERDETEKSI (" + str(sg.get("confidence")) + ")" if sg["detected"]
                                      else "tidak terdeteksi") + f" [hal {sg.get('halaman')}]")
        te = res.get("penandatangan", {}).get("tanda_tangan_elektronik")
        if te is not None:
            s.append("TTE (pernyataan): " + (f"ADA ({te.get('penerbit') or 'penerbit tidak disebut'}) [hal {','.join(map(str, te['halaman']))}]"
                                            if te["ada_pernyataan"] else "tidak ada"))
    return "\n".join(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Parser RAB -> JSON")
    ap.add_argument("inputs", nargs="+", help="file RAB (.pdf/.xlsx/.txt/.md) atau folder")
    ap.add_argument("-o", "--output", help="file JSON (satu input) atau folder (batch)")
    ap.add_argument("--csv", help="tulis item datar ke CSV (satu input)")
    ap.add_argument("--no-split", action="store_true",
                    help="jangan pisahkan dokumen gabungan; perlakukan semua halaman sebagai satu RAB")
    ap.add_argument("--split-output", action="store_true",
                    help="dokumen gabungan: tulis SATU JSON per RAB (skema identik rab0353.json) + _ringkasan.json; "
                         "-o = folder keluaran")
    ap.add_argument("--full", action="store_true",
                    help="tulis skema JSON lengkap lama (items datar + semua field); default: JSON ringkas")
    ap.add_argument("--flat", action="store_true",
                    help="JSON ringkas + sertakan 'items' datar (tanpa field duplikat)")
    ap.add_argument("--indent", type=int, default=2, help="indentasi JSON (0 = satu baris, paling kecil)")
    ap.add_argument("--no-signature", action="store_true", help="lewati deteksi tanda tangan")
    ap.add_argument("--sig-debug", help="folder untuk menyimpan potongan gambar area tanda tangan (PNG)")
    ap.add_argument("--debug-lines", action="store_true",
                    help="cetak baris hasil rekonstruksi (halaman, x0, teks) lalu keluar")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for i in args.inputs:
        p = Path(i)
        if p.is_dir():
            files += sorted(x for x in p.iterdir()
                            if x.suffix.lower() in (".pdf", ".xlsx", ".xlsm", ".txt", ".md"))
        else:
            files.append(p)

    for f in files:
        if args.debug_lines:
            for ln in load_lines(f):
                x = "" if ln.x0 is None else f"{ln.x0:7.1f}"
                print(f"p{ln.page} x0={x:>7} | {ln.text}")
            continue
        res = parse_rab(f, detect_sig=not args.no_signature, sig_debug_dir=args.sig_debug,
                        split=not args.no_split)
        print(summarize(res), "\n")
        if args.split_output:
            od = Path(args.output) if args.output else f.parent
            for pth in write_split(res, od, f.stem, full=args.full, flat=args.flat):
                print("  ->", pth)
        elif args.output:
            out = Path(args.output)
            if len(files) > 1 or out.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                out = out / (f.stem + ".json")
            obj = res if args.full else compact_result(res, args.flat)
            out.write_text(json.dumps(obj, ensure_ascii=False, indent=args.indent or None,
                                      separators=(",", ":") if not args.indent else None), encoding="utf-8")
        if args.csv and len(files) == 1:
            write_csv(res, Path(args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
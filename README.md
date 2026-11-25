# TIMA — Pemeriksaan TOR & RAB

Upload 1 TOR + 1..n RAB (PDF) → klasifikasi → ekstraksi → checklist → pasangan TOR–RAB → laporan + ZIP.

```
all-tima/
├── app/
│   ├── config.py        # semua path & endpoint dari env (tanpa path lokal)
│   ├── pipeline.py      # orkestrator run (juga CLI)
│   ├── report.py        # checklist_tor/rab.json+md, llm_tor.json, hasil_gabungan.json, ringkasan.html, ZIP
│   ├── vlm_client.py    # klien VLM jarak jauh (placeholder endpoint Qwen3-VL-2B)
│   └── api.py           # FastAPI
├── Checklist-Rule-TOR-RAB/
│   ├── checklist_common.py   # rules loader, chunk evidence, validator, laporan per dokumen
│   ├── checklist_tor.py      # Section I (I.A–I.T); I.M–I.R → endpoint SFT (placeholder)
│   ├── checklist_rab.py      # Section III + IV.B/IV.C/IV.D
│   ├── rules/  katalog_sbm/  wilayah_indonesia.csv
├── Ekstraktor-TOR-RAB/       # tor_extractor.py, ExtractorRAB.py (tidak diubah)
├── Dockerfile  docker-compose.yml  docker-entrypoint.sh  requirements.txt  .env.example
```

`Model-Checklist-Engine/` (adapter LoRA) **tidak** ikut ke image — dipakai di VPS model.

## Menjalankan di VPS

```bash
cp .env.example .env          # isi VLM_ENDPOINT_URL dan SFT_ENDPOINT_URL bila VPS model sudah siap
docker compose up -d --build
curl http://localhost:8000/health
```

```bash
# upload (field "files" boleh berulang)
curl -F "files=@TOR.pdf" -F "files=@RAB1.pdf" -F "files=@RAB2.pdf" http://localhost:8000/runs
# -> {"run_id": "...", "status_url": "/runs/<id>", "zip_url": "/runs/<id>/zip"}
curl http://localhost:8000/runs/<id>                 # status: menunggu | berjalan | selesai | gagal
curl -o hasil.zip http://localhost:8000/runs/<id>/zip
```

Hasil tersimpan di `./data/workspace/runs/<run_id>/` (volume).

## Menjalankan lokal (tanpa Docker)

```bash
pip install -r requirements.txt          # + Tesseract (ind, eng) terpasang di sistem
python -m app.pipeline TOR.pdf RAB1.pdf RAB2.pdf
uvicorn app.api:app --reload             # atau API
```
Di Windows, bila Tesseract tidak ada di PATH: `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.

Modul checklist tetap bisa dipakai sendiri:
```bash
cd Checklist-Rule-TOR-RAB
python checklist_tor.py <folder checklist_rule_based> -o hasil_cek
python checklist_rab.py <folder JSON RAB> -o hasil_cek --konsistensi
```

## Kontrak endpoint model (placeholder — sesuaikan dengan API di VPS model)

**VLM** (`VLM_API_STYLE=openai`, mis. vLLM `--model Qwen/Qwen3-VL-2B-Instruct`): POST chat completion berisi
`image_url` `data:image/png;base64,...` + prompt `DESC_PROMPT` ekstraktor. Jawaban harus berformat
`JENIS: … / TEKS: … / DESKRIPSI: …`. Gaya `custom`: body `{"model","prompt","image_b64","max_new_tokens"}`, respons `{"text"}`.
Bila endpoint tidak bisa dihubungi, gambar sisanya langsung dilewati (tidak menunggu timeout satu per satu).

**SFT** (Qwen3-4B-Instruct-2507 + LoRA): POST `SFT_ENDPOINT_URL` body
`{"task","model","jenis_dokumen","nama_file","kriteria_kode","pertanyaan","konteks":{kode_field: isi},"candidates":[{"chunk_id","quote"}],"output_schema"}`,
respons `{"status":"LOLOS|TIDAK|MANUAL","alasan","evidence_quote","evidence_chunk_id"}`.
`evidence_quote` harus identik dengan salah satu `candidates[].quote`; bila tidak, status diturunkan ke MANUAL.
Tanpa endpoint, I.M–I.R berstatus MANUAL.

## Variabel environment

Lihat `.env.example`. Path bawaan relatif terhadap root repo, jadi tidak perlu diisi di Docker.

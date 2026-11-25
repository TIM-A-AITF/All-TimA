---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-4B-Instruct-2507
- lora
- continued-pretraining
---

# CPT-Fondasi-Model

LoRA adapter hasil **Continued Pre-Training (CPT)** di atas `Qwen/Qwen3-4B-Instruct-2507`, digabungkan dari 5 domain teks bahasa Indonesia sekaligus: **kominfo, legal (regulasi), news, wiki, socmed**. Tujuannya membuat model lebih terbiasa dengan gaya bahasa dan istilah domain pemerintahan/regulasi, sebagai fondasi sebelum tahap SFT (lihat [Model-Checklist-Engine](https://github.com/TIM-A-AITF/Model-Checklist-Engine) untuk tahap selanjutnya).

> **Catatan:** adapter ini dilatih di atas base model yang **sudah instruct-tuned**, berbeda dari eksperimen CPT sebelumnya (fondasi bahasa Indonesia umum di atas base model mentah). Repo ini hanya menyimpan run gabungan terbaru; histori run per-domain (kominfo saja, legal saja) ada di riwayat kerja tim dan tidak disertakan di sini.

## Data Training

25.098.240 token, gabungan 5 domain, `max_seq_len=1024`, `lora_r=16`, `lr=5e-5`, 1468 steps. Dataset mentah tidak disertakan di repo ini (lihat README Checklist-TOR/Ekstraktor-TOR untuk sumber dokumen asli).

## Hasil — Perplexity Sebelum vs Sesudah CPT

Diukur di held-out set tiap domain (lihat `run_4b_instruct_merged.json`):

| Domain | PPL sebelum | PPL sesudah | Perubahan |
|---|---|---|---|
| Legal | 3,43 | 2,06 | **-40%** |
| Kominfo | 3,56 | 2,25 | **-37%** |
| News | 7,36 | 5,07 | -31% |
| Wiki | 11,95 | 8,50 | -29% |
| Socmed | 23,37 | 17,66 | -24% |

## Contoh Kualitatif (lihat `instr_before.json` / `instr_after.json`)

Beberapa prompt umum dan domain-spesifik dibandingkan sebelum/sesudah CPT. Contoh: definisi "RAB" sebelum CPT salah kepanjangan ("Rencana Anggaran Bangunan"), setelah CPT berubah menjadi istilah yang lebih tepat untuk pengadaan pemerintah ("Rincian Anggaran Biaya").

### Keterbatasan — penting dibaca sebelum presentasi

CPT terbukti **mengubah gaya bahasa dan istilah domain** secara konsisten (perplexity turun di semua domain). Namun ini **tidak menjamin akurasi faktual**: pada contoh "dasar hukum", model tetap mengutip nomor Undang-Undang yang berbeda sebelum dan sesudah CPT, dan keduanya belum diverifikasi benar. Artinya CPT baik untuk membuat model "berbicara" dalam gaya dan istilah yang tepat, tetapi sitasi hukum/angka yang dihasilkan tetap perlu diverifikasi manusia — bukan sesuatu yang boleh diklaim sudah terselesaikan oleh CPT saja.

## File di Repo Ini

- `cpt-qwen3-4b-instruct-batch-1-run.ipynb` — notebook training
- `adapter_cpt_4b_instruct_merged/` — adapter LoRA hasil CPT
- `run_4b_instruct_merged.json`, `log_history_4b_instruct_merged.json` — metrik training & evaluasi
- `instr_before.json`, `instr_after.json` — contoh kualitatif sebelum/sesudah CPT

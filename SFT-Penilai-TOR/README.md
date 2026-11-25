---
base_model: microsoft/Phi-4-mini-instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:microsoft/Phi-4-mini-instruct
- lora
- sft
- transformers
- trl
---

# SFT-Penilai-TOR

LoRA adapter untuk **Phi-4-mini-instruct**, dilatih dengan SFT (Supervised Fine-Tuning) sebagai penilai kelogisan unsur TOR pada kriteria checklist **I.M–I.R** (Dasar Hukum, Gambaran Umum, Latar Belakang, Metode Pelaksanaan, Tahapan dan Waktu Pelaksanaan, Strategi Pencapaian Keluaran, Penerima Manfaat, Kurun Waktu Pencapaian Keluaran, Biaya yang Diperlukan, Manajemen Risiko).

> Folder ini terpisah dari [Model-Checklist-Engine](https://github.com/TIM-A-AITF/Model-Checklist-Engine) (adapter Qwen3-4B yang sedang dipakai tim untuk integrasi Checklist-TOR). Repo ini adalah hasil eksperimen lanjutan dengan base model lebih ringan (Phi-4-mini) dan dataset yang divalidasi pada dokumen TOR asli dengan test set bebas kebocoran — disimpan terpisah agar tidak menimpa pekerjaan yang sudah terintegrasi, keputusan untuk menggantikan atau menjalankan paralel ada di tim.

## Training Data

Dataset "neo": 778 dokumen TOR sumber (55 grok, 7 chatgpt, 345 haiku5.5, 212 gemini, 159 dokumen asli). Tiap butir TOR dipasangkan versi asli (LOLOS) dengan versi yang dirusak secara terkontrol (REVISI) melalui 11 metode perturbasi (mis. `TUKAR_UNSUR`, `HAPUS_NOMINAL`, `HAPUS_REGULASI`, `HAPUS_WAKTU`, `KOSONG`, `SIMBOL`). Split: train 611 / val 110 / test 57 dokumen sumber (928 baris butir di test), dokumen test terpisah dari training untuk menghindari kebocoran.

## Training Procedure

Dilatih di Kaggle (GPU T4, fp16) menggunakan `transformers`, `trl`, `peft`. Notebook: `phi-4-mini-instruct result v2.ipynb`. LoRA r=16, target modules: `gate_up_proj`, `o_proj`, `qkv_proj`, `down_proj`.

## Evaluation

Dievaluasi pada test set bebas kebocoran (928 baris, 10 jenis butir).

| Metrik | Baseline (sebelum SFT) | Setelah SFT |
|---|---|---|
| Balanced accuracy (SEMUA) | 0,50 | 0,958 |
| Akurasi (SEMUA) | — | 0,960 |
| F1 "tidak logis" (SEMUA) | 0,00 | 0,940 |

Baseline (model dasar tanpa fine-tune) selalu menebak "logis" untuk semua butir (recall "tidak logis" = 0%), sehingga balanced accuracy-nya tepat 0,50 di setiap butir — lompatan ini murni kontribusi SFT.

Akurasi per butir (setelah SFT), lihat `test_metrics_per_butir.csv`:

| Butir | Akurasi | F1 "tidak logis" |
|---|---|---|
| Dasar Hukum | 1,000 | 1,000 |
| Penerima Manfaat | 0,995 | 0,991 |
| Kurun Waktu Pencapaian Keluaran | 0,982 | 0,972 |
| Manajemen Risiko | 0,981 | 0,964 |
| Biaya yang Diperlukan | 0,977 | 0,953 |
| Tahapan dan Waktu Pelaksanaan | 0,955 | 0,933 |
| Latar Belakang | 0,946 | 0,957 |
| Strategi Pencapaian Keluaran | 0,898 | 0,909 |
| Metode Pelaksanaan | 0,875 | 0,825 |
| Gambaran Umum | 0,839 | 0,800 |

File hasil lengkap: `test_metrics_per_butir.csv`, `test_metrics_per_butir_BASELINE.csv`, `test_metrics_per_metode.csv` (akurasi per metode perturbasi), `test_perplexity.csv`, `test_bertscore.csv` (kualitas alasan tekstual, BERTScore F1 rata-rata ~0,83).

### Keterbatasan

Label "tidak logis" pada data training/test dibuat melalui perturbasi terkontrol terhadap TOR asli (bukan kesalahan organik dari penulis TOR). Skor di atas mengukur kemampuan model mendeteksi pola kerusakan tersebut; validasi lanjutan pada kasus TOR bermasalah di dunia nyata (bukan hasil rusak buatan) masih diperlukan sebelum dipakai sebagai satu-satunya dasar keputusan.

### Framework versions

- PEFT 0.21.3
- TRL 1.18.0
- Transformers 5.23.0

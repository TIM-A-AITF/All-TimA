# Checklist-RAB

Notebook pemeriksaan kelengkapan dan validasi dokumen RAB. Pemeriksaan utama kriteria RAB Section III (berdasarkan key-value JSON), plus validasi tambahan Section IV (IV.B, IV.C, IV.D).

## Isi Repo

- `checklist_engine_RAB_city - Copy.ipynb` — notebook utama
- `auto_gazetteer_indonesia.csv` — data gazetteer (nama wilayah) untuk validasi
- `katalog_sbm/` — katalog Standar Biaya Masukan
- `rules/` — konfigurasi rules & referensi (termasuk SBM 2026/2027)

## Struktur Folder yang Diharapkan

```text
ROOT/
├── rules/
│   ├── rules_config.json
│   ├── KEP-331-2021_Lampiran_II_hal_tercetak_62-122.json
│   ├── sbm_2026.json
│   └── sbm_2027.json
├── input/
│   └── RAB/
│       └── *.json           # JSON hasil parsing RAB (lihat Ekstraktor-RAB)
├── checklist_rab.ipynb
└── hasil_cek/
    └── RAB/
        └── *.json / *.md
```

## Menjalankan

1. Letakkan JSON hasil parsing RAB di `input/RAB/`
2. Pastikan `rules/rules_config.json` tersedia
3. Jalankan notebook dari atas ke bawah

## Status

Dokumentasi ini masih versi awal.

## **Alur Sistem Pemeriksaan TOR dan RAB**

Sistem bekerja menggunakan Google Colab dengan alur ekstraksi dokumen, pemrosesan data, pemeriksaan checklist, hingga menghasilkan laporan otomatis.

**1. Input Dokumen**

Upload file PDF TOR dan satu atau lebih PDF RAB secara bersamaan.

**2. Identifikasi Dokumen**

Sistem mengenali jenis dokumen berdasarkan nama file dan judul halaman pertama.

Jalur TOR

**3A. Ekstraksi TOR**

- Ekstraksi teks dan tabel PDF.
- OCR untuk dokumen scan.
- Deteksi gambar dan tanda tangan.
- Deskripsi gambar menggunakan Qwen3-VL-2B.
- Menghasilkan JSON detail.

  **4A. Checklist TOR**

Pemeriksaan 25 kriteria menggunakan aturan dan model Qwen3-4B-Instruct-2507 + LoRA.

Jalur RAB

**3B. Ekstraksi RAB**

Menggunakan ExtractorRAB untuk mengambil struktur, metadata, dan rincian anggaran.

**4B. Checklist RAB**

Pemeriksaan menggunakan rule engine:

- III.A–III.L
- IV.B: BAS
- IV.C: SBM
- IV.D: Batas 30%

**5. Pengolahan Hasil Pemeriksaan**

Hasil checklist TOR dan RAB disusun menjadi status pemeriksaan, alasan, dan evidence sesuai hasil masing-masing engine.

**6. Output dan Laporan**

- `checklist_tor.json` dan `.md`
- `checklist_rab.json` dan `.md`
- `llm_tor.json` untuk Tim B
- `hasil_gabungan.json`
- `ringkasan.html`
- Seluruh hasil dikemas dalam satu file ZIP.

### **Ringkasan alur**

**Upload PDF → Identifikasi TOR/RAB → Ekstraksi masing-masing dokumen → Checklist Engine → Penggabungan hasil → Laporan dan ZIP.**

Catatan: Pemeriksaan kesesuaian RAB terhadap TOR (IV.A) belum dilakukan oleh sistem ini.

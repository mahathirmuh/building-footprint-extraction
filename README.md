# densepoly — Ekstraksi Poligon Bangunan pada Permukiman Padat

Final project Pengolahan Citra Digital, S2 Informatika ITS — Mahathir Muhammad (6025252008).

U-Net++ dengan prior morfologi celah, head orientasi berbasis mask, dan regularisasi poligon terpandu. Output: satu
poligon rapi per rumah (COCO JSON / GeoJSON), dilatih **tanpa anotasi poligon**.

| File | Isi |
|---|---|
| `CLAUDE.md` | konteks proyek, aturan kerja, keputusan metodologi, jebakan teknis (dibaca Claude Code tiap sesi) |
| `PLAN.md` | rencana Fase 0–10 dengan kriteria "Selesai jika" |
| `docs/METHOD.md` | spesifikasi algoritma: prior, label, model, loss, watershed, regularisasi, parameter |
| `docs/DATA.md` | unduhan, deduplikasi CrowdAI, protokol WHU, strata kepadatan |
| `docs/EXPERIMENTS.md` | skenario B0–FULL+Y, training, tuning, prioritas, kriteria keputusan |
| `docs/EVALUATION.md` | metrik, statistik, uji kewarasan |
| `docs/COLAB.md` | training di Google Colab dan pekerjaan CPU berat |
| `docs/progress.md` | log kerja per fase |
| `reference/` | kode demo tahap ETS dan prototipe uji spesifikasi (referensi saja) |

## Cara memulai dengan Claude Code

1. Buat repo GitHub privat `densepoly`, salin semua file paket ini ke root repo, lalu commit.
2. Buka Claude Code di folder repo. Claude Code otomatis membaca `CLAUDE.md`.
3. Kerjakan satu fase per sesi. Pakai `/clear` di antara fase supaya konteks tetap bersih; status tersimpan di
   `docs/progress.md` dan git.

Contoh prompt per fase:

| Fase | Prompt |
|---|---|
| 0 | `Baca CLAUDE.md dan PLAN.md. Kerjakan Fase 0 saja. Jelaskan rencanamu dulu sebelum menulis kode.` |
| 1 | `Lanjut Fase 1. Data CrowdAI sudah ada di data/crowdai sesuai docs/DATA.md §1. Mulai dari deduplikasi. Tanyakan protokol WHU ke saya sebelum menulis skrip WHU.` |
| 2 | `Lanjut Fase 2. Tunjukkan gambar QA label dan prior dulu sebelum menghitung prior untuk seluruh dataset.` |
| 3 | `Lanjut Fase 3. Prediksi HiSup ada di data/crowdai/hisup_release/. Jalankan uji kewarasan docs/EVALUATION.md §6.1, §6.2, §6.4, §6.6 dan tulis docs/sanity_report.md.` |
| 4 | `Lanjut Fase 4. Mulai dari tes regularisasi dan uji oracle (§6.3), baru baseline B0.` |
| 5 | `Lanjut Fase 5. Siapkan notebook Colab dan benchmark. Jangan mulai training panjang sebelum saya setuju dengan estimasi jam GPU.` |
| 6–9 | `Lanjut Fase <n> sesuai PLAN.md. Ringkas hasil dan keputusan di docs/progress.md.` |
| 10 | `Lanjut Fase 10: tabel, gambar, dan notebooks/demo_hasil.ipynb. Pastikan notebook lolos Restart & Run all, lalu ekspor ke HTML.` |

Tips:
- Kalau Claude Code menyimpang dari spesifikasi, minta dia menunjuk bagian `docs/` yang relevan, lalu memperbaiki
  kode atau mengusulkan perubahan dokumen. Jangan diam-diam membiarkan keduanya berbeda.
- Mintalah penjelasan dalam Bahasa Indonesia untuk setiap modul selesai. Itu bahan laporan dan jawaban ke dosen.

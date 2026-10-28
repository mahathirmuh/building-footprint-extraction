# EXPERIMENTS.md — Skenario, Training, Tuning, dan Kriteria Keputusan

## 1. Pertanyaan → perbandingan → bukti

| Pertanyaan / kontribusi | Perbandingan utama | Metrik utama (klaim) | Data |
|---|---|---|---|
| RQ2 / N1 (head θ) | B5 vs B5′ vs FULL-P0 (semua input RGB) | `PoLiS_paired` dan `MTA_inst` di strata padat | `crowdai/test` |
| RQ1 / N3 (prior morfologi) | FULL-P2 vs FULL-P0 (MBI + celah sekaligus) | `merge_rate` di strata padat | `crowdai/test` |
| Atribusi N3 | FULL-P2 vs FULL-P1 (peta celah); FULL-P1 vs FULL-P0 (MBI) | sama | `crowdai/test` (Tier 2) |
| RQ3 (dekat SOTA, biaya) | FULL final vs HiSup (prediksi rilisan, data bersih); vs angka paper; efisiensi | AP; juga AP_boundary, PoLiS, C-IoU, parameter, FPS | `crowdai/test`, WHU |
| N2 (evaluasi per kepadatan) | semua baris di atas per strata; Δpadat − Δjarang | sama | semua |
| Klasik vs deep learning | B0 vs B2 / B4 | AP, IoU (deskriptif) | `crowdai/test` |

Dekomposisi N1, semuanya dengan input RGB (P0), sehingga prior N3 tidak ikut tercampur:
- **B5 → B5′:** model berbeda (B4 vs FULL-P0), sumber θ sama (minAreaRect) → efek **pelatihan** θ (multi-task).
- **B5′ → FULL-P0:** model sama (FULL-P0), sumber θ berbeda (minAreaRect vs head) → efek **pemanduan** θ.
- **B5 → FULL-P0:** efek total N1 (keduanya sekaligus).

Definisi metrik ada di `docs/EVALUATION.md`.

## 2. Matriks skenario

| Kode | Config | Model | Input | Output | Loss | Instance | Poligon | Training |
|---|---|---|---|---|---|---|---|---|
| B0 | `exp/b0` | klasik | RGB → MBI | — | — | Otsu + watershed | `mar` | tidak |
| B1 | `exp/b1` | U-Net EffB4 | RGB | 1 (biner) | BCE + Dice | CC | `raw` | ya |
| B2 | `exp/b2` | U-Net++ EffB4 | RGB | 1 | BCE + Dice | CC | `raw` | ya |
| B3 | `exp/b3` | U-Net++ | RGB | 1 | w·BCE + Dice | CC | `raw` | ya |
| B4 | `exp/b4` | U-Net++ | RGB | 3 | w·CE + Dice | watershed | `raw` | ya |
| B5 | `exp/b5` | model B4 | RGB | 3 | — | watershed | `mar` | tidak |
| B5′ | `exp/b5p` | model FULL-P0 | RGB | 5 | — | watershed | `mar` | tidak |
| FULL-P0 | `exp/full_p0` | U-Net++ | RGB | 5 | w·CE + Dice + λ·L_θ | watershed | `theta` | ya |
| FULL-P1 | `exp/full_p1` | U-Net++ | RGB + MBI | 5 | sama | watershed | `theta` | ya |
| FULL-P2 | `exp/full_p2` | U-Net++ | RGB + MBI + celah | 5 | sama | watershed | `theta` | ya |
| FULL+Y | `exp/full_y` | FULL final + YOLOv8n-OBB | sama dengan FULL final | 5 | — | watershed + seed YOLO | `theta` | YOLO saja |

- **FULL final** = varian FULL yang dipilih dengan aturan §8 (dipakai untuk SOTA, WHU, dan FULL+Y).
- **B5′ adalah tambahan** dari slide. Fungsinya memisahkan efek pelatihan dan efek pemanduan head θ.
- Config inferensi saja (`b5`, `b5p`, `full_y`) menunjuk ke checkpoint run sumbernya dengan seed yang sama; jangan
  menyalin bobot.

## 3. Pengaturan training (default; sama untuk semua model yang dibandingkan)

| Item | Nilai |
|---|---|
| Optimizer | AdamW, lr 2e-4, weight decay 1e-4 |
| Jadwal lr | warmup linear 1 epoch → cosine ke 1e-6 |
| Batch | 16 (CrowdAI 320²; WHU 224²). **Micro-batch minimal 8**: gradient accumulation hanya memperbesar batch efektif untuk optimizer, tidak untuk statistik BatchNorm |
| Presisi | AMP fp16 + GradScaler (T4 tidak cocok untuk bf16); loss dalam float32 |
| Budget | E epoch yang sama untuk semua run satu dataset; E ditetapkan setelah it/s diukur (§5) |
| Augmentasi | flip H/V, rot90, transpose (p 0,5 masing-masing); brightness/contrast ±0,1 hanya di RGB (p 0,5) |
| Seleksi checkpoint | IoU foreground di `val_tune` tiap epoch (sama untuk model biner dan 3 kelas); simpan `best.pt` dan `last.pt` |
| Early stopping | tidak ada (budget tetap) |
| Seed | 0, 1, 2; `cudnn.benchmark=True` (variasi antar-seed dilaporkan) |
| Loader | `num_workers=2` di Colab, `pin_memory=True`, label diturunkan on-the-fly dari cache instance |

## 4. Split validasi dan protokol tuning post-processing

`val_tune` (10% klaster train) dibagi tetap (seed 0) menjadi:
- `val_tune_sub` (1.000 citra): tuning post-processing.
- `val_select` (sisanya): memilih varian FULL final dan keputusan lain yang butuh data yang belum dipakai tuning.

Protokol tuning (CPU, tanpa training ulang):
1. **Tahap 1, instance:** grid `τ × m_min × A_min` (36 kombinasi) dengan poligon `raw`; pilih berdasarkan AP.
2. **Tahap 2, poligon:** dengan hasil tahap 1, grid `ρ × k × δ` (36 kombinasi) untuk mode `mar`/`theta`; pilih AP
   tertinggi, dan bila selisih < 0,2 AP pilih `PoLiS_paired` terkecil.
3. Grid sama untuk semua model. Model biner (B1–B3) hanya punya `A_min`; B0 menambah `t_min` (`docs/METHOD.md` §10).
4. Tuning di seed 0, parameternya dipakai ulang untuk seed 1–2 (dicatat).
5. Bekukan ke `configs/postproc/<dataset>/<exp>.yaml` (beserta skor val-nya) **sebelum** menyentuh test.

## 5. Prioritas run dan biaya

| Tier | Run training | Fase | Keterangan |
|---|---|---|---|
| 1 (wajib) | B4 ×3 seed, FULL-P0 ×3, FULL-P2 ×3 (CrowdAI) | 5–6 | B0, B5, B5′ tanpa training; cukup untuk klaim N1 dan N3 gabungan |
| 2 | FULL-P1 ×3; B1, B2, B3 ×1; FULL final di WHU ×1 | 7 | atribusi prior, rantai ablasi lengkap, banding WHU |
| 3 | B1–B3 seed tambahan; WHU ×3 seed; FULL+Y; indeks kepadatan sekunder | 7–9 | bila waktu ada |

Biaya GPU diisi setelah it/s diukur di Fase 5: `jam GPU = E × (iterasi per epoch) / (it/s × 3600)`.

| Run | GPU | it/s | iterasi/epoch | E | jam GPU per run | jumlah run | total |
|---|---|---|---|---|---|---|---|
| U-Net++ EffB4, CrowdAI 320², bs 16 | `TODO` | `TODO` | ≈ jumlah `train_dedup` / 16 | `TODO` | `TODO` | 9 (Tier 1) | `TODO` |
| U-Net++ EffB4, WHU 224², bs 16 | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | 1 | `TODO` |

Biaya CPU (diukur di CPU sandbox saat dokumen disusun; ukur ulang di mesinmu dengan 50 tile/citra):

| Pekerjaan | Per unit | Jumlah unit | Total (1 inti) |
|---|---|---|---|
| p-hash 6 varian (CrowdAI train + val) | ±6,4 ms/citra + baca disk | 341.058 citra | ±0,6 jam + I/O |
| MBI by reconstruction + peta celah | ±1,4 s/tile 320² | ±35 rb tile (train_dedup + val_tune + test) | ±14 jam |
| Tuning post-processing per model | `TODO` (ukur) | 72 kombinasi × 1.000 citra | `TODO` |

Kalau total Tier 1 melebihi kuota, kurangi E (sama untuk semua run) atau batasi jumlah tile per epoch; jangan
mengurangi seed Tier 1. Diskusikan dulu.

## 6. Protokol perbandingan SOTA

**CrowdAI bersih**
- HiSup: file prediksi rilisan → filter ke `image_id` di `crowdai/test` → evaluasi dengan `run_eval` yang sama
  dengan model kita. File GT subset dibuat dari `val/annotation.json` dengan id yang sama.
- Kita: FULL final, rata-rata ± SD 3 seed.
- Laporkan `CIoU` (semua citra) dan `CIoU_p2p` (aturan Pix2Poly), serta PoLiS dan MTA versi vendor, untuk kedua metode.
- Tabel terpisah **"dilaporkan di paper (val CrowdAI, bocor)"**: FFL, PolyWorld, HiSup, PolyR-CNN, Pix2Poly. Salin
  ulang angkanya dari PDF asli dan cantumkan nomor tabelnya (`configs/paper_numbers.yaml`); jangan disalin dari slide.

**WHU (protokol A)**
- FULL final dilatih di patch train Pix2Poly, dievaluasi di test dengan skrip yang di-vendor (aturan Pix2Poly:
  `CIoU_p2p`, PoLiS, MTA).
- Bandingkan dengan angka paper Pix2Poly (HiSup 79,62; Pix2Poly 81,63; FFL 32,19 / 54,52 C-IoU), ditandai sumbernya.

**Efisiensi:** jumlah parameter, FPS (batch 1, 320², AMP, 20 warmup + 200 iterasi), dan waktu post-processing per
citra (CPU 1 thread). Catat nama GPU/CPU. Angka efisiensi pembanding dari paper diberi catatan perangkat berbeda.

## 7. FULL+Y (opsional)

- Data: label OBB dari peta instance train dedup (`docs/METHOD.md` §8), dataset YAML Ultralytics.
- Model `yolov8n-obb.pt`, `imgsz` = 320 (CrowdAI) / 224 (WHU), 100 epoch, batch 32, seed 0, augmentasi bawaan.
- `c_y` disetel di `val_tune_sub`. Laporkan tambahan waktu inferensi (YOLO + penggabungan seed).

## 8. Kriteria keputusan (dibekukan 28 Okt 2026)

Aturan ini ditetapkan sebelum ada hasil apa pun. Kalau ingin mengubahnya, lakukan **sebelum Fase 3** (sebelum angka
HiSup di data bersih dihitung), dengan alasan tertulis di `docs/progress.md`. Setelah itu tidak boleh diubah.

1. **N1 didukung** bila di strata padat `crowdai/test`, FULL-P0 lebih baik dari B5 pada `PoLiS_paired` **dan**
   `MTA_inst` (keduanya dihitung pada GT yang ter-match oleh kedua metode), CI 95% bootstrap selisih berpasangan tidak
   memuat 0, arahnya sama di ketiga seed, dan AP tidak turun lebih dari 0,5 poin. Dekomposisi B5 → B5′ → FULL-P0
   dilaporkan dengan CI, sebagai analisis deskriptif.
2. **N3 (prior gabungan) didukung** bila FULL-P2 lebih baik dari FULL-P0 pada `merge_rate` di strata padat, CI tidak
   memuat 0, arah konsisten di ketiga seed, dan AP tidak turun lebih dari 0,5 poin. Satu metrik utama, jadi tidak ada
   koreksi multiplisitas.
3. **Klaim khusus "peta celah membantu"** hanya bila FULL-P2 vs FULL-P1 juga memenuhi kriteria nomor 2 (butuh Tier 2).
   FULL-P1 vs FULL-P0 (MBI) dilaporkan deskriptif.
4. **"Efek terbesar di area padat"** diklaim hanya bila CI bootstrap berstrata untuk Δpadat − Δjarang (pada metrik utama
   klaim yang bersangkutan) tidak memuat 0.
5. **"Mendekati SOTA":** selisih AP FULL final terhadap HiSup di `crowdai/test` ≤ 3 poin.
6. **Pemilihan FULL final:** AP tertinggi di `val_select`; bila selisih < 0,2 AP, pilih varian dengan kanal input
   lebih sedikit.
7. Bila kriteria tidak terpenuhi, laporkan apa adanya beserta analisisnya. Perbandingan di luar daftar ini adalah
   eksploratif dan ditandai begitu.

## 9. Templat tabel hasil (diisi oleh `scripts/tables.py`)

**Ablasi (`crowdai/test`, rata-rata ± SD; B1–B3 satu seed)**

| Kode | AP | AP50 | AP_b | IoU | PoLiS ↓ | MTA ↓ | C-IoU | C-IoU_p2p | N-ratio | merge rate ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | | | | | | | | | | |
| B1 | | | | | | | | | | |
| B2 | | | | | | | | | | |
| B3 | | | | | | | | | | |
| B4 | | | | | | | | | | |
| B5 | | | | | | | | | | |
| B5′ | | | | | | | | | | |
| FULL-P0 | | | | | | | | | | |
| FULL-P1 | | | | | | | | | | |
| FULL-P2 | | | | | | | | | | |

**N1 di strata padat (GT yang ter-match oleh semua metode di baris ini)**

| Kode | PoLiS_paired ↓ | MTA_inst ↓ | galat θ median ↓ | AP | Δ vs B5 [CI 95%] |
|---|---|---|---|---|---|
| B5 | | | | | — |
| B5′ | | | | | |
| FULL-P0 | | | | | |

**N3 di strata padat**

| Kode | merge rate ↓ | split rate ↓ | recall@0,5 | AP | Δ vs FULL-P0 [CI 95%] |
|---|---|---|---|---|---|
| FULL-P0 | | | | | — |
| FULL-P1 | | | | | |
| FULL-P2 | | | | | |

**SOTA (`crowdai/test`)**

| Metode | Anotasi poligon? | AP | AP_b | PoLiS | MTA | C-IoU | C-IoU_p2p | Param (M) | FPS | Sumber |
|---|---|---|---|---|---|---|---|---|---|---|
| HiSup | ya | | | | | | | | | prediksi rilisan, dievaluasi ulang |
| FULL final (kita) | tidak | | | | | | | | | run kita, 3 seed |

# EVALUATION.md — Metrik, Statistik, dan Uji Kewarasan

## 1. Format prediksi dan path

- Prediksi: `outputs/preds/<dataset>/<exp>/<run>/<split>.json`. Metrik:
  `outputs/metrics/<dataset>/<exp>/<run>/<split>.json`. `<run>` = `seed<k>` untuk model yang dilatih, `ref` untuk run
  tanpa seed (B0, `oracle_*`, `hisup_release`).
- Isi file prediksi: list COCO results, satu objek per instance:
  `{"image_id": int, "category_id": int, "segmentation": [[x1, y1, x2, y2, ...]], "score": float, "bbox": [x, y, w, h],
  "theta_deg": float | null, "kappa": float | null}`. Field tambahan diabaikan pycocotools tetapi dipakai diagnostik θ.
- `image_id` dan `category_id` diambil dari file GT (cek nilainya; konversi WHU Pix2Poly memakai `category_id = 100`).
- Koordinat di kerangka tile asli (padding sudah dibuang), satu cincin luar per instance, minimal 3 titik unik,
  konvensi titik penutup sama dengan GT.
- File GT subset per split dibuat sekali: `outputs/gt/<dataset>/<split>.json`. Isinya semua citra split, termasuk yang
  tanpa bangunan.

## 2. Metrik

### 2.1 Daftar

| Metrik | Tingkat | Definisi | Implementasi |
|---|---|---|---|
| AP, AP50, AP75, AP_S/M/L | instance | COCO mask AP, IoU 0,50:0,95 | `pycocotools` COCOeval `segm` |
| AP_boundary | instance | AP dengan Boundary IoU (dilation ratio 0,02) | `boundary-iou-api` COCOeval `boundary` |
| IoU, F1 | piksel | gabungan semua poligon vs gabungan GT; utama ΣI/ΣU seluruh split; juga rata-rata per citra | `eval/pixel.py` |
| Boundary IoU | piksel | Boundary IoU mask gabungan, rata-rata per citra | fungsi `boundary_iou` dari boundary-iou-api |
| `CIoU` (utama) | per citra | §2.2; **semua** citra split | `eval/pixel.py` (memakai fungsi vendor) |
| `CIoU_p2p` | per citra | §2.2; aturan Pix2Poly: hanya citra yang punya prediksi | vendor `cIoU.py` |
| N-ratio | per citra | rata-rata N / N_GT pada citra dengan GT (dua varian, mengikuti C-IoU) | vendor `cIoU.py` |
| `PoLiS` (vendor) ↓ | poligon | §2.2, aturan Pix2Poly/HiSup | vendor `polis.py` |
| `PoLiS_paired` ↓ | instance | PoLiS per pasangan dari pencocokan satu-satu (§2.4), rata-rata antar-pasangan | `eval/instance.py` + `compare_polys` vendor |
| `MTA` (vendor) ↓ | poligon | §2.2, aturan Frame Field/Pix2Poly (derajat) | vendor `angle_eval.py` |
| `MTA_inst` ↓ | instance | ukuran kontur yang sama, dihitung per pasangan (§2.4) tanpa penggabungan | `eval/instance.py` + fungsi vendor |
| `merge_rate` ↓ | instance | fraksi GT yang tertutup (luas(g∩p) / luas(g) ≥ 0,5) oleh prediksi yang juga menutup GT lain | `eval/instance.py` |
| `split_rate` ↓ | instance | fraksi GT yang ditutup ≥ 2 prediksi (luas(g∩p) / luas(p) ≥ 0,5) | `eval/instance.py` |
| `recall_inst50` | instance | fraksi GT yang punya pasangan di §2.4 | `eval/instance.py` |
| Vertex F1 (sekunder) | poligon | pada pasangan §2.4, verteks dipasangkan (Hungarian) bila jarak ≤ 2 px | `eval/vertex.py` |
| Galat θ (diagnostik) | instance | `min(Δ mod 90, 90 − Δ mod 90)` dengan Δ = abs(θ_pred − θ_GT) derajat, pada pasangan §2.4 | `eval/orientation.py` |
| Efisiensi | model | parameter (juta), FPS, waktu post-processing per citra | `scripts/benchmark.py` |

### 2.2 Perilaku metrik vendor yang harus dipahami

Dibaca dari kode Pix2Poly (`eval/hisup_eval_utils/metrics/`) dan pycocotools, 28 Okt 2026:
- **C-IoU (`cIoU.py`)**: per citra, `IoU(mask gabungan prediksi, mask gabungan GT) × (1 − abs(N − N_GT)/(N + N_GT))`,
  dengan N = jumlah verteks ring pertama tiap poligon. Citra yang diiterasi berasal dari objek hasil `loadRes`, dan
  `loadRes` membuang citra tanpa prediksi. Jadi **citra tanpa prediksi tidak ikut dihitung**: metode yang diam di citra
  sulit tidak dihukum. Selain itu `mask_gti`/`N_GT` tidak di-reset, sehingga pada citra yang punya prediksi tetapi tanpa
  anotasi GT nilainya basi (atau `UnboundLocalError` bila itu citra pertama). Fungsinya juga hanya mencetak, tidak
  me-return nilai.
  - `CIoU_p2p` = aturan di atas dengan `mask_gti`/`N_GT` di-reset per citra. Untuk WHU protokol A tidak mengubah angka,
    karena setiap patch punya minimal satu anotasi (patch negatif diberi anotasi kosong).
  - `CIoU` (utama) = rumus yang sama atas **semua** citra split: tanpa prediksi dan GT tidak kosong → 0; keduanya
    kosong → 1.
- **PoLiS (`polis.py`)**: pencocokan memakai IoU **bounding box** (bukan mask) > 0,5; tiap GT mengambil prediksi
  dengan IoU bbox tertinggi, sehingga satu prediksi bisa dipakai beberapa GT. PoLiS dirata-rata per citra, lalu citra
  bernilai tepat 0 (termasuk citra tanpa pasangan) **dilewati** saat merata-rata antar-citra. Hanya ring pertama yang
  dipakai. Kalau semua citra bernilai 0 (misalnya GT dipakai sebagai prediksi), kode asli membagi dengan nol.
  `compare_polys` menjumlahkan jarak n verteks tetapi membagi dengan 2·(n+1), karena titik penutup ikut dihitung di
  penyebut. Akibatnya nilainya n/(n+1) kali definisi Avbelj dkk. Ini konsisten untuk semua metode, jadi tetap dipakai
  demi kesebandingan dengan angka paper.
- **MTA (`angle_eval.py`)**: per citra, semua poligon GT digabung (`unary_union`, lalu `buffer(1e-4)`), begitu juga
  semua poligon prediksi. Kontur prediksi disampel tiap 2 px, diproyeksikan ke kontur GT, lalu sudut tangen maksimum
  dihitung per poligon (hasil gabungan) dan dikumpulkan dari semua citra. Akibatnya, **dinding antar-rumah yang
  bersentuhan hilang** dari GT maupun prediksi, padahal di situlah N1 diklaim. Karena itu klaim N1 memakai `MTA_inst`.
- **COCOeval**: `evaluate()` menjalankan `p.imgIds = list(np.unique(p.imgIds))`, sehingga duplikat `imgIds`
  dibuang. Ini memengaruhi bootstrap (§4).

### 2.3 Aturan vendor
- Salin `polis.py`, `angle_eval.py`, `cIoU.py` dari `Pix2Poly/eval/hisup_eval_utils/metrics/` (MIT) ke
  `third_party/pix2poly_metrics/`. Jangan memakai `HiSup/hisup/utils/metrics/cIoU.py` (variabel `mask_gti` tidak
  terdefinisi dan ada typo `img['weight']`).
- **Boleh** (perubahan pembungkus yang tidak mengubah matematika metrik): me-return nilai; mengeluarkan nilai per
  citra (untuk PoLiS: rata-rata citra dan status valid; untuk MTA: jumlah dan banyaknya nilai per citra); menerima
  subset `imgIds`; menangani pembagian nol dengan mengembalikan `None` plus jumlah citra valid; menghapus impor
  `descartes`; me-reset `mask_gti` dan `N_GT` per citra.
- **Tidak boleh:** mengubah pencocokan, ambang, sampling kontur, buffer, atau cara pooling.
- Setiap perubahan dicatat di header file dan di `docs/progress.md`. Tes regresi: pada file contoh, versi vendor
  dan versi asli memberi angka identik untuk semua kasus yang tidak terkena perubahan.

### 2.4 Pencocokan satu-satu (dasar metrik `*_paired`, `*_inst`, recall, Vertex F1, galat θ)
Seperti COCO: urutkan prediksi menurut skor; tiap prediksi dipasangkan dengan GT belum terpakai yang punya **IoU mask**
tertinggi, asalkan IoU ≥ 0,5. Untuk membandingkan dua metode A dan B, metrik berpasangan dihitung hanya pada GT yang
ter-match di **kedua** metode, sehingga keduanya diukur pada bangunan yang sama.

## 3. Evaluasi per strata

- AP: `cocoEval.params.imgIds = ids_strata`.
- Metrik vendor yang di-pool (`CIoU`, `CIoU_p2p`, `PoLiS`, `MTA`): nilai strata = metrik yang sama, dihitung dengan
  aturan vendor atas citra-citra strata itu. Simpan nilai per citra (C-IoU; PoLiS rata-rata citra + status valid; MTA
  jumlah dan banyak nilai) supaya pooling ulang dan bootstrap tidak perlu menghitung geometri lagi.
- Metrik instance dan berpasangan: dihitung pada GT yang berada di citra strata itu.
- Strata `kosong` hanya untuk metrik piksel dan jumlah positif palsu per citra.

## 4. Statistik

- Antar-seed: rata-rata ± SD (n = 3).
- **Bootstrap klaster per citra**, 1.000 replikasi, untuk semua metrik di `docs/EXPERIMENTS.md` §8:
  - Tiap replikasi menyampel ulang `image_id` dengan pengembalian (duplikat diizinkan). Sampel yang sama dipakai untuk
    kedua metode yang dibandingkan. Δ = metrik(B) − metrik(A) dihitung per seed, lalu dirata-rata antar-seed.
    CI 95% = persentil 2,5–97,5.
  - **AP:** jangan memberi `imgIds` duplikat ke COCOeval, karena duplikatnya dibuang diam-diam (§2.2), sehingga
    resampling berubah menjadi subsampel ±63% tanpa pengembalian. Jalankan `evaluate()` **sekali**, lalu implementasikan
    ulang langkah `accumulate()` atas daftar indeks citra yang memuat duplikat, memakai `evalImgs` yang sudah di-cache.
    Tes: resampel identitas harus memberi AP yang sama persis dengan `stats[0]`.
  - Metrik per citra dan berpasangan: rata-rata ulang dari nilai per citra/per pasangan yang tersimpan (§3).
- **Δpadat − Δjarang:** bootstrap berstrata, yaitu citra disampel ulang di dalam tiap strata secara terpisah, lalu
  dihitung (Δ di padat) − (Δ di jarang).
- Hanya perbandingan di `docs/EXPERIMENTS.md` §8 yang menjadi klaim. Perbandingan lain ditandai eksploratif.

## 5. Skema JSON metrik

`outputs/metrics/<dataset>/<exp>/<run>/<split>.json`:

```json
{
  "dataset": "crowdai", "split": "test", "exp": "full_p2", "run": "seed0",
  "git": "<hash>", "created": "<ISO time>",
  "checkpoint": "outputs/runs/crowdai/full_p2/seed0/best.pt",
  "postproc": {"tau": 0.4, "m_min": 8, "A_min": 10, "rho": 0.9, "k": 0.04, "delta": 15},
  "overall": {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_boundary": 0.0, "IoU": 0.0, "IoU_per_image": 0.0,
              "F1": 0.0, "BoundaryIoU": 0.0, "CIoU": 0.0, "CIoU_p2p": 0.0, "N_ratio": 0.0,
              "PoLiS": 0.0, "PoLiS_valid_images": 0, "MTA": 0.0, "PoLiS_paired": 0.0, "MTA_inst": 0.0,
              "merge_rate": 0.0, "split_rate": 0.0, "recall_inst50": 0.0, "vertex_F1": 0.0,
              "theta_err_median": null, "theta_err_p90": null,
              "n_images": 0, "n_pred": 0, "n_fallback": 0},
  "per_stratum": {"jarang": {}, "sedang": {}, "padat": {}, "kosong": {"IoU": 0.0, "fp_per_image": 0.0}},
  "efficiency": {"params_M": 0.0, "fps": 0.0, "postproc_ms": 0.0, "gpu": "", "cpu": ""}
}
```

Simpan juga `per_image.parquet` (id, strata, IoU, C-IoU, flag punya prediksi, PoLiS citra + valid, MTA jumlah + banyak,
n_pred, n_gt) dan `pairs.parquet` (id citra, id GT, id prediksi, IoU, PoLiS pasangan, MTA pasangan, galat θ) untuk
bootstrap. Cache `evalImgs` COCOeval disimpan sebagai pickle di folder yang sama.

## 6. Uji kewarasan (wajib lolos sebelum angka apa pun dilaporkan)

### 6.1 GT sebagai prediksi (Fase 3)
Ubah GT menjadi file prediksi (skor 1). Harus menghasilkan: AP = 100 (±0,1); AP_boundary = 100; `CIoU` = `CIoU_p2p`
= 1; `PoLiS_paired` = 0; `MTA_inst` ≈ 0 dan `MTA` ≤ 0,5° (GT di-buffer 1e-4); `PoLiS` vendor tidak terdefinisi
(semua citra 0), sehingga pembungkus mengembalikan `None` dengan `PoLiS_valid_images = 0`, bukan error;
`merge_rate` dan `split_rate` = 0 kecuali di tempat poligon GT saling tumpang tindih (laporkan jumlahnya).

Tes sintetis:
- Persegi sejajar sumbu yang digeser d px searah x → `PoLiS` = `PoLiS_paired` = 0,4·d. Dua verteks di tiap sisi
  berjarak d, sehingga totalnya 4d; dibagi 2·5 karena titik penutup ikut di penyebut (§2.2). Definisi asli memberi d/2.
- Poligon identik → `MTA_inst` = 0. Diputar α ∈ {5°, 10°, 20°} terhadap pusatnya → `MTA_inst` ≥ α − 0,5°.

### 6.2 Kasus tepi (Fase 3)
- Citra tanpa prediksi: `CIoU` = 0 bila GT tidak kosong, 1 bila keduanya kosong; `CIoU_p2p` melewatinya.
- Citra yang punya prediksi tetapi tanpa GT: `CIoU` = `CIoU_p2p` = 0 (setelah reset `mask_gti`).
- Poligon prediksi tidak valid ditolak atau diperbaiki secara konsisten, lalu dicatat.
- Evaluasi file yang sama dua kali memberi angka identik.

### 6.3 Oracle post-processing (Fase 4, di `val_tune_sub`)
- (a) **Bolak-balik:** poligon GT → raster → `mask_to_exterior` → raster: IoU rata-rata ≥ 0,99. Metode di
  `docs/METHOD.md` §6.1 memberi 1,000 pada uji awal.
- (b) **Mask instance GT** → `polygonize(raw | mar | theta dengan θ GT)` → semua metrik. Ini batas atas tahap poligon.
- (c) **Peta 3 kelas GT** dijadikan probabilitas sempurna → marker → watershed → poligon. Bila AP < 95, kemungkinan ada
  bug di marker/watershed.

### 6.4 Reproduksi HiSup (Fase 3)
File prediksi rilisan HiSup di **val penuh** (`val_full`) dievaluasi dengan pipeline kita. AP harus berada dalam ±0,3
poin dari angka paper (79,4); AP_boundary dan PoLiS dibandingkan dengan tabel paper HiSup (salin dari PDF, cantumkan
nomor tabel). Bila tidak cocok, periksa versi pycocotools, file anotasi yang dipakai (`annotation.json` vs
`annotation-small.json`), dan pemakaian skor. Setelah lolos, jalankan di `crowdai/test`.

### 6.5 (Opsional) Reproduksi Pix2Poly di WHU (Fase 8)
Bila checkpoint atau prediksi rilisan dipakai, `CIoU_p2p` test harus mendekati 81,63.

### 6.6 Bootstrap AP (Fase 3)
Resampel identitas = `stats[0]` persis. Resampel dengan duplikat memberi distribusi AP yang masuk akal (CI memuat
nilai titik). Tunjukkan juga, satu kali, bahwa memberi `imgIds` duplikat langsung ke COCOeval memberi CI yang berbeda
(bukti mengapa cara itu tidak dipakai).

Ringkasan hasil uji kewarasan ditulis ke `docs/sanity_report.md` (tanggal, versi paket, angka, lolos/gagal).

## 7. Pelaporan

- `scripts/tables.py` dan `scripts/figures.py` hanya membaca file JSON/parquet di `outputs/metrics/`. Tidak ada angka
  yang diketik tangan.
- Setiap sel tabel bisa ditelusuri ke path JSON (`scripts/tables.py --trace` mencetak sumber tiap sel).
- Angka dari paper lain disimpan di `configs/paper_numbers.yaml` dengan field `source` (paper, tabel, halaman) dan
  `split_note` (misalnya "val CrowdAI, bocor").
- Laporan menjelaskan perilaku metrik vendor (§2.2) dan menyebut varian mana yang dipakai di tiap tabel.

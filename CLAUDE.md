# CLAUDE.md — Ekstraksi Poligon Bangunan pada Permukiman Padat

## Tentang proyek

Final project mata kuliah Pengolahan Citra Digital (S2 Informatika ITS) — Mahathir Muhammad (6025252008).

**Judul kerja:** Ekstraksi Poligon Bangunan pada Permukiman Padat — U-Net++ dengan Prior Morfologi Celah,
Head Orientasi Berbasis Mask, dan Regularisasi Poligon Terpandu.

**Tujuan:** dari citra udara/satelit RGB, menghasilkan **satu poligon rapi per rumah** (output COCO JSON dan
GeoJSON), termasuk rumah yang atapnya berdempetan, **tanpa anotasi poligon** saat training (cukup mask per bangunan).

**Pipeline singkat:** RGB + 2 kanal prior morfologi → U-Net++ (EfficientNet-B4) → 3 kelas (latar / bangunan /
kontak) + orientasi θ (sin 4θ, cos 4θ) → marker `P_b − P_k > τ` → watershed → kontur per rumah → regularisasi
poligon (kotak searah θ, atau Douglas-Peucker + pelurusan ke θ). Spesifikasi lengkap: `docs/METHOD.md`.

**Kontribusi baru (boleh diklaim):**
- **N1 — metode:** head orientasi θ yang labelnya diturunkan dari mask (minAreaRect), dipakai memandu regularisasi.
- **N2 — evaluasi:** metrik per tingkat kepadatan, di CrowdAI yang sudah dideduplikasi (dan WHU).
- **N3 — preprocessing:** prior morfologi (MBI + peta celah black top-hat multi-arah) **ditambahkan** ke RGB dengan
  bobot awal nol.

**Diadopsi (jangan diklaim sebagai kebaruan):** kelas kontak + watershed (TernausNetV2, Iglovikov dkk. 2018),
bobot tepi (U-Net; Bischke dkk. 2019), U-Net++ (Zhou dkk. 2018), MBI (Huang & Zhang 2011), deduplikasi p-hash
(Adimoolam dkk.).

**Hipotesis yang diuji:** head θ menurunkan PoLiS dan MTA; prior celah menambah rumah yang terpisah; keduanya paling
terasa di kelompok tile terpadat. Hasil negatif tetap dilaporkan apa adanya.

Rencana kerja per fase ada di `PLAN.md`. Detail ada di `docs/`:
`METHOD.md` (algoritma), `DATA.md` (dataset, deduplikasi, label), `EXPERIMENTS.md` (skenario B0–FULL+Y),
`EVALUATION.md` (metrik dan uji kewarasan), `COLAB.md` (training di Colab dan pekerjaan CPU berat). Log kerja di
`docs/progress.md`.

## Status (28 Oktober 2026)

- Desain di file ini adalah versi ETS yang sudah disempurnakan (perubahannya dicatat di `docs/progress.md`). Kode demo
  tahap ETS ada di `reference/demo/`, prototipe uji spesifikasi di `reference/prototypes/`. Logikanya boleh di-port ke
  `src/`, tetapi **jangan diimpor langsung**.
- Fakta yang **sudah diverifikasi** (28 Okt 2026) dari repo dan halaman resmi:

| Hal | Hasil cek |
|---|---|
| CrowdAI train / val | 280.741 / 60.317 citra (dari file hash repo `CrowdAI_Hash_and_search`) |
| CrowdAI train, p-hash unik (tanpa augmentasi) | 114.548; README repo: ±250 rb dari 280 rb adalah duplikat persis **atau** teraugmentasi |
| CrowdAI val tanpa duplikat persis di train | **3.949** citra (dari `val_in_train_np.json`; direproduksi persis). Bila varian putar/cermin ikut dicek: 1.511 citra = **212 tile unik** (`val_clean`) |
| HiSup | merilis file prediksi val CrowdAI (JSON) → bisa dievaluasi ulang tanpa menjalankan model HiSup |
| Pix2Poly, protokol WHU | data `WHU_aerial_0.2` (0,2 m), tile **512²** (paper: 9.420 / 1.537 / 3.848), patch 224 × 224 overlap 34, GT dari mask biner (connected component + simplify 1,8) |
| WHU resmi | ada versi 0,3 m (8.189 tile 512²), versi 0,2 m (raster + **format COCO**), dan shapefile seluruh area. Raster 0,2 m (29 Okt 2026): tile **1024²** GeoTIFF, train 2.943 / val 627 / test 2.220, label dibuat ulang Jul 2024; COCO resmi memisahkan bangunan per bangunan |
| smp 0.6.0 | `in_channels≠3` menyalin bobot RGB secara siklik lalu mengalikan 3/in_ch; input wajib kelipatan 32 |
| Ultralytics 8.4.x | `yolov8n-obb.pt` tersedia; label OBB: `cls x1 y1 x2 y2 x3 y3 x4 y4` ternormalisasi |

- Fase 0 selesai (kerangka repo, config, paths, fixture sintetis; lihat `docs/progress.md`). Fase 1 selesai dan
  disetujui (29 Okt 2026): split dan strata kepadatan CrowdAI dan WHU (opsi R, `docs/DATA.md` §3.2) dibekukan.
  Fase 2 selesai dan disetujui (30 Okt 2026): label diturunkan saat training; prior dibekukan dan di-cache untuk
  semua split. Fase 3 selesai dan disetujui (30 Okt 2026): pipeline evaluasi lolos uji kewarasan (53 dari 53) dan
  mereproduksi angka HiSup di val penuh (AP 79,43; dilaporkan di paper 79,4). Fase 4 (post-processing dan B0)
  berjalan. Belum ada model.

## Cara bekerja di proyek ini

- Kerjakan **satu fase `PLAN.md` dalam satu waktu**. Lanjut hanya setelah kriteria "Selesai jika" terpenuhi dan saya
  menyetujuinya.
- Sebelum menulis kode besar, jelaskan rencana dan alasannya singkat (termasuk alternatif bila ada). Setelah selesai,
  jelaskan apa yang dibuat **dalam Bahasa Indonesia**. Saya harus bisa menjelaskannya ulang ke dosen.
- Kode, nama variabel, docstring, dan komentar teknis dalam bahasa Inggris.
- **Tes dulu di data sintetis.** Setiap modul label, prior, post-processing, dan metrik wajib punya unit test dengan
  citra sintetis yang jawabannya diketahui (lihat `tests/fixtures`). Tes tidak boleh butuh internet, GPU, atau dataset.
- **Minta konfirmasi** sebelum: training lebih dari ±10 menit, menjalankan proses di seluruh dataset (prior, hash,
  inferensi penuh), menghapus data atau checkpoint, atau mengubah keputusan metodologi di bawah.
- **Jangan pernah mengarang angka.** Semua angka laporan berasal dari output skrip (`outputs/metrics/**/*.json`).
  Angka dari paper lain selalu ditandai "dilaporkan di paper".
- **Jangan pernah menyetel parameter di data test.** Ambang dan grid post-processing disetel di `val_tune_sub`;
  pemilihan model memakai `val_select`.
- Commit git per langkah kecil dengan pesan jelas. `data/`, `outputs/`, `*.pt`, dan output notebook tidak di-commit
  (`nbstripout`). Versi notebook berisi output diekspor ke `outputs/report/`.
- Setelah tiap fase, tambahkan ringkasan ke `docs/progress.md`: apa yang dibuat, angka penting, masalah, keputusan.
- Kalau menemukan fakta yang bertentangan dengan dokumen ini, berhenti dan laporkan; jangan diam-diam menyesuaikan.

## Keputusan metodologi (jangan diubah tanpa diskusi)

1. **Arsitektur:** `smp.UnetPlusPlus("efficientnet-b4", encoder_weights="imagenet")`. Output satu head konvolusi
   5 kanal: 3 logit kelas + 2 kanal orientasi (tanh). Model B1–B4 memakai jumlah kanal output sesuai skenario.
2. **Input 5 kanal dengan bobot awal nol:** model dibuat 3 kanal (pretrained), lalu konvolusi pertama diperluas ke
   5 kanal: bobot RGB **tidak diubah**, bobot kanal 4–5 = 0. **Jangan** memakai `in_channels=5` bawaan smp.
3. **Kelas:** 0 = latar, 1 = bangunan, 2 = kontak. Kontak = piksel bangunan yang berjarak ≤ r piksel dari bangunan
   lain (setara "hasil dilasi ≥ 2 bangunan", dibatasi ke piksel bangunan). Default r = 2.
4. **Orientasi:** target (sin 4θ, cos 4θ), θ dari `cv2.boxPoints(cv2.minAreaRect(...))` (vektor sisi), bukan dari
   nilai sudut OpenCV. Loss θ hanya di piksel bangunan GT. Label diturunkan **setelah** augmentasi geometri.
5. **Loss:** `L = w·CE + Dice + λ·L_θ`, dengan `w = 1 + α·exp(−d²/2σ²)` dari distance transform ke tepi bangunan.
6. **Tanpa resize.** Tile CrowdAI 300 × 300 di-pad ke 320 (citra: reflect; label: ignore 255), prediksi di-crop
   kembali ke 300.
7. **Deduplikasi CrowdAI wajib.** Train = satu wakil per klaster duplikat (p-hash dengan augmentasi rotasi/flip).
   Klaster train diacak sekali (seed 0): 10% pertama = `val_tune`, dibagi lagi menjadi `val_tune_sub` (1.000 citra,
   tuning post-processing) dan `val_select` (sisanya, pemilihan model); 10% berikutnya = `test` (holdout, untuk
   semua klaim N1–N3 dan strata kepadatan); sisanya `train_dedup`. `val_clean` = val CrowdAI yang tidak punya
   duplikat (termasuk putar/cermin) di train, satu wakil per klaster (212 tile), khusus untuk banding HiSup. Val
   CrowdAI yang bocor (`val_full`) **tidak dipakai** untuk training, tuning, atau klaim; hanya untuk mereproduksi
   angka HiSup. (Diubah 28 Okt 2026, opsi E: val bersih hanya 212 tile, terlalu kecil untuk klaim utama.)
8. **Strata kepadatan (N2)** dihitung dari GT test **sebelum** evaluasi model, disimpan di
   `splits/density_<dataset>_<split>.json`, lalu dibekukan. Default: tertil jumlah bangunan GT per tile; tile tanpa bangunan = strata "kosong".
9. **Ablasi satu perubahan per langkah** (`docs/EXPERIMENTS.md`). Uji N1 seluruhnya memakai input RGB: B5 (model B4,
   θ minAreaRect) → B5′ (model FULL-P0, θ minAreaRect) → FULL-P0 (model FULL-P0, θ head). B5 dan FULL-P0 berbeda dua
   hal (pelatihan θ dan sumber θ); B5′ memisahkan keduanya. Varian FULL dengan prior (P1, P2) **tidak** dipakai untuk N1.
10. **Post-processing disetel per model dengan grid yang sama** di `val_tune_sub`, lalu dibekukan ke
   `configs/postproc/<dataset>/<exp>.yaml` sebelum test. Kriteria keputusan di `docs/EXPERIMENTS.md` §8 sudah
   dibekukan (28 Okt 2026).
11. **Pembanding SOTA:** HiSup dievaluasi dari file prediksi rilisannya, difilter ke `val_clean` (satu-satunya
    subset yang juga bersih bagi HiSup, yang dilatih dengan seluruh train v1.0). PolyR-CNN dan
    Pix2Poly di CrowdAI hanya dikutip dari paper (split bocor, ditandai). Di WHU (opsi R, 29 Okt 2026), tile resmi
    1024² dipotong menjadi kuadran 512² yang berisi bangunan (test: bagian `300` saja), lalu dibuat patch Pix2Poly
    224/34. GT `p2p` (konversi Pix2Poly persis) dipakai untuk banding angka paper; perbandingannya hanya kira-kira,
    karena test Pix2Poly tidak bisa direkonstruksi persis. GT utama (bangunan resmi, terpisah per bangunan) dipakai
    untuk training dan klaim N1–N3 di WHU.
12. **Evaluasi:** pakai skrip metrik Pix2Poly (`polis`, `angle`/MTA, `ciou`) yang di-vendor + pycocotools +
    boundary-iou-api, untuk angka yang sebanding dengan paper. Klaim N1 memakai metrik berpasangan per instance
    (`PoLiS_paired`, `MTA_inst`); C-IoU dilaporkan dalam dua varian (`CIoU` semua citra, `CIoU_p2p` aturan Pix2Poly).
    Pipeline evaluasi dianggap valid hanya setelah lolos uji kewarasan di `docs/EVALUATION.md` §6.
13. **Seed:** 0, 1, 2 untuk perbandingan inti; laporkan rata-rata ± SD dan selang kepercayaan bootstrap.

## Data (ringkas; detail di `docs/DATA.md`)

| Dataset | Peran | Catatan |
|---|---|---|
| CrowdAI Mapping Challenge | train, `val_tune`, `test` (holdout); metrik poligon; banding HiSup di `val_clean` | v1.0, unduh manual dari AIcrowd (perlu akun); v2.0 tidak dipakai (lihat "Yang perlu saya putuskan") |
| WHU Building (aerial) | banding C-IoU dengan Pix2Poly/HiSup (GT `p2p`, kira-kira); uji kedua rumah berdempetan (GT utama) | 0,2 m, opsi R (`docs/DATA.md` §3.2) |
| SpaceNet 6 Rotterdam (opsional) | gambar kualitatif, uji lintas-dataset | CC BY-SA 4.0 |

**Nama dataset dan split** (dipakai di CLI, config, dan path output): dataset `crowdai` dan `whu`; split
`train_dedup`, `val_tune` (`val_tune_sub`, `val_select`), `test`. Untuk CrowdAI, `test` = 10% klaster train yang
disisihkan. Khusus CrowdAI: `val_clean` (212 tile val tanpa duplikat di train) untuk banding HiSup, dan `val_full`
(val asli yang bocor) hanya untuk mereproduksi angka HiSup.

Path data selalu dari variabel lingkungan `DENSEPOLY_DATA` (default `./data`), output dari `DENSEPOLY_OUT`
(default `./outputs`). Jangan menulis path absolut di kode.

## Lingkungan

- Python ≥ 3.10 (ikuti versi Colab). Paket inti: `torch`, `segmentation-models-pytorch==0.6.*`, `timm`,
  `opencv-python-headless>=4.10,<5`, `scikit-image`, `scipy`, `shapely>=2`, `pycocotools`, `imagehash`,
  `omegaconf`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `pytest`, `multiprocess` (diimpor skrip metrik vendor),
  `pyarrow` (tabel parquet metrik).
- Evaluasi: `boundary-iou-api` (BSD-2-Clause), bagian COCO-nya di-vendor persis ke `third_party/boundary_iou/`
  (30 Okt 2026), karena `pip install` dari GitHub hanya memasang `__init__.py`. `COCOeval.accumulate()`-nya memakai
  `np.float` (rusak di numpy ≥ 1.24) dan tidak dipakai: AP dihitung `eval/stats.py` dari `evalImgs`.
- Opsional FULL+Y: `ultralytics` (lisensi **AGPL-3.0**; hanya untuk eksperimen, jangan dicampur ke paket inti).
- Training penuh di GPU Colab (`docs/COLAB.md`). Pekerjaan CPU berat (hash ±0,6 jam-inti, prior ±14 jam-inti, evaluasi
  val penuh, tuning post-processing) dijalankan di laptop multi-core bila datanya ada di sana, atau di sesi CPU Colab
  (`docs/COLAB.md` §8). Selalu ukur dulu di sampel kecil dan minta konfirmasi.
- Bobot pretrained smp diunduh dari Hugging Face Hub (`smp-hub/efficientnet-b4.imagenet`). Tes selalu memakai
  `encoder_weights=None`.

## Struktur folder

```
densepoly/
├── CLAUDE.md  PLAN.md  README.md  pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── data/{crowdai,whu}.yaml
│   ├── exp/{b0,b1,b2,b3,b4,b5,b5p,full_p0,full_p1,full_p2,full_y}.yaml
│   ├── postproc/<dataset>/<exp>.yaml        # hasil tuning di val_tune (dibekukan)
│   └── paper_numbers.yaml                   # angka dari paper lain + sumber (tabel, halaman)
├── src/densepoly/
│   ├── config.py  paths.py  utils/
│   ├── data/        # crowdai.py, whu.py, dedup.py, density.py, dataset.py, transforms.py
│   ├── labels/      # instances.py, contact.py, orientation.py, boundary_weight.py
│   ├── priors/      # mbi.py, gap.py, cache.py
│   ├── models/      # factory.py, first_conv.py
│   ├── losses.py  train.py  infer.py
│   ├── postproc/    # markers.py, watershed.py, contours.py, regularize.py, polygonize.py, yolo_seeds.py
│   ├── classical/b0.py
│   └── eval/        # coco_io.py, pixel.py, instance.py, vertex.py, orientation.py, run_eval.py, stats.py
├── third_party/pix2poly_metrics/            # polis.py, angle_eval.py, cIoU.py (MIT, dengan catatan perbaikan)
├── scripts/         # CLI: prepare_*, dedup_crowdai, make_density, precompute_priors, train, predict, evaluate,
│                    #      tune_postproc, benchmark, tables, figures
├── notebooks/       # colab_train.ipynb (menjalankan skrip di Colab), demo_hasil.ipynb (demo untuk dosen, Fase 10);
│                    # keduanya tipis: logika tetap di src/, notebook hanya memanggil dan memplot
├── tests/           # fixtures/ sintetis + test_*.py
├── reference/       # demo/ (kode demo ETS) dan prototypes/ (uji spesifikasi); hanya referensi
├── docs/            # METHOD, DATA, EXPERIMENTS, EVALUATION, COLAB, progress
├── splits/          # daftar id dedup, val_tune, test, val_clean, density_*.json (di-commit, kecil)
├── data/            # (gitignored)
└── outputs/         # (gitignored) gt/, runs/, preds/, metrics/, tables/, figures/
```

## Perintah

```bash
pip install -e ".[dev]"                     # + ".[yolo]" untuk FULL+Y; metrik vendor ada di third_party/
pytest -q                                   # wajib lolos sebelum bilang "selesai"
python scripts/dedup_crowdai.py --data crowdai        # path dibaca dari DENSEPOLY_DATA; output ke splits/
python scripts/make_density.py --data crowdai --split test
python scripts/prepare_gt.py --data crowdai           # GT per split untuk evaluasi → outputs/gt/ (WHU: + _p2p)
python scripts/precompute_priors.py --data crowdai --split train_dedup
python scripts/train.py --data crowdai --exp full_p2 --seed 0 [--resume]
python scripts/predict.py --data crowdai --exp full_p2 --seed 0 --split test
python scripts/evaluate.py --data crowdai --exp full_p2 --seed 0 --split test
python scripts/tables.py && python scripts/figures.py
```

## Jebakan teknis (baca sebelum menulis kode)

- **smp `in_channels`:** untuk `in_channels>3`, smp mengisi kanal baru dengan salinan siklik bobot RGB lalu
  mengalikan semua bobot dengan 3/in_ch. Bobot pretrained RGB jadi berubah. Pakai `models/first_conv.py`
  (zero-init, cari `nn.Conv2d` pertama dengan `in_channels==3`, ganti `.weight` di tempat, perbarui
  `encoder._in_channels` dan `encoder._out_channels[0]`). Tes: output model 5 kanal == model 3 kanal untuk input
  apa pun.
- **smp 0.6:** `model.decoder(features)` menerima list (versi 0.3.x: `*features`). Jangan memanggil decoder langsung;
  cukup `classes=5` di head bawaan.
- **Kelipatan 32:** smp melempar error bila H atau W bukan kelipatan 32.
- **θ dan augmentasi:** flip horizontal/vertikal dan transpose mengubah tanda sin 4θ; rotasi 90° tidak mengubah
  apa pun. Paling aman: augmentasi peta instance, lalu turunkan semua label sesudahnya.
- **Koordinat:** numpy `(row, col)` vs OpenCV/shapely/COCO `(x, y)`. COCO `segmentation` = `[[x1, y1, x2, y2, ...]]`,
  minimal 3 titik, **tanpa lubang** (format poligon COCO tidak mendukung lubang).
- **Ekstraksi kontur:** `cv2.findContours` memberi pusat piksel tepi sehingga poligon menyusut ±0,5–1 px. Itu fatal
  untuk rumah kecil (10 × 10 px → IoU turun ±20%). Pakai marching squares (`skimage.measure.find_contours` pada
  mask yang di-pad, level 0,5, koordinat −1 + 0,5) dan buktikan dengan uji bolak-balik di `docs/EVALUATION.md`.
- **Kotak regularisasi:** pakai momen orde-2, bukan bounding box. Bounding box ikut membesar karena tonjolan derau
  (prototipe: IoU 0,89 vs 0,98–0,99). Lihat `reference/prototypes/regularize_proto.py`.
- **Titik penutup poligon:** ikuti konvensi GT (cek apakah titik pertama diulang di akhir). Jumlah verteks
  memengaruhi C-IoU dan N-ratio.
- **Watershed:** marker dipotong ke foreground (`M & F`) sebelum diberi label; komponen foreground tanpa marker tetap
  jadi instance sendiri; semua pelabelan sisi prediksi memakai konektivitas 4 (`docs/METHOD.md` §6.2).
- **Perilaku skrip metrik** (detail di `docs/EVALUATION.md` §2.2):
  - `HiSup/hisup/utils/metrics/cIoU.py` rusak (`mask_gti` tak terdefinisi, `img['weight']`), jadi jangan dipakai.
  - C-IoU Pix2Poly hanya menghitung citra yang **punya prediksi**, tidak me-reset `mask_gti` (IoU basi pada citra
    berprediksi tanpa GT; C-IoU-nya tetap ≈ 0 karena `N_GT` di-reset), dan tidak me-return nilai.
  - PoLiS Pix2Poly mencocokkan via IoU **bounding box** > 0,5, melewati citra bernilai 0, dan membagi dengan n+1.
    `polis.py` menimpa `segmentation` GT yang kosong di tempat, jadi setiap metrik vendor memakai objek COCO sendiri.
  - MTA menggabungkan semua poligon per citra (`unary_union`), sehingga dinding antar-rumah bersentuhan hilang.
    Nilainya didominasi artefak sudut (rotasi 1° → 9,65°). Di GEOS 3.14 skrip aslinya crash pada citra ber-GT kosong
    yang punya prediksi; pembungkus 4 mengembalikan hasil kosong untuk citra itu. Cosinus yang terbulatkan sedikit di
    atas 1 (semua tepi sejajar persis) membuat skrip asli memberi NaN; pembungkus 5 memotongnya ke 1.
  - `angle_eval.py` mengimpor `descartes` hanya untuk plot; impor itu dipindah ke dalam fungsi plot.
  - `loadRes` pycocotools butuh `bbox` di setiap hasil poligon (tanpa itu `segmentation` dianggap RLE), dan `area`
    deteksi = luas bbox (memengaruhi AP_S/M/L).
  - `COCOeval.evaluate()` membuang `imgIds` duplikat, jadi bootstrap AP harus memakai `evalImgs` yang di-cache.
- **WHU:**
  - GT `p2p` (cara Pix2Poly: connected component dari label raster) menyatukan bangunan yang bersentuhan. GT utama
    dari COCO resmi memisahkannya. Klaim "rumah berdempetan" di WHU hanya boleh memakai GT utama.
  - TIFF resmi berkompresi LZW, jadi perlu `imagecodecs`.
  - Tile tanpa bangunan disimpan sebagai `int8`. Nilainya nol semua, tetapi label bernilai negatif harus ditolak.
  - Patch tidak disimpan sebagai citra. Potong dari tile resmi memakai `patches.csv`.
- **AMP:** hitung softmax, CE, dan Dice dalam float32 (`logits.float()`).
- **EfficientNet + BatchNorm:** micro-batch < 8 membuat BN tidak stabil. Gradient accumulation hanya memperbesar
  batch efektif untuk optimizer, tidak untuk statistik BN, jadi micro-batch tetap minimal 8.
- **Loss dan padding:** piksel 255 tidak boleh menyumbang loss di model mana pun, termasuk biner (BCE target dari
  `y == 1`, `w = 0` di 255, `DiceLoss(mode="binary", ignore_index=255)`).
- **Skor instance untuk AP:** rata-rata `1 − P_latar` di piksel instance. B0 tidak punya probabilitas: pakai skor
  konstan 1 dan catat bahwa AP B0 setara satu titik precision–recall.

## Yang perlu saya putuskan / lengkapi

- **Protokol WHU: opsi R dipilih** (29 Okt 2026; menggantikan protokol A dari 28 Okt 2026). Pix2Poly memakai tile
  512² yang tidak bisa direkonstruksi persis dari rilisan resmi (1024²), sehingga angka C-IoU paper (HiSup 79,62;
  Pix2Poly 81,63) hanya bisa dibandingkan secara kira-kira. Karena COCO resmi memisahkan bangunan yang bersentuhan,
  WHU dipakai dengan dua GT pada patch yang sama (keputusan #11, `docs/DATA.md` §3.2). Alternatif B (tile 0,3 m 512²
  seperti di slide) tidak dipakai.
- **CrowdAI: v1.0 + deduplikasi sendiri dipilih** (28 Okt 2026, opsi A). v2.0 (Mei 2025, kebocoran antar-split
  diperbaiki panitia) tidak dipakai di eksperimen inti, karena prediksi HiSup dan angka paper pembanding hanya ada di
  v1.0. v2.0 disebut di laporan. Opsional di akhir: satu run FULL di v2.0 bila waktu GPU tersisa.
- **Desain test CrowdAI: opsi E** (28 Okt 2026; keputusan #7 diubah). `test` = 10% klaster train yang disisihkan
  (2.928 tile) untuk semua klaim N1–N3 dan strata kepadatan; `val_clean` = 212 tile val bersih untuk banding HiSup.
  Alasan: val yang bebas duplikat (termasuk putar/cermin) di train hanya 212 tile (`docs/DATA.md` §2.2).
- Unduhan CrowdAI v1.0 selesai (28 Okt 2026). Unduhan WHU 0,2 m (raster + COCO resmi) selesai (29 Okt 2026). Ruang
  Google Drive: `TODO` (WHU saja ±16 GB citra).
- Tier GPU Colab (T4 gratis / Pro): `TODO`. Repo GitHub privat + token di Colab Secrets: `TODO`.
- Definisi kepadatan: default disetujui (28 Okt 2026), yaitu jumlah bangunan GT per tile dibagi tertil. Ambang CrowdAI
  disetujui dan dibekukan 28 Okt 2026: 6 dan 11 bangunan per tile. Indeks sekunder diganti menjadi tile bersentuhan
  vs tidak (r = 2), karena tertil rasio bersentuhan degenerasi. Ambang WHU (GT utama, per patch 224², r = 3)
  disetujui dan dibekukan 29 Okt 2026: 3 dan 6 bangunan per patch.
- Prior (Fase 2) dibekukan 29 Okt 2026: mode celah `contrast`, L_g 9 px (WHU 13), skala MBI CrowdAI 11–81 px dan WHU
  15, 31, 45, 61, 75, 91, 105, 121 px (SE garis harus ganjil). Radius kontak r dan bobot kelas kontak dibahas di
  Fase 5, karena kontak hanya ±0,01% piksel bangunan di kedua dataset (`docs/progress.md` Fase 2).
- Kriteria keputusan di `docs/EXPERIMENTS.md` §8 (termasuk ambang "mendekati SOTA" ≤ 3 AP dan aturan pemilihan FULL
  final) dibekukan, lalu dikonfirmasi tanpa perubahan saat Fase 3 dimulai (30 Okt 2026). Sejak itu terkunci.
- Sitasi yang perlu dicek: Dahatonde dkk. (2024) [lengkapi]; Huang & Zhang (2011) halaman/DOI; halaman Pix2Poly
  (README repo menyebut WACV 2025 hlm. 8473–8482, slide menulis 8484–8493); paper deduplikasi Adimoolam dkk.
  (README repo: Oral CVPR 2026, arXiv 2304.02296).

## Referensi kunci

- Zhou dkk. (2018). UNet++: A nested U-Net architecture. DLMIA, LNCS 11045. arXiv:1807.10165
- Iglovikov dkk. (2018). TernausNetV2. CVPRW. arXiv:1806.00844
- Bischke dkk. (2019). Multi-task learning for segmentation of building footprints. arXiv:1709.05932
- Girard dkk. (2021). Polygonal building extraction by frame field learning. CVPR. arXiv:2004.14875
- Xu dkk. (2023). HiSup. ISPRS JPRS 198:284–296. https://doi.org/10.1016/j.isprsjprs.2023.03.006
- Jiao dkk. (2024). PolyR-CNN. ISPRS JPRS 218:33–43. https://doi.org/10.1016/j.isprsjprs.2024.10.006
- Adimoolam dkk. (2025). Pix2Poly. WACV. arXiv:2412.07899 — repo: github.com/yeshwanth95/Pix2Poly
- Adimoolam dkk. (2026). Data leakage detection and de-duplication in large scale geospatial image datasets.
  arXiv:2304.02296 — repo: github.com/yeshwanth95/CrowdAI_Hash_and_search
- Huang & Zhang (2011). Morphological building index (MBI). PE&RS 77(7) [verifikasi halaman]
- Maniyar dkk. (2025). Feature-augmented deep networks for multiscale building segmentation. arXiv:2505.05321
- Ji, Wei & Lu (2019). Dataset WHU. IEEE TGRS 57(1)
- Cheng dkk. (2021). Boundary IoU. CVPR. arXiv:2103.16562 — github.com/bowenc0221/boundary-iou-api
- Avbelj dkk. (2015). PoLiS: a metric for polygon comparison. IEEE GRSL 12(1) [verifikasi]
- Zorzi dkk. (2022). PolyWorld (asal kode C-IoU). CVPR. arXiv:2111.15491

## Singkatan

| Singkatan | Arti |
|---|---|
| AP / AP_boundary | Average Precision COCO berbasis IoU mask / berbasis Boundary IoU |
| B0–B5, B5′, FULL, FULL+Y | kode skenario eksperimen (`docs/EXPERIMENTS.md`) |
| C-IoU | complexity-aware IoU: IoU × (1 − selisih relatif jumlah verteks), per gambar |
| CC | connected component |
| DP | Douglas-Peucker (penyederhanaan kontur) |
| MBI | Morphological Building Index |
| MTA | max tangent angle error (galat sudut kontur, derajat) |
| N1 / N2 / N3 | kontribusi: orientasi dari mask / evaluasi per kepadatan / prior morfologi celah |
| OBB | oriented bounding box |
| P0 / P1 / P2 | ablasi input: RGB / + MBI / + MBI + peta celah |
| PoLiS | jarak rata-rata simetris antara verteks dan batas poligon pasangan (piksel) |
| p-hash | perceptual hash (DCT) untuk deteksi duplikat citra |
| smp | segmentation_models_pytorch |
| θ, τ, r, α, σ, λ, k, ρ, δ | orientasi, ambang marker, radius kontak, bobot tepi, lebar bobot tepi, bobot loss θ, faktor ε DP, ambang persegi, toleransi pelurusan |

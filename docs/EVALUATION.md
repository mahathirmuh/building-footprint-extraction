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
- File GT subset per split dibuat sekali dengan `scripts/prepare_gt.py`: `outputs/gt/<dataset>/<split>.json` (WHU juga
  `<split>_p2p.json`). Isinya semua citra split, termasuk yang tanpa bangunan, dengan aturan `densepoly.eval.coco_io`:
  - id anotasi digeser +1 bila file sumbernya mulai dari 0 (§2.2);
  - sampel negatif WHU dibuang, citranya tetap;
  - anotasi lain dibiarkan apa adanya.

  Field `info` mencatat sumber, offset id, dan jumlah setiap kasus khusus.
- Prediksi diperiksa `coco_io.check_predictions`: secara ketat untuk output kita (setiap pelanggaran aturan di atas
  adalah error), dan longgar untuk prediksi rilisan. Pada mode longgar, hanya ring dengan < 3 titik berbeda yang
  dibuang; masalah lain dihitung dan prediksinya dievaluasi apa adanya.

## 2. Metrik

### 2.1 Daftar

| Metrik | Tingkat | Definisi | Implementasi |
|---|---|---|---|
| AP, AP50, AP75, AP_S/M/L | instance | COCO mask AP, IoU 0,50:0,95 | `pycocotools` COCOeval `segm` |
| AP_boundary | instance | AP dengan IoU := min(IoU mask, Boundary IoU) (dilation ratio 0,02) | `evaluate()` dari COCOeval `boundary` boundary-iou-api (`third_party/boundary_iou/`), lalu `accumulate` kita (§4) |
| IoU, F1 | piksel | gabungan semua poligon vs gabungan GT; utama ΣI/ΣU seluruh split; juga rata-rata per citra | `eval/pixel.py` |
| Boundary IoU | piksel | Boundary IoU mask gabungan (Cheng dkk. 2021: IoU pita batas selebar 2% diagonal citra), rata-rata per citra | `eval/pixel.py` dengan `mask_to_boundary` boundary-iou-api (paket itu tidak punya fungsi `boundary_iou` terpisah) |
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
  dengan N = jumlah verteks ring pertama tiap poligon (titik penutup ikut terhitung bila diulang). Citra yang diiterasi
  adalah `getImgIds(catIds)` dari objek hasil `loadRes`, yaitu hanya citra yang punya prediksi. Jadi **citra tanpa
  prediksi tidak ikut dihitung**: metode yang diam di citra sulit tidak dihukum. `N_GT` di-reset per citra, tetapi
  `mask_gti` tidak. Pada citra yang punya prediksi tetapi tanpa anotasi GT, IoU-nya dihitung terhadap mask GT citra
  sebelumnya (atau `UnboundLocalError` bila itu citra pertama). C-IoU citra itu tetap ≈ 0, karena N_GT = 0 membuat
  faktor verteks ≈ 1e-9/N. Fungsinya juga hanya mencetak, tidak me-return nilai. (Dikoreksi 30 Okt 2026 saat
  vendoring: semula ditulis bahwa `N_GT` juga tidak di-reset.)
  - `CIoU_p2p` = aturan di atas dengan `mask_gti` di-reset per citra. Reset ini hanya mengubah IoU citra berprediksi
    tanpa GT; C-IoU-nya tetap ≈ 0 (selisih < 1e-9, diuji di `tests/test_pix2poly_metrics.py`). Di WHU (opsi R) ada
    patch yang tidak punya anotasi GT sama sekali (test: 227 patch di GT `p2p`, 246 di GT utama; patch dengan > 5
    piksel bangunan yang semua poligonnya terbuang filter, `docs/DATA.md` §3.2). (Koreksi 28 Okt 2026: semula ditulis
    bahwa setiap patch WHU punya minimal satu anotasi.)
  - `CIoU` (utama) = rumus yang sama atas **semua** citra split: tanpa prediksi dan GT tidak kosong → 0; keduanya
    kosong → 1.
- **PoLiS (`polis.py`)**: pencocokan memakai IoU **bounding box** (bukan mask) > 0,5; tiap GT mengambil prediksi
  dengan IoU bbox tertinggi, sehingga satu prediksi bisa dipakai beberapa GT. PoLiS dirata-rata per citra, lalu citra
  bernilai tepat 0 (termasuk citra tanpa pasangan) **dilewati** saat merata-rata antar-citra. Hanya ring pertama yang
  dipakai. Kalau semua citra bernilai 0 (misalnya GT dipakai sebagai prediksi), kode asli membagi dengan nol.
  `compare_polys` menjumlahkan jarak n verteks tetapi membagi dengan 2·(n+1), karena titik penutup ikut dihitung di
  penyebut. Akibatnya nilainya n/(n+1) kali definisi Avbelj dkk. Ini konsisten untuk semua metode, jadi tetap dipakai
  demi kesebandingan dengan angka paper. `evaluateImg` menimpa `segmentation` GT yang kosong menjadi `[[]]` di objek
  COCO yang dipakainya, jadi objek COCO itu tidak boleh dipakai bersama metrik lain.
- **MTA (`angle_eval.py`)**: per citra, semua poligon GT digabung (`unary_union`, lalu `buffer(1e-4)`), begitu juga
  semua poligon prediksi. Kontur prediksi disampel tiap 2 px, diproyeksikan ke kontur GT, lalu sudut tangen maksimum
  dihitung per poligon (hasil gabungan) dan dikumpulkan dari semua citra. Akibatnya, **dinding antar-rumah yang
  bersentuhan hilang** dari GT maupun prediksi, padahal di situlah N1 diklaim. Karena itu klaim N1 memakai `MTA_inst`.
  Fakta tambahan dari menjalankan kode vendor (30 Okt 2026):
  - Nilainya **tidak sebanding** dengan galat sudut sebenarnya. Persegi panjang 40 × 16 px yang diputar 1°, 5°, 10°,
    20°, dan 30° terhadap pusatnya memberi 9,65°, 20,4°, 11,3°, 20,0°, dan 60,0°, karena maksimum per poligon
    didominasi proyeksi di sekitar sudut. Poligon identik memberi 0,0027°, bukan 0, karena GT di-buffer 1e-4.
  - Di shapely 2.2.0 / GEOS 3.14.1, citra yang GT-nya kosong tetapi punya prediksi membuat skrip asli crash
    (`GEOSException` saat interseksi dengan `GEOMETRYCOLLECTION (POLYGON EMPTY)`). Maksud kodenya (presisi 0, jadi
    semua prediksi tersaring) dipertahankan oleh perubahan pembungkus 4.
- **COCOeval**: `evaluate()` menjalankan `p.imgIds = list(np.unique(p.imgIds))`, sehingga duplikat `imgIds`
  dibuang. Ini memengaruhi bootstrap (§4). Selain itu `evaluateImg` menyimpan id GT pasangan di `dtm`, dan
  `accumulate` membaca `dtm` = 0 sebagai tidak ter-match. Akibatnya, deteksi yang cocok dengan GT ber-`id` 0 dihitung
  positif palsu (dicek di `cocoeval.py` pycocotools 2.0.11). Id anotasi WHU hasil konversi Pix2Poly mulai dari 0,
  begitu juga GT val CrowdAI (515.364 anotasi; dicek 30 Okt 2026).
- **`loadRes`** menyimpan poligon prediksi apa adanya hanya bila setiap hasil punya `bbox`. Tanpa `bbox`,
  `segmentation` dianggap RLE dan `maskUtils.area` gagal. Dengan `bbox`, `area` deteksi = luas bbox (bukan luas
  poligon), sehingga pengelompokan AP_S/M/L untuk deteksi memakai luas bbox. Poligon dengan tepat 4 koordinat dibaca
  `frPyObjects` sebagai bbox lalu crash, dan ring berluas nol mendapat IoU 0 dengan dirinya sendiri.
- **GT val CrowdAI** punya 3.644 anotasi ber-`area` 0. Himpunannya persis sama dengan anotasi yang mask
  pycocotools-nya kosong: 3.488 ring berluas nol, sisanya ring setipis ≤ 1 px² (dicek 30 Okt 2026 dari file
  `scripts/prepare_gt.py`). Jadi field `area` konsisten dengan mask, dan tidak ada bangunan nyata yang salah masuk
  kelompok AP_S.
- **Anotasi negatif WHU** (`segmentation: []`): `mask.frPyObjects([], h, w)` pycocotools melempar `IndexError`
  (dicek 28 Okt 2026). Di kode vendor (30 Okt 2026): `cIoU.py` menganggapnya mask kosong (N_GT bertambah 0),
  `polis.py` menimpanya menjadi `[[]]` lalu tidak pernah memasangkannya, dan `angle_eval.py` tidak membuat poligon
  darinya. Jadi membuang anotasi negatif (citranya tetap) memberi angka yang sama untuk ketiga metrik vendor, dan wajib
  untuk pycocotools.

### 2.3 Aturan vendor
- Salin `polis.py`, `angle_eval.py`, `cIoU.py` dari `Pix2Poly/eval/hisup_eval_utils/metrics/` (MIT) ke
  `third_party/pix2poly_metrics/`. Jangan memakai `HiSup/hisup/utils/metrics/cIoU.py` (variabel `mask_gti` tidak
  terdefinisi dan ada typo `img['weight']`).
- **Boleh** (perubahan pembungkus yang tidak mengubah matematika metrik): me-return nilai; mengeluarkan nilai per
  citra (untuk PoLiS: rata-rata citra dan status valid; untuk MTA: jumlah dan banyaknya nilai per citra); menerima
  subset `imgIds`; menangani pembagian nol dengan mengembalikan `None` plus jumlah citra valid; menghapus impor
  `descartes`; me-reset `mask_gti` per citra (`N_GT` sudah di-reset di kode asli). Ditambahkan saat vendoring
  (30 Okt 2026), dengan alasan di header tiap file: melewati teks progres `cIoU.py` yang menghitung ulang empat
  rata-rata setiap citra (kuadratik); mengganti debugger `ipdb` dengan `raise`; dan di MTA, mengembalikan hasil kosong
  untuk citra yang GT-nya kosong (setara maksud kode asli, yang crash di GEOS 3.14).
- **Tidak boleh:** mengubah pencocokan, ambang, sampling kontur, buffer, atau cara pooling.
- Setiap perubahan dicatat di header file dan di `docs/progress.md`. Tes regresi: pada file contoh, versi vendor
  dan versi asli memberi angka identik untuk semua kasus yang tidak terkena perubahan.

### 2.4 Pencocokan satu-satu (dasar metrik `*_paired`, `*_inst`, recall, Vertex F1, galat θ)
Seperti COCO: urutkan prediksi menurut skor; tiap prediksi dipasangkan dengan GT belum terpakai yang punya **IoU mask**
tertinggi, asalkan IoU ≥ 0,5. Untuk membandingkan dua metode A dan B, metrik berpasangan dihitung hanya pada GT yang
ter-match di **kedua** metode, sehingga keduanya diukur pada bangunan yang sama.

Rincian implementasi (`eval/instance.py`, 30 Okt 2026):
- Berbeda dari COCOeval (maxDets 100), tidak ada batas jumlah prediksi per citra.
- IoU yang sama persis dimenangkan GT yang lebih dulu.
- GT bermask kosong hanya mengurangi recall. Prediksi bermask kosong (hanya mungkin lewat pemeriksaan longgar) tidak
  ikut merge/split.
- `MTA_inst` per pasangan = sudut terbesar dari `compute_contour_metrics` vendor yang dijalankan pada pasangan itu
  saja. Bila vendor tidak memberi nilai (semua tepi tersaring), pasangan itu tidak ikut dirata-rata dan jumlahnya
  dicatat.
- Vertex F1 = 2·Σ verteks cocok / (Σ verteks GT + Σ verteks prediksi) atas semua pasangan, tanpa titik penutup.
- Galat θ memakai `theta_deg` yang disimpan metode bersama poligonnya. Bila metode tidak menyimpannya (B4, HiSup),
  nilainya `null`. θ_GT memakai aturan label (sisi panjang `minAreaRect`).

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
    Implementasi (`eval/stats.py`, 30 Okt 2026): `evalImgs` diindeks ulang untuk daftar citra itu, lalu `accumulate()`
    dan `summarize()` asli pycocotools dijalankan tanpa diubah. Hasilnya identik dengan COCOeval pada semua citra
    maupun subset, dan citra ganda setara dengan salinannya (diuji). AP_boundary memakai cara yang sama atas `evalImgs`
    boundary-iou-api.
  - Metrik per citra dan berpasangan: rata-rata ulang dari nilai per citra/per pasangan yang tersimpan (§3).
- **Δpadat − Δjarang:** bootstrap berstrata, yaitu citra disampel ulang di dalam tiap strata secara terpisah, lalu
  dihitung (Δ di padat) − (Δ di jarang).
- Implementasi (`eval/stats.py`, 30 Okt 2026):
  - Satu replikasi disimpan sebagai vektor "berapa kali tiap citra terambil". Matriks bobot yang sama dipakai untuk
    semua metode, seed, dan metrik.
  - Metrik rasio = Σ bobot · pembilang per citra / Σ bobot · penyebut per citra, diambil dari tabel `run_eval`. Metrik
    berpasangan hanya memakai GT yang ter-match di kedua run (per pasangan seed).
  - `compare` mengembalikan Δ per seed, rata-ratanya, CI 95% persentil, dan jumlah replikasi yang penyebutnya kosong.
    `compare_strata` melakukan hal yang sama untuk Δpadat − Δjarang.
  - Waktu di laptop ini, CrowdAI test: 1.000 replikasi metrik rasio < 0,1 s; AP ±55 ms per replikasi (±25 ms di
    strata padat), sehingga satu perbandingan AP 3 seed × 2 metode makan ±5,5 menit.
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

Implementasi (`eval/run_eval.py`, `scripts/evaluate.py`; 30 Okt 2026):
- Nama file: `<split>[_<variant>].json`. Di sebelahnya ada `<split>[_<variant>].per_image.parquet`, `.pairs.parquet`,
  dan `.evalimgs.pkl` (cache AP dan AP_boundary; tidak dibuat untuk val penuh).
- Field tambahan:
  - di tingkat atas: `variant`, `gt_file`, `pred_file`, `pred_check` (mode ketat/longgar dan jumlah tiap masalah),
    `coco_stats` (12 angka COCO untuk segm dan boundary), dan `seconds`;
  - di `overall` dan tiap strata: `N_ratio_p2p`, `MTA_values`, `n_images_with_pred`, `n_gt`, `n_pairs`, dan
    `n_pairs_without_mta`.
- `per_stratum` memuat strata utama (`jarang`, `sedang`, `padat`, `kosong`) dan indeks sekunder (`bersentuhan`,
  `tidak_bersentuhan`).
- Satuan: AP, IoU, C-IoU, dan semua rate berupa pecahan [0, 1]; PoLiS dalam piksel; MTA dan galat θ dalam derajat.
- Kolom `per_image.parquet`: `image_id`, `stratum`, `stratum_secondary`, `n_pred`, `n_gt`, `gt_px`, `pred_px`,
  `inter_px`, `iou`, `ciou`, `ciou_p2p`, `n_vertices`, `n_vertices_gt`, `biou`, `polis_image` (0 = dilewati skrip
  vendor), `mta_sum`, `mta_count`, `n_matched`, `n_merged`, `n_split`.
- Kolom `pairs.parquet`: `image_id`, `gt_id`, `pred_index`, `score`, `iou`, `polis`, `mta`, `theta_err`,
  `vertices_gt`, `vertices_pred`, `vertex_tp`.
- Metrik vendor per strata = nilai per citra yang di-pool ulang dengan aturan skripnya. Di semua citra, hasil pooling
  ulang itu diperiksa otomatis terhadap angka skrip vendor sendiri.

## 6. Uji kewarasan (wajib lolos sebelum angka apa pun dilaporkan)

### 6.1 GT sebagai prediksi (Fase 3)
Ubah GT menjadi file prediksi (skor 1). Harus menghasilkan: AP = 100 (±0,1); AP_boundary = 100; `CIoU` = `CIoU_p2p`
= 1; `PoLiS_paired` = 0; `MTA_inst` ≈ 0 dan `MTA` ≤ 0,5° (GT di-buffer 1e-4); `PoLiS` vendor tidak terdefinisi
(semua citra 0), sehingga pembungkus mengembalikan `None` dengan `PoLiS_valid_images = 0`, bukan error;
`merge_rate` dan `split_rate` = 0 kecuali di tempat poligon GT saling tumpang tindih (laporkan jumlahnya).

**GT bermask kosong** di CrowdAI mendapat IoU 0 bahkan dengan dirinya sendiri di pycocotools, jadi tidak bisa
dicocokkan oleh metode apa pun. Yang termasuk: ring berluas nol, ditambah ring setipis ≤ 2 px² yang tidak menutup satu
piksel pun (`coco_io.empty_mask_ids`). Jumlahnya: test 181 (0,72% anotasi), `val_clean` 17, val penuh 3.644 (0,71%);
WHU 0. GT **tidak diubah** (protokol resmi, sama dengan evaluasi HiSup; diputuskan 30 Okt 2026). Karena itu uji ini
dijalankan dua kali:
- tanpa GT bermask kosong (di GT maupun prediksi): semua target di atas berlaku, termasuk AP = 100 (±0,1);
- dengan GT apa adanya: jumlah GT yang tidak ter-match harus sama dengan jumlah GT bermask kosong, dan semua GT lain
  ter-match dengan IoU 1. Selisih AP dari 100 dilaporkan.

Tes sintetis:
- Persegi sejajar sumbu yang digeser d px searah x → `PoLiS` = `PoLiS_paired` = 0,4·d. Dua verteks di tiap sisi
  berjarak d, sehingga totalnya 4d; dibagi 2·5 karena titik penutup ikut di penyebut (§2.2). Definisi asli memberi d/2.
- Poligon identik → `MTA_inst` ≈ 0 (< 0,01°; GT di-buffer 1e-4). Diputar α ∈ {5°, 10°, 20°} terhadap pusatnya →
  `MTA_inst` ≥ α − 0,5°.

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
`annotation-small.json`), dan pemakaian skor. Setelah lolos, jalankan di `crowdai/val_clean` (total + per strata).
Prediksi HiSup hanya ada untuk val, dan `test` kita diambil dari train yang dipakai melatih HiSup. (Dikoreksi 30 Okt
2026; semula `crowdai/test`, sisa dari sebelum opsi E.)

### 6.5 (Opsional) Reproduksi Pix2Poly di WHU (Fase 8)
Bila checkpoint atau prediksi rilisan dipakai, `CIoU_p2p` test terhadap GT `p2p` harus mendekati 81,63. Test kita
direkonstruksi dari rilisan resmi (3.943 tile 512², Pix2Poly 3.848; `docs/DATA.md` §3.2), jadi selisih kecil tidak
berarti pipeline salah. Catat selisihnya dan jangan menyetel apa pun agar cocok.

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

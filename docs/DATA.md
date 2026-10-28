# DATA.md — Dataset, Deduplikasi, Split, dan Strata

## 1. Unduhan (dilakukan manual oleh saya) dan struktur folder

| Data | Sumber | Catatan |
|---|---|---|
| CrowdAI Mapping Challenge | https://www.aicrowd.com/challenges/mapping-challenge/dataset_files | perlu akun AIcrowd; `train.tar.gz`, `val.tar.gz` |
| Prediksi HiSup untuk val CrowdAI | Google Drive di README HiSup (bagian *Evaluation*) | file JSON format COCO |
| WHU Building (aerial) | http://gpcv.whu.edu.cn/data/building_dataset.html | lihat §3 untuk versi yang diunduh |
| Checkpoint Pix2Poly (opsional) | Google Drive / MEGA di README Pix2Poly | untuk memastikan protokol WHU |
| SpaceNet 6 Rotterdam (opsional) | AWS Open Data (SpaceNet) | CC BY-SA 4.0 |

Struktur di `$DENSEPOLY_DATA` (default `./data`):

```
data/
├── crowdai/
│   ├── train/images/*.jpg   train/annotation.json
│   ├── val/images/*.jpg     val/annotation.json
│   ├── hisup_release/*.json            # prediksi HiSup (val penuh)
│   └── hashes/{train,val}_aug.json     # cache p-hash 6 varian (dibuat skrip)
├── whu/
│   ├── raw_0.2/{train,val,test}/{image,label}/   # protokol A (nama folder mengikuti Pix2Poly)
│   ├── pix2poly_224/{train,val,test}/{images/,annotation.json}
│   └── raw_0.3/{train,val,test}/{image,label}/   # protokol B (opsional)
├── instances/<dataset>/<split>/<image_id>.png    # cache peta instance uint16 (dibuat skrip)
└── priors/<dataset>/<split>/<image_id>.png       # cache prior: R = MBI, G = celah, B = 0 (uint8)
```

## 2. CrowdAI

### 2.1 Fakta terverifikasi (28 Okt 2026, dari file di repo `CrowdAI_Hash_and_search`)

| Hal | Jumlah |
|---|---|
| Citra train | 280.741 |
| Citra val | 60.317 |
| p-hash unik di train (tanpa augmentasi) | 114.548 |
| Duplikat persis atau teraugmentasi di train (README repo) | ±250 rb → ±30 rb citra unik |
| Val tanpa duplikat persis di train (`val_in_train_np.json`) | 3.949 (±93,5% val bocor) |

Repo menyediakan `train_hashes/train_hashes.json` (format `[{image_name, hash}]`, p-hash 64-bit dalam hex) dan
`val_in_train_np.json` (format `[{val_image_name, val_image_hash, duplicates_in_train: [...]}]`). Pencocokan di repo
itu adalah **kesamaan persis** string p-hash (`imagehash.phash`), bukan jarak Hamming.

### 2.2 Deduplikasi (`scripts/dedup_crowdai.py`)
Implementasi sendiri (lebih mudah diuji), mengikuti logika repo tersebut, dengan atribusi.
1. Untuk setiap citra train dan val, hitung `imagehash.phash` dari 6 varian: asli, rotasi 90/180/270, flip
   horizontal, flip vertikal (sama dengan `compute_hashes_augmented.py`). Pakai multiprocess; cache ke
   `data/crowdai/hashes/`.
2. **Klaster train:** union-find, dua citra satu klaster bila salah satu hash variannya sama. Wakil klaster = nama
   file terkecil (deterministik). Simpan anggota tiap klaster (`splits/crowdai_train_clusters.json`).
3. **val_tune:** acak klaster (seed 0), ambil 10% klaster → wakilnya menjadi `val_tune`; sisanya `train_dedup`.
   Lalu `val_tune` dibagi tetap (seed 0): `val_tune_sub` = 1.000 citra untuk tuning post-processing, `val_select` =
   sisanya untuk memilih model (`docs/EXPERIMENTS.md` §4).
4. **Test bersih (`crowdai_clean`):** citra val yang **tidak satu pun** dari 6 hash-nya muncul di himpunan hash train
   (semua varian). Lalu klaster val bersih dengan cara yang sama dan ambil satu wakil per klaster. Hasilnya
   diharapkan ≤ 3.949, karena pencocokan kita juga menangkap duplikat teraugmentasi.
5. `splits/dedup_report.json`: jumlah per langkah, ukuran klaster, dan perbandingan dengan 3.949.
   Bila hasil kita **lebih dari** 3.949, kemungkinan p-hash kita berbeda dari hash repo (versi `imagehash` atau Pillow
   berbeda). Bandingkan hash asli kita dengan `train_hashes.json` untuk 100 citra sebelum melanjutkan.
6. **Cek manual:** gambar 50 klaster acak (grid thumbnail) → `outputs/figures/qa/dedup_clusters.png`. Waspadai
   tabrakan hash pada tile yang hampir seragam (misalnya tile vegetasi tanpa bangunan). Bila ditemukan, tambahkan
   verifikasi piksel (MSE citra 32 × 32 setelah diselaraskan dengan augmentasinya) dan catat keputusannya.

**Tes wajib:** tidak ada nama citra dan tidak ada klaster di dua split; varian augmentasi buatan dari satu citra
fixture masuk satu klaster.

### 2.3 Rasterisasi anotasi
- `pycocotools` `annToMask`; gambar urut luas menurun sehingga bangunan kecil yang tumpang tindih tidak hilang.
- Lewati anotasi dengan `segmentation` kosong atau kurang dari 3 titik; catat jumlahnya.
- Cek konvensi poligon GT: apakah titik pertama diulang di akhir, dan bagaimana arah putarannya. Output kita
  mengikuti konvensi itu (memengaruhi jumlah verteks pada C-IoU dan N-ratio).
- Simpan peta instance ke cache `uint16` PNG agar loader di Colab cepat.

### 2.4 Ukuran tile
300 × 300 di-pad ke 320 × 320 (citra: reflect; label: 255). Prediksi di-crop kembali ke 300 sebelum post-processing.
Tidak ada resize.

## 3. WHU Building (aerial)

### 3.1 Versi yang tersedia (halaman resmi, dicek 28 Okt 2026)

| Versi | Isi | Catatan |
|---|---|---|
| 0,3 m | 8.189 tile 512 × 512, label raster | split per area: train 4.736, val 1.036, test 2.416 (sesuai slide; jumlahnya 8.188, halaman resmi menyebut 8.189) |
| 0,2 m raster | tile dan label raster (`whu_raster_1024`) | dipakai Pix2Poly (`WHU_aerial_0.2`) |
| 0,2 m COCO | `train.zip`, `validation.zip`, `test.zip`, `annotation.zip` | format COCO resmi |
| Asli 0,075 m | satu citra besar + shapefile seluruh area | instance vektor sebenarnya |

Halaman resmi tidak menyebut apakah tile punya georeferensi; yang disebut hanya bahwa citra asli bisa disusun ulang
dari urutan pemotongan.

### 3.2 Protokol A (default): mengikuti Pix2Poly
Angka WHU di slide (HiSup C-IoU 79,62; Pix2Poly 81,63; FFL 32,19 vs 54,52) diukur Pix2Poly dengan protokol ini.
Hasil pembacaan `data_preprocess/whu_buildings_to_coco.py`:
- Input `WHU_aerial_0.2/{split}/image` dan `label` (raster 0,2 m).
- Dipotong menjadi patch **224 × 224 dengan overlap 34 px**.
- GT dibuat dari mask biner: `skimage.measure.label` (CC) → `cv2.findContours` (`RETR_TREE`) → `shapely.simplify(1.8,
  preserve_topology=False)`; poligon dengan lebar atau tinggi bbox ≤ 5 px dibuang.
- Patch tanpa bangunan tetap dimasukkan sebagai sampel negatif (anotasi dengan `segmentation` kosong);
  `category_id = 100`.
- Angka di paper diukur di split **test**, dengan `evaluation.py` Pix2Poly (`--eval-type ciou|polis|angle|coco_iou`).

Konsekuensi untuk kita:
1. `scripts/prepare_whu_pix2poly.py` menjalankan logika yang sama tanpa diubah (atribusi MIT).
2. Model WHU dilatih dari **raster poligon COCO hasil konversi** (bukan dari label raster asli), supaya konvensi GT
   konsisten (poligon GT dari `findContours` sedikit menyusut, dan model ikut mempelajari itu).
3. GT WHU berasal dari CC, jadi bangunan yang bersentuhan menjadi satu instance. WHU dipakai untuk banding
   SOTA, **bukan** untuk klaim pemisahan rumah berdempetan. Klaim itu diuji di CrowdAI.
4. `val_tune` WHU = split val Pix2Poly.

### 3.3 Protokol B (alternatif): tile 0,3 m seperti slide
Instance dari CC; tidak bisa dibandingkan langsung dengan angka C-IoU di paper. Kalau dipilih, banding SOTA di WHU
harus menjalankan ulang checkpoint pembanding, atau tidak dilakukan sama sekali.

### 3.4 Analisis opsional: COCO resmi WHU 0,2 m
Hitung berapa pasang poligon GT yang bersinggungan (`touches` atau jarak < 1 px). Kalau bangunan berdempetan memang
terpisah di anotasi resmi, set ini bisa menjadi uji tambahan untuk pemisahan rumah di WHU.

## 4. Strata kepadatan (kontribusi N2)

- **Indeks utama:** `n` = jumlah instance GT non-kosong per tile.
- **Ambang:** tertil `q33` dan `q67` dari `n` pada tile test dengan `n ≥ 1`. Strata: `jarang` (n ≤ q33),
  `sedang` (q33 < n ≤ q67), `padat` (n > q67), `kosong` (n = 0). Karena `n` bilangan bulat, ukuran strata bisa tidak
  persis sama; laporkan ukurannya.
- Ambang yang sama dipakai untuk `val_tune` (definisi identik).
- **Indeks sekunder (analisis ketahanan):** rasio bersentuhan = fraksi instance yang punya piksel kontak (§3.2
  METHOD, r = 2). Perbandingan utama diulang dengan tertil indeks ini dan dilaporkan di lampiran.
- Strata `kosong` tidak masuk AP/PoLiS/MTA (tidak terdefinisi), tetapi masuk metrik piksel dan hitungan positif palsu.
- File `splits/density_<dataset>_<split>.json` berisi definisi, ambang, ukuran strata, daftar `image_id` per strata, tanggal,
  dan hash git. Dibuat **sebelum** evaluasi model mana pun, lalu dibekukan. Ambang dihitung hanya dari jumlah GT,
  tidak dari output model.

## 5. Cache dan statistik

- Prior: satu PNG per tile (R = MBI, G = celah). Mudah dilihat dan cukup satu file per tile.
- Statistik prior di `configs/data/<ds>_prior_stats.json`: `q99.5` (skala), serta mean dan std setelah diskalakan,
  dihitung dari 500 tile train acak (seed 0).
- Laporkan waktu hitung prior per tile (CPU) dan total; dipakai untuk metrik efisiensi. Saat dokumen disusun: ±1,4 s
  per tile 320² (MBI) + ±20 ms (celah), atau ±14 jam-inti untuk ±35 rb tile. Tempat menjalankan: `docs/COLAB.md` §8.

## 6. Format `splits/`

| File | Isi |
|---|---|
| `crowdai_train_dedup.txt` | satu nama file per baris |
| `crowdai_val_tune.txt` | satu nama file per baris (gabungan dua file di bawah) |
| `crowdai_val_tune_sub.txt` | 1.000 citra untuk tuning post-processing |
| `crowdai_val_select.txt` | sisa `val_tune`, untuk pemilihan model |
| `crowdai_clean_test.txt` | satu nama file per baris |
| `crowdai_train_clusters.json` | `{wakil: [anggota...]}` |
| `dedup_report.json` | jumlah per langkah dan perbandingan |
| `density_<dataset>_<split>.json` | definisi, ambang, strata → daftar `image_id` |

## 7. Lisensi dan atribusi

- Kode yang diadaptasi: Pix2Poly (MIT), HiSup (MIT), CrowdAI_Hash_and_search (MIT). Cantumkan pemberitahuan
  lisensi di header file yang diadaptasi dan di `third_party/`.
- boundary-iou-api: cek lisensinya saat instalasi dan catat.
- Ultralytics: AGPL-3.0. Pisahkan sebagai dependensi opsional `[yolo]`.
- Data: sitasi Ji, Wei & Lu (2019) untuk WHU; cek syarat penggunaan CrowdAI di halaman AIcrowd; SpaceNet 6 di bawah
  CC BY-SA 4.0.

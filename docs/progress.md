# progress.md — Log Kerja

Tambahkan satu bagian setiap fase selesai. Format: apa yang dibuat, angka penting (dengan path JSON sumbernya),
masalah, keputusan, dan pertanyaan terbuka.

---

## Fase desain (ETS) — selesai 28 Oktober 2026

**Dibuat:** desain metode, slide ETS (24 slide), kode demo di `reference/demo/`, prototipe di `reference/prototypes/`,
dan dokumen Claude Code ini. Dokumen ditinjau ulang oleh agen terpisah; temuannya diterapkan (lihat bawah).

**Diverifikasi saat menyusun dokumen** (detail di `CLAUDE.md`, bagian Status, dan `docs/EVALUATION.md` §2.2):
- CrowdAI: 280.741 train, 60.317 val, 114.548 p-hash unik di train, 3.949 val tanpa duplikat persis di train.
- HiSup merilis prediksi val CrowdAI → pembanding cukup dievaluasi ulang dari file prediksi.
- Protokol WHU Pix2Poly: 0,2 m, patch 224 overlap 34, GT dari CC + `findContours` + simplify 1,8.
- Skrip metrik:
  - `HiSup/.../cIoU.py` rusak.
  - C-IoU Pix2Poly hanya menghitung citra yang punya prediksi, dan `mask_gti`/`N_GT` basi pada citra berprediksi
    tanpa GT.
  - PoLiS Pix2Poly mencocokkan via IoU bbox, melewati citra bernilai 0, dan membagi dengan n+1.
  - MTA menggabungkan poligon per citra.
  - `COCOeval` membuang `imgIds` duplikat.
- smp 0.6.0 mengubah bobot RGB bila `in_channels>3` → pakai perluasan dengan bobot nol.
- Marching squares + offset memberi IoU bolak-balik 1,000; indeks `cv2.findContours` apa adanya 0,84 (0,81 untuk
  rumah 10 × 10).
- Prototipe regularisasi: kotak dari bounding box hanya IoU ±0,89 pada tepi berderau; kotak dari momen orde-2
  0,98–0,99 (kontur mentah 0,95–0,97). Bentuk tanpa derau: L/T/dinding miring tepat 6/8/5 verteks di 8 sudut rotasi.
- Waktu CPU (sandbox penyusun dokumen, 1 inti): MBI by reconstruction ±1,4 s/tile 320², peta celah ±20 ms/tile,
  p-hash 6 varian ±6,4 ms/citra.

**Keputusan baru dibanding slide:**
- B5′ ditambahkan untuk memisahkan efek pelatihan θ dan efek pemanduan θ. Uji N1 seluruhnya memakai input RGB
  (B5 vs B5′ vs FULL-P0), supaya prior N3 tidak ikut tercampur.
- Klaim N1 memakai metrik berpasangan per instance (`PoLiS_paired`, `MTA_inst`), karena MTA vendor menggabungkan
  rumah yang bersentuhan. Klaim N3 memakai satu metrik utama (`merge_rate`, strata padat); atribusi ke peta celah
  butuh FULL-P2 vs FULL-P1.
- C-IoU dilaporkan dua varian: `CIoU` (semua citra) dan `CIoU_p2p` (aturan Pix2Poly).
- Bootstrap AP memakai `evalImgs` yang di-cache, bukan `imgIds` duplikat.
- Kriteria keputusan dibekukan (`docs/EXPERIMENTS.md` §8): "mendekati SOTA" ≤ 3 AP; FULL final dipilih dari AP di
  `val_select`; "efek terbesar di padat" diuji langsung lewat Δpadat − Δjarang.
- Kontak didefinisikan hanya pada piksel bangunan (slide: "dilasi ≥ 2 bangunan"; demo ETS memakai r = 3 dan ikut
  mencakup celah latar).
- Jalur kotak memakai momen orde-2 dan ambang ρ berupa IoU kotak vs kontur, default 0,90 (demo ETS: rasio luas
  terhadap bounding box 0,75), disetel di `val_tune_sub`. Sisi miring pendek dan sisi sangat pendek dilebur saat
  pelurusan.
- Urutan run: Tier 1 (B4, FULL-P0, FULL-P2 × 3 seed) lebih dulu; B1–B3 dan FULL-P1 di Tier 2.
- Metrik tambahan: merge rate, split rate, recall instance.

**Pertanyaan terbuka:** protokol WHU (A/B), definisi kepadatan final, sitasi yang perlu dicek (lihat `CLAUDE.md`).

---

## Klarifikasi spesifikasi — 28 Oktober 2026 (sebelum Fase 0)

**Dibuat:** isi paket `densepoly-claude-code/` dipindah ke root repo; salinan ganda `CLAUDE.md` dan `PLAN.md` dihapus
setelah dicek identik. Repo git dibuat dan terhubung ke `github.com/mahathirmuh/building-footprint-extraction`.
Commit pertama berisi dokumen desain apa adanya; perubahan di bawah ada di commit terpisah.

**Keputusan** (disetujui 28 Okt 2026; perubahan §8 dilakukan sebelum Fase 3 sesuai aturannya):
- **Kandidat FULL final = FULL-P0 dan FULL-P2** (`docs/EXPERIMENTS.md` §8 poin 6). Alasan: FULL final dipilih di
  Fase 6 dan dipakai untuk WHU, SOTA, dan FULL+Y, sedangkan FULL-P1 baru dilatih di Fase 7 (Tier 2, bisa batal bila
  kuota habis). FULL-P1 tetap dipakai untuk atribusi prior.
- **λ ditetapkan sebelum FULL seed 0** dari run pendek, lalu sama untuk semua seed (`PLAN.md` Fase 5–6,
  `docs/METHOD.md` §10). Alasan: aturan lama mengecek λ dari log seed 0 lalu boleh mengubahnya untuk seed 1–2,
  sehingga ketiga seed bisa memakai config berbeda.
- **Seleksi checkpoint memakai `val_select`**, bukan seluruh `val_tune` (`docs/EXPERIMENTS.md` §3–4). Alasan:
  menyamakan dengan aturan `CLAUDE.md` ("pemilihan model memakai `val_select`"); `val_tune_sub` hanya untuk tuning
  post-processing.

**Pertanyaan terbuka:**
- Ambang "F1 kontak rendah" untuk menaikkan bobot kelas kontak (`docs/METHOD.md` §5): ditetapkan sebelum Fase 5.
- Pembagian `val_tune_sub`/`val_select` untuk WHU: ditetapkan bersama protokol WHU di Fase 1 (`docs/DATA.md` §3.2).

---

## Fase 0 — Setup repo (selesai dan disetujui 28 Oktober 2026)

**Dibuat:**
- Kerangka folder sesuai `CLAUDE.md`. `pyproject.toml` berisi paket inti dan extras `[dev]`, `[eval]`, `[yolo]`;
  `torch`/`timm` tidak dipin supaya torch bawaan Colab tidak di-install ulang. `.gitignore` memakai `/data/` dan
  `/outputs/` (berjangkar ke root) supaya subpaket `src/densepoly/data/` tetap ikut git. `.gitattributes`: LF untuk
  semua file teks; filter nbstripout terpasang dan teruji (output dan `execution_count` notebook dibuang).
- `src/densepoly/paths.py`: `DENSEPOLY_DATA`/`DENSEPOLY_OUT` (default `<repo>/data`, `<repo>/outputs`) dan path baku
  cache instance/prior, run, prediksi, metrik, GT (`docs/DATA.md` §1, `docs/EVALUATION.md` §1). Nama dataset atau
  split yang tidak dikenal ditolak.
- `src/densepoly/config.py` + `configs/`: gabungan base ← data ← exp ← CLI. `base.yaml` berperan sebagai skema, jadi
  key salah ketik (di YAML maupun CLI), override tanpa `=`, dan nilai `???` yang belum diisi langsung error. Config
  data `crowdai`/`whu` (parameter piksel `docs/METHOD.md` §10) dan 11 config eksperimen (matriks
  `docs/EXPERIMENTS.md` §2).
- `tests/fixtures/synthetic.py`: adegan sintetis 300 × 300 ber-seed. Poligon COCO adalah sumber kebenaran; peta
  instance = rasterisasi pycocotools (urut luas menurun); RGB digambar dari peta instance. Isinya 2–3 rumah persegi
  diputar 5–85°, deret 3–5 rumah bersentuhan (diputar; dinding bersama memakai verteks identik), satu rumah L, deret
  3–4 rumah bercelah gelap 1–2 px, dan latar bertekstur. Tersedia ekspor ke format COCO.
- `tests/conftest.py`: koneksi internet diblokir selama tes.

**Angka penting** (output pytest/ruff, bukan metrik model): `pytest -q` → 124 tes lolos (2,8 s, CPU, offline);
`ruff check` dan `ruff format --check` bersih.

**Lingkungan lokal:** Windows 11, Python 3.12.0 (`.venv`), torch 2.14.1+cpu, smp 0.6.0, timm 1.0.30,
opencv-python-headless 4.14.0, scikit-image 0.26.0, scipy 1.18.1, shapely 2.2.0, pycocotools 2.0.11, numpy 2.5.4,
omegaconf 2.4.0, pandas 3.1.0.

**Temuan dan masalah:**
- Nama distribusi boundary-iou-api adalah `boundary_iou` (dari `setup.py`-nya), jadi requirement `[eval]` memakai nama
  itu, di-pin ke commit `37d2558`. Paket ini menarik `fvcore`, `cityscapesscripts`, dan `panopticapi`; instalasinya
  diuji di Fase 3.
- pycocotools 2.0.11 dengan numpy 2.5 memunculkan `DeprecationWarning` (`__array__` copy keyword) di `mask.decode`.
  Hasil tidak terpengaruh; peringatan itu disaring di konfigurasi pytest. Pantau bila numpy di-upgrade.
- `per_image.parquet` (`docs/EVALUATION.md` §5) butuh `pyarrow`, yang belum ada di daftar paket. Tambahkan di Fase 3.
- Tes fixture membuktikan pycocotools membagi piksel dinding bersama tanpa tumpang tindih dan tanpa celah, juga pada
  deret yang diputar, asalkan verteks dindingnya identik.
- Commit `6467e5a` kehilangan baris judul karena salah quoting di shell; isinya benar dan tidak di-force-push.

**Keputusan:**
- Default path data/output berjangkar ke root repo, bukan folder kerja (`CLAUDE.md` menulis `./data`).
- `pytest` ada di extra `[dev]` sesuai `PLAN.md` Fase 0 (`CLAUDE.md` mencantumkannya di daftar paket inti).
- Poligon fixture tanpa titik penutup; konvensi final mengikuti GT CrowdAI (dicek di Fase 1).

**Pertanyaan terbuka:** ambang bobot kelas kontak dan pembagian val WHU (bagian klarifikasi di atas), protokol WHU,
dan definisi kepadatan (Fase 1).

---

## Fase 1 — Data (selesai dan disetujui 29 Oktober 2026)

**Ringkasan** (rincian di bawah; `pytest -q` → 197 tes lolos):

- **CrowdAI v1.0:**
  - Unduhan lengkap: 280.741 + 60.317 citra.
  - Hash identik dengan rilisan paper deduplikasi. Angka 114.548 dan 3.949 direproduksi persis.
  - Split final (opsi E): `train_dedup` 23.427, `val_tune` 2.928 (1.000 + 1.928), `test` 2.928 (holdout klaster
    train), dan `val_clean` 212 (val bersih, untuk banding HiSup).
  - Strata dibekukan: ambang 6 dan 11 bangunan per tile; indeks sekunder bersentuhan vs tidak.
- **WHU 0,2 m:**
  - Rilisan resmi berupa tile 1024², sedangkan Pix2Poly memakai tile 512² yang test-nya tidak bisa direkonstruksi
    persis.
  - Opsi R: kuadran 512² berisi bangunan; test hanya bagian `300`.
  - GT utama dari bangunan resmi, yang memisahkan rumah berdempetan, ditambah GT `p2p` untuk banding angka paper.
  - Patch: 84.762 / 13.761 / 35.487.
  - Strata dibekukan: ambang 3 dan 6 bangunan per patch.
- **Keputusan selama Fase 1:**
  - CrowdAI tetap v1.0, bukan v2.0 (opsi A).
  - Test CrowdAI = holdout klaster train, karena val bersih hanya 212 tile (opsi E; keputusan #7 dan §8.5 diubah).
  - Indeks kepadatan sekunder dibuat biner, karena 95% tile CrowdAI tidak punya bangunan bersentuhan.
  - WHU memakai opsi R (keputusan #11 diubah).
- **Temuan untuk fase berikut:**
  - Bangunan bersentuhan jarang di CrowdAI: hanya 1,3% bangunan berjarak ≤ 2 px. Di WHU angkanya 5,7% (r = 3).
    Ini penting untuk radius kontak (Fase 2) dan bobot kelas kontak (Fase 5).
  - Poligon GT CrowdAI mengulang titik pertama.
  - Ring GT berluas nol (±0,7%) dan GT id 0 di WHU perlu ditangani di Fase 3.

**Dibuat** (diuji di data sintetis):
- `src/densepoly/data/crowdai.py`: rasterisasi peta instance (urut luas menurun, identik dengan fixture), statistik
  anotasi (citra tanpa anotasi, ring tidak valid, multi-ring, piksel tumpang tindih, anotasi tertutup penuh), cek
  konvensi titik penutup dan arah putaran, cek file di anotasi vs di disk.
- `src/densepoly/data/dedup.py` + `scripts/dedup_crowdai.py`: p-hash 6 varian (cache per 5.000 citra, bisa
  dilanjutkan bila terputus), union-find, split `train_dedup`/`val_tune`/`val_tune_sub`/`val_select`/`test`/`val_clean`,
  laporan dibanding angka referensi, mosaik QA (10 klaster terbesar + 40 acak). `--benchmark N` mengukur waktu dulu.
- `src/densepoly/data/density.py` + `scripts/make_density.py`: strata tertil + indeks sekunder tile bersentuhan vs
  tidak; file strata yang sudah ada tidak ditimpa tanpa `--force`.
- `src/densepoly/labels/contact.py` dibuat lebih awal dari jadwal Fase 2, karena indeks sekunder kepadatan
  memakainya.
- `scripts/prepare_crowdai.py`: `--stats` → `splits/crowdai_stats.json`; `--cache` → peta instance uint16 dan
  `index.csv` per split.
- Uji end-to-end pada unduhan CrowdAI palsu (duplikat hasil rotasi/flip dan citra val yang bocor dari train).
- WHU protokol A: `src/densepoly/data/whu.py` + `scripts/prepare_whu_pix2poly.py`, port
  `data_preprocess/whu_buildings_to_coco.py` Pix2Poly (commit `4e4c003`, MIT; aslinya dari HiSup). Salinan persis ada
  di `third_party/pix2poly/`. Tes membuktikan poligon (WKB), grid patch, dan seluruh loop patch identik dengan kode
  asli. Ditambah `touching_pairs` untuk mengecek COCO resmi WHU (`docs/DATA.md` §3.4) dan uji end-to-end pada unduhan
  WHU palsu. `make_density.py` kini juga menerima `--data whu`.

**Keputusan (28 Okt 2026):** protokol WHU **A**; definisi kepadatan default (jumlah bangunan GT per tile, tertil),
dengan angka ambang menunggu data. `val_tune_sub` WHU = tile utuh acak (seed 0) sampai ≥ 1.000 patch.

**Temuan yang bertentangan dengan dokumen** (dari membaca kode Pix2Poly utuh; dokumen terkait sudah dikoreksi):
- Sampel negatif WHU adalah patch dengan ≤ 5 piksel bangunan. Patch dengan > 5 piksel yang semua poligonnya terbuang
  filter tidak punya anotasi sama sekali, jadi klaim `docs/EVALUATION.md` §2.2 bahwa setiap patch WHU punya minimal
  satu anotasi tidak tepat. Detail lain (lubang sebagai ring tambahan, kontur lewat pusat piksel, id mulai 0, urutan
  `os.listdir`) dicatat di `docs/DATA.md` §3.2.
- Untuk Fase 3: `frPyObjects([])` pycocotools melempar `IndexError` pada anotasi negatif WHU, dan COCOeval menghitung
  deteksi yang cocok dengan GT ber-id 0 sebagai positif palsu (`cocoeval.py` baris 295 dan 375).

**Catatan metode:** dengan r = 2, rumah yang dipisah celah 1 px tetap berlabel kontak (jarak pusat piksel 2 ≤ r),
sesuai definisi `docs/METHOD.md` §3.2; celah 2 px tidak. `tests/test_contact.py` memastikan perilaku ini.

**Temuan saat mengunduh CrowdAI (28 Okt 2026):**
- AIcrowd kini menandai file v1.0 sebagai `DEPRECATED` karena kebocoran train–test, masalah yang ditangani deduplikasi
  kita. Ada versi baru `mapping-challenge-v2.0.tar.gz` (3,17 GB): split train/val/test lengkap dengan GT, 258.044 tile
  (64.511 unik) dan 2.033.816 anotasi (508.454 unik), memuat versi teraugmentasi dan tidak.
- `LICENSE.md` v2.0: lisensi CC BY-NC-SA 4.0; dataset diadaptasi dari SpaceNet Buildings V2 dengan kode
  `spMohanty/mapping-challenge-preprocessing` cabang `2024-revision-with-bug-fixes`. Jadi v2.0 adalah regenerasi resmi
  oleh panitia, bukan subset hasil paper deduplikasi Adimoolam dkk. (README repo mereka: tautan subset "coming soon").
- v1.0 tetap dibutuhkan, karena prediksi rilisan HiSup dan angka paper pembanding dihitung di val v1.0. Pilihan tetap
  memakai split sendiri dari v1.0 atau beralih ke split v2.0 menyentuh keputusan metodologi #7, jadi dibahas setelah isi
  v2.0 diperiksa.
- Cara hash kita identik dengan `compute_hashes_augmented.py` repo `CrowdAI_Hash_and_search` (commit `afc58d7`):
  6 varian dengan `rotate(90/180/270)` (pada citra persegi sama dengan `transpose`) dan flip H/V, lalu `phash`.
  `train_hashes.json`/`val_hashes.json` repo hanya berisi hash asli (satu per citra), jadi angka 3.949 dari
  `val_in_train_np.json` sebanding dengan `original_hash_not_in_train_originals` di `dedup_report.json` kita.

**Isi v2.0** (diperiksa 28 Okt 2026; baru JSON-nya yang diekstrak ke `data/crowdai_v2/`):
- `CHANGELOG.md` (rilis 8 Mei 2025): kebocoran antar-split train/val/test diperbaiki; setiap citra diberi key
  `augmentation_source_id`; ada `annotation_non_augmented.json` terpisah; GT test dirilis; skrip preprocessing dirilis.
- Train 206.892, val 25.552, dan test 25.600 citra, masing-masing tepat 4 versi (asli + 3 augmentasi) dari 51.723,
  6.388, dan 6.400 tile unik (total 64.511, cocok dengan halaman AIcrowd). Anotasi non-augmentasi: 408.321, 50.194,
  dan 49.939. Kategori `100`. Nama file adalah id 64-bit.
- `*_image_id_to_file_path_map.json` memetakan id ke tile sumber SpaceNet (nama AOI + koordinat). Untuk tile
  non-augmentasi, tidak ada tile sumber yang muncul di dua split (0 untuk ketiga pasangan). Keempat AOI (Vegas,
  Shanghai, Khartoum, Paris) ada di setiap split dengan proporsi mirip, jadi split tidak dipisah per wilayah.
- Keterbatasan untuk proyek ini: prediksi HiSup dan angka paper pembanding semuanya dihitung di v1.0, jadi tidak ada
  pembanding di v2.0. Model yang dilatih di v1.0 (termasuk HiSup) kemungkinan sudah melihat area test v2.0, karena
  keduanya diturunkan dari citra SpaceNet yang sama (belum diverifikasi).
- **Keputusan (28 Okt 2026): opsi A**, tetap v1.0 + deduplikasi sendiri (keputusan #7 tidak berubah). Alasannya,
  posisi terhadap SOTA (`docs/EXPERIMENTS.md` §8) dan analisis N2 untuk HiSup hanya bisa di v1.0. Opsi B (pindah ke
  v2.0: test 6.400 tile resmi, tetapi tanpa pembanding) tidak dipilih. v2.0 disebut di laporan sebagai perbaikan
  resmi panitia. Opsional di akhir: satu run FULL di v2.0 bila waktu GPU tersisa.

**Data CrowdAI v1.0** (diekstrak 28 Okt 2026; ukuran, SHA-256, dan statistik anotasi di `docs/DATA.md` §1 dan §2.1):
- Jumlah citra sama dengan referensi (train 280.741, val 60.317), tidak ada file yang hilang, kategori `100`.
- Temuan: 98,8% ring GT mengulang titik pertama di akhir, jadi output kita ikut mengulang (`docs/DATA.md` §2.3).
  Ring berluas nol 0,64% (train) dan 0,68% (val): `area` = 0 dan tidak menutup satu piksel pun, dilewati
  rasterisasi; untuk Fase 3, COCOeval tetap menghitungnya sebagai GT. Nama file yang sama ada di train dan val dengan
  isi berbeda.
- p-hash citra asli kita identik dengan file hash repo deduplikasi pada 500 citra acak train dan 500 val.
- Waktu hash: 13,7 ms/citra di satu core, jadi 1,3 jam-inti untuk 341.058 citra (estimasi CLAUDE.md ±0,6 jam-inti).
  Paralel 13 worker di laptop ini (Intel Core Ultra 7 165U, 14 thread, RAM 31,5 GB): 305 citra/s (±4,2× satu core),
  jadi run penuh ±19 menit.

**Hasil deduplikasi** (28 Okt 2026; detail dan bukti di `docs/DATA.md` §2.2):
- Kedua angka referensi tepat: 114.548 hash asli unik di train dan 3.949 val tanpa duplikat persis (himpunan nama
  identik). Hash asli kita identik dengan file repo untuk semua citra.
- 29.283 klaster train (sesuai "±30 rb" di README repo). Split awal: `train_dedup` 26.355, `val_tune` 2.928
  (`val_tune_sub` 1.000, `val_select` 1.928); setelah opsi E, lihat di bawah.
- **Test bersih hanya 212 tile** (1.769 bangunan). Dari 3.949 citra val itu, 2.438 adalah salinan rotasi/flip citra
  train, dan 1.511 sisanya hanyalah 212 tile yang masing-masing muncul 2–15 kali. Semua pasangan dicek di tingkat
  piksel: tidak ada yang salah buang, dan tidak ada duplikat dekat yang lolos pencocokan persis.
- Tabrakan hash hanya terjadi di tiga klaster train terbesar (tile hitam nodata). Akibatnya ±100 tile train hilang,
  tetapi tidak ada kebocoran.

**Masalah:** dengan 212 tile, strata kepadatan hanya 82/69/61 tile (ambang tertil 5 dan 11 bangunan), sehingga uji
N1–N3 dan kriteria §8 (≤ 3 AP dari HiSup) lemah secara statistik. Opsi yang dibahas: A (test tetap 212 tile), E (test
holdout dari train), B/C (pindah ke v2.0: test 6.400 tile, tetapi banding HiSup hilang atau perlu investigasi
lintas-versi).

**Keputusan (28 Okt 2026): opsi E** (keputusan #7 dan kriteria §8 poin 5 diubah, sebelum Fase 3):
- Klaster train diacak sekali (seed 0): 10% pertama `val_tune` (identik dengan sebelumnya), 10% berikutnya `test`
  (2.928 tile, untuk semua klaim N1–N3 dan strata), sisanya `train_dedup` (23.427). Seluruh klaster masuk satu split,
  jadi model tidak pernah melihat tile test dalam bentuk putar/cermin apa pun.
- 212 tile val bersih menjadi split `val_clean`, khusus banding HiSup. Alasannya, HiSup dilatih dengan seluruh train
  v1.0, sehingga hanya subset inilah yang juga bersih baginya. Kriteria "mendekati SOTA" (≤ 3 AP) kini diukur di
  `val_clean`, dengan CI bootstrap.
- Kode: `split_train` (+ tes bahwa `val_tune` tidak berubah), nama split `val_clean` (khusus CrowdAI, seperti
  `val_full`), `crowdai_val_clean_clusters.json`. Dokumen yang disesuaikan: CLAUDE.md #7 dan #11, `docs/DATA.md`
  §2.2 dan §6, `docs/EXPERIMENTS.md` RQ3, §6, dan §8.5, PLAN Fase 1, 3, dan 9, serta `docs/COLAB.md` §1.

**Cache instance** (28 Okt 2026, `splits/crowdai_raster_stats.json`): `train_dedup` 23.427 citra (198.284 instance),
`val_tune` 2.928 (24.802), `test` 2.928 (24.964), `val_clean` 212 (1.769). Ring berluas nol yang dilewati: 1.344,
170, 181, dan 17. Tidak ada anotasi yang tertutup penuh oleh anotasi lain.

**Strata kepadatan CrowdAI** (dihitung, disetujui, dan dibekukan 28 Okt 2026; `splits/density_crowdai_*.json`,
tabel di `docs/DATA.md` §4):
- Indeks utama: tertil jumlah bangunan per tile di test = **6 dan 11**. Test: jarang 1.158 tile (3.870 bangunan),
  sedang 909 (8.219), padat 857 (12.875), kosong 4. `val_tune`: 1.191 / 873 / 864 / 0. `val_clean`: 91 / 60 / 61 / 0.
- Indeks sekunder (rasio bersentuhan, r = 2) ternyata **degenerasi**: 95% tile test tidak punya bangunan bersentuhan,
  jadi kedua tertilnya 0. **Keputusan:** diganti menjadi dua kelompok, `bersentuhan` (minimal satu instance punya
  piksel kontak) vs `tidak_bersentuhan`: test 146 vs 2.778 tile, `val_tune` 127 vs 2.801, `val_clean` 10 vs 202.
- Temuan: bangunan bersentuhan sangat jarang di CrowdAI. Bangunan yang punya tetangga dalam ≤ 1, 2, 3, 5, dan 10 px
  di test: 0,30%, 1,33%, 3,08%, 8,32%, dan 34,7%. Piksel kontak (r = 2) hanya 0,01% piksel bangunan. Per strata
  (≤ 2 px / ≤ 5 px): jarang 1,0% / 4,4%, sedang 1,2% / 6,4%, padat 1,5% / 10,7%. Akibatnya kelas kontak punya
  sangat sedikit contoh, dan klaim "rumah berdempetan" hanya bisa diuji pada subset kecil. Usulan: desain tidak
  diubah sekarang; dibahas lagi di Fase 2 (fraksi kontak per dataset, radius r) dan Fase 5 (bobot kelas kontak).

**Status kriteria "Selesai jika" Fase 1:** jumlah citra CrowdAI dilaporkan dan cocok dengan referensi, dan selisihnya
(212 tile `val_clean`) dijelaskan; tes membuktikan tidak ada citra atau klaster di dua split; ambang kepadatan
CrowdAI disetujui dan dibekukan.

**Unduhan WHU (29 Okt 2026)** (detail di `docs/DATA.md` §3.1, §3.2, dan §3.4):
- `whu_raster_1024.rar` (16,0 GB, RAR5) membungkus tiga ZIP tanpa kompresi: `r_train.zip`, `r_val.zip`, dan
  `r_test.zip`. CRC32 ketiganya cocok dengan header RAR, jadi unduhan utuh.
- `tar.exe` bawaan Windows (libarchive) salah membaca arsip ini sebagai ZIP terakhir dan hanya menampilkan isi val.
  ZIP-nya dibaca langsung dari rentang byte di dalam RAR.
- Isi: tile 1024² dengan nama berupa angka, train 2.943, val 627, test 2.220 (dua bagian area: `300` 1.540 dan
  `310` 680). Label dibuat ulang 20 Jul 2024. TIFF memakai kompresi LZW, sehingga `skimage.io.imread` (dipakai port
  Pix2Poly) butuh paket `imagecodecs`; OpenCV bisa membacanya.
- Baru label (165 MB) dan COCO resmi yang diekstrak, ke `data/whu/raw_0.2/{split}/label/` dan
  `data/whu/official_coco_0.2/`. Citra (16,4 GB) belum.

**Bertentangan dengan dokumen** (keputusan #11, `docs/DATA.md` §3.1–3.2), jadi konversi ditahan:
- Paper Pix2Poly memakai tile 512² 0,2 m: 9.420 / 1.537 / 3.848 citra, atau 84.780 / 13.833 / 34.632 patch.
- Skrip Pix2Poly tidak bisa membaca nama file berupa angka.
- Potongan 512² yang berisi bangunan di rilisan resmi: train 9.418, val 1.529, test 5.431 (bagian `300` saja: 3.943).
  Train dan val hampir sama dengan Pix2Poly, tetapi test tidak bisa direkonstruksi persis. Angka C-IoU WHU di paper
  (HiSup 79,62; Pix2Poly 81,63) jadi hanya sebanding secara kira-kira.

**COCO resmi** memisahkan bangunan yang bersentuhan:
- Test: 394 pasang poligon berjarak < 1 px.
- Val: IoU piksel terhadap raster 0,975; CC 8-tetangga menggabungkan 1,1% bangunan; 5,7% bangunan punya tetangga
  dalam r = 3 px (≈ 0,6 m). Di CrowdAI, dengan r = 2 px, angkanya 1,3%.

**Keputusan (29 Okt 2026): opsi R** menggantikan protokol A (keputusan #11; `docs/DATA.md` §3.2):
- Tile resmi 1024² dipotong menjadi kuadran 512². Hanya kuadran yang berisi bangunan yang dipakai, dan test hanya
  memakai bagian `300`. Setiap kuadran diproses seperti satu tile Pix2Poly (patch 224, overlap 34).
- Ada dua GT pada patch yang sama. GT utama (`annotation.json`) berasal dari bangunan resmi yang dirunut per bangunan
  dengan langkah poligon Pix2Poly. GT `p2p` (`annotation_p2p.json`) adalah konversi Pix2Poly persis.
- `patches.csv` menyimpan tile dan posisi setiap patch. Patch citra tidak ditulis ke disk.
- Kode baru: `whu.quadrants`, `instance_patch_annotations`, `convert_tile` (paralel), `official_by_tile`,
  `paths.annotation_file(..., variant="p2p")`, `paths.patch_index`, dan dependensi `imagecodecs`.
- Fixture WHU palsu kini meniru rilisan resmi. Ada 197 tes; salah satunya membuktikan GT `p2p` identik dengan skrip
  asli Pix2Poly yang dijalankan pada setiap kuadran.
- Citra resmi diekstrak (16,4 GB). GeoTIFF memastikan GSD 0,2 m: sampel 117 tile, semuanya 1024 × 1024 × 3.
- Label `int8` hanya muncul di tile tanpa bangunan (nol semua): 339 / 137 / 535 tile, sama dengan jumlah citra tanpa
  anotasi di COCO resmi.

**Hasil konversi** (8 worker, 10,5 menit; `splits/whu_stats.json`):

| Folder | Tile dipakai | Kuadran | Patch (Pix2Poly) | Poligon GT utama / p2p | Negatif utama / p2p |
|---|---|---|---|---|---|
| train | 2.943 | 9.418 | 84.762 (84.780) | 356.219 / 354.231 | 10.366 / 10.308 |
| val | 627 | 1.529 | 13.761 (13.833) | 40.888 / 40.685 | 3.765 / 3.753 |
| test | 1.540 (bagian `300`) | 3.943 | 35.487 (34.632) | 140.689 / 139.962 | 6.153 / 6.133 |

- Selisih jumlah patch dengan Pix2Poly berasal dari 2, 8, dan 95 tile 512² (× 9 patch).
- GT utama berisi ±0,5% lebih banyak poligon, karena bangunan berdempetan dipisah. GT `p2p` punya lebih banyak poligon
  berlubang (train 233 vs 13), karena gumpalan gabungan bisa mengurung halaman di antara rumah.
- Semua anotasi resmi di tile yang dipakai dirasterisasi tanpa tumpang tindih. WHU `val_tune_sub` 1.017 patch,
  `val_select` 12.744.
- Gambar QA `outputs/figures/qa/whu_gt_main_vs_p2p.png`: rumah gandeng yang digabung GT `p2p` menjadi satu poligon
  dipisah per bangunan oleh GT utama.

**Strata kepadatan WHU** (dari GT utama, per patch 224²; **disetujui dan dibekukan 29 Okt 2026**, dihitung ulang di
`0acb993` agar provenance bersih, dan isinya identik):
- Tertil jumlah bangunan per patch di test = **3 dan 6**. Test: kosong 6.399 (= 6.153 patch negatif + 246 patch tanpa
  anotasi), jarang 11.189, sedang 10.030, padat 7.869. `val_tune`: 3.886 / 4.976 / 2.834 / 2.065.
- Indeks sekunder (r = 3): test 1.371 patch bersentuhan vs 27.717 tidak; `val_tune` 371 vs 9.504.

**Status kriteria "Selesai jika" Fase 1 (29 Okt 2026): terpenuhi.**
- Jumlah citra CrowdAI cocok dengan semua angka referensi, dan selisih test (212 tile `val_clean`) dijelaskan.
- Tes membuktikan tidak ada citra atau klaster di dua split. Split WHU terpisah per area resmi.
- Ambang kepadatan CrowdAI dan WHU disetujui dan dibekukan.

**Fase 1 disetujui dan ditutup 29 Okt 2026.** Berikutnya Fase 2 (label dan prior morfologi).

## Fase 2 — Label dan prior morfologi (selesai dan disetujui 30 Oktober 2026)

**Ringkasan** (rincian di bawah; `pytest -q` → 265 tes lolos):

- **Label:** kelas (latar / bangunan / kontak), θ (sin 4θ, cos 4θ) dari `minAreaRect`, dan bobot tepi, semuanya
  diturunkan dari peta instance **sesudah** augmentasi geometri. Label dihitung saat training, tidak di-cache.
- **Statistik label:** kontak hanya ±0,01% piksel bangunan (4,5% tile/patch punya kontak); θ-valid ≥ 99,99%.
- **Prior:** MBI (*white top-hat by reconstruction*, 4 arah × 8 skala) dan peta celah (*black top-hat* 4 arah, mode
  `contrast`), dibekukan 29 Okt 2026.
- **Cache prior:** lengkap untuk semua split dan sudah diperiksa ulang. CrowdAI 29.495 citra dalam 1,64 jam, WHU 4.267
  tile dalam 3,94 jam (12 worker).
- **Keputusan selama Fase 2:** mode celah `contrast` dengan L_g 9 (WHU 13); skala MBI WHU dibulatkan ke bilangan
  ganjil; radius kontak r dan bobot kelas kontak ditunda ke Fase 5.
- **Temuan untuk fase berikut:**
  - Kontak sangat langka, jadi r dan bobot kelas kontak (Fase 5) perlu diputuskan dengan hati-hati.
  - Poligon GT utama WHU menyusut ±0,5 px per sisi (konvensi kontur Pix2Poly). Ini perlu diperhitungkan di uji
    bolak-balik Fase 3.
  - MBI juga menyala di jalan dan kolam, dan peta celah juga menangkap bayangan. Ini bahan pembahasan N3.
  - Model P1/P2 harus menghitung prior untuk setiap citra baru saat inferensi. Usulan (belum diputuskan): biaya
    prior per citra diukur dengan 1 thread dan dimasukkan ke tabel efisiensi (Fase 8).
  - *Sleep* yang dipicu dari luar proses tidak bisa dicegah skrip. Pekerjaan CPU panjang berikutnya perlu laptop yang
    tetap menyala, dan pengukur waktunya perlu diperiksa.

**Dibuat** (diuji di data sintetis; `pytest -q` → 265 tes lolos):

- `labels/orientation.py`: θ dari sisi kotak `minAreaRect` yang lebih panjang. Bangunan selebar 1 px (misalnya
  terpotong tepi tile) tetap mendapat arahnya. Hasilnya (sin 4θ, cos 4θ) dan mask valid V (luas ≥ A_θ).
- `labels/boundary_weight.py`: piksel batas (kedua sisi setiap tepi tetangga-4), lalu `w = 1 + α·exp(−d²/2σ²)`.
- `labels/targets.py`: semua target diturunkan dari satu peta instance.
- `labels/instances.py`: cache peta instance untuk kedua dataset. WHU memakai GT utama (`--cache`).
- `data/transforms.py`: geometri dihedral pada RGB, prior, dan peta instance, lalu target diturunkan sesudahnya.
  Fotometrik ±10% hanya pada RGB. Padding: reflect; kelas 255; θ, V, dan w 0. Tes membuktikan target yang diturunkan
  ulang sama dengan target yang ditransformasi, mengikuti tabel tanda sin 4θ di METHOD §3.5.
- `priors/mbi.py` (*white top-hat by reconstruction*, DMP), `priors/gap.py` (mode `max` dan `contrast`),
  `priors/cache.py` (PNG R = MBI, G = celah; skala persentil 99,5; statistik normalisasi).
- Skrip: `precompute_priors.py` (`--benchmark`, `--stats`, cache yang bisa dilanjutkan), `label_stats.py`,
  `figures.py --qa-labels/--qa-gap-mode`, dan `gap_check.py`.
- `paths`: `val_tune_sub` dan `val_select` memakai folder cache `val_tune`. Prior WHU dikunci per tile resmi.

**Ketidaksesuaian spesifikasi:** SE garis harus berpanjang ganjil (METHOD §2.2, §10), sedangkan skala WHU
15, 30, …, 120 berisi bilangan genap. Usulan: 15, 31, 45, 61, 75, 91, 105, 121 (15·k dibulatkan ke ganjil
berikutnya). Sudah dipakai di `configs/data/whu.yaml`; disetujui 29 Okt 2026 (keputusan 2 di bawah).

**Statistik label** (`splits/<ds>_label_stats.json`):

| Dataset / split | Kontak / piksel bangunan | Tile/patch dengan kontak | θ-valid / piksel bangunan |
|---|---|---|---|
| CrowdAI `train_dedup` (r = 2) | 0,012% (55.405 px) | 4,5% | 99,995% |
| CrowdAI `test` | 0,014% | 5,0% | 99,995% |
| WHU train (r = 3) | 0,014% (119.434 px) | 4,5% | 99,99% |
| WHU test | 0,011% | 3,9% | 99,99% |

- Kontak sangat langka di kedua dataset: sekitar 1 piksel per 8.000 piksel bangunan.
- Di WHU, poligon GT utama (konvensi kontur Pix2Poly) menyusut ±0,5 px per sisi. Celah antar-rumah jadi bertambah
  ±1 px, sehingga sebagian pasangan berjarak 2–3 px keluar dari r = 3.
- Label dihitung saat training (tidak di-cache), jadi radius r dan bobot kelas kontak masih bisa diputuskan di Fase 5
  tanpa mengulang cache.

**Pemeriksaan peta celah di `val_tune`** (`outputs/metrics/<ds>/phase2/gap_check_val_tune.json`, commit `34854e4`):
- Metrik: AUC piksel celah sempit (latar yang melihat dua bangunan dalam r + 1 px) terhadap latar lain dan terhadap
  atap.
- CrowdAI (2.928 tile): `contrast` lebih baik di setiap L_g (L_g 9: 0,788 vs 0,782; L_g 11: 0,797 vs 0,792).
- WHU (1.000 patch): `max` sedikit lebih baik (L_g 13: 0,733 vs 0,722).
- Selisih antar-mode dan antar-L_g ≤ 0,02, jadi pilihannya hampir tidak berpengaruh.

**Gambar QA:** `outputs/figures/qa/{crowdai,whu}_{labels,gap_mode}.png`.
- Kontak hanya garis tipis di tile yang bersentuhan.
- θ di CrowdAI didominasi 0°/90° (pola grid), sedangkan di WHU beragam.
- MBI menyala di atap terang, tetapi juga di jalan dan kolam renang, dan lemah di atap gelap (kelemahan MBI yang
  dikenal).
- Peta celah menangkap celah antar-atap, dan juga bayangan serta tepi jalan.

**Benchmark prior** (12 worker):
- CrowdAI: 29.495 tile, ±1,6 jam.
- WHU: 4.267 tile resmi 1024², ±4,8 jam.
- Satu core tanpa beban: ±0,6 s per tile 300² dan ±13 s per tile 1024²; per satuan luas sesuai METHOD (±1,3 s per
  320²). Saat 12 proses berjalan bersamaan, waktu per unit naik ±3× karena laptop didominasi E-core.

**Keputusan (29 Okt 2026, disetujui):**
1. Mode celah `contrast` dengan L_g default 9 px (WHU 13), dibekukan.
2. Skala MBI dibekukan: CrowdAI 11–81 px; WHU 15, 31, 45, 61, 75, 91, 105, 121 px.
3. Run prior penuh dijalankan: statistik dari 500 unit train, lalu cache CrowdAI dan WHU berurutan.
4. Radius kontak r dan bobot kelas kontak dibahas di Fase 5.

**Run prior penuh** (29 Okt 2026, 15.44–23.26, 12 worker, kode commit `191fb96`):

Statistik beku (`configs/data/<ds>_prior_stats.json`; 500 unit train, 20.000 piksel per unit, seed 0):

| Dataset | q99,5 (MBI / celah) | mean (MBI / celah) | std (MBI / celah) |
|---|---|---|---|
| CrowdAI | 0,0521 / 0,2902 | 0,0771 / 0,1020 | 0,1624 / 0,1545 |
| WHU | 0,0604 / 0,6392 | 0,1039 / 0,1436 | 0,1980 / 0,1848 |

Cache (`splits/<ds>_prior_cache.json`):

| Dataset / split | Unit | Wall | Detik per unit di worker |
|---|---|---|---|
| CrowdAI `train_dedup` | 23.427 | 70,4 menit | 2,14 |
| CrowdAI `val_tune` | 2.928 | 13,2 menit | 3,15 |
| CrowdAI `test` | 2.928 | 13,7 menit | 3,27 |
| CrowdAI `val_clean` | 212 | 1,4 menit | 3,97 |
| WHU `train_dedup` | 2.604 | 146,0 menit | 40,1 |
| WHU `val_tune` | 490 | 24,9 menit | 36,1 |
| WHU `test` | 1.173 | 65,4 menit | 39,8 |

- Total wall: CrowdAI 1,64 jam, WHU 3,94 jam (benchmark: ±1,6 dan ±4,8 jam).
- "Detik per unit di worker" diukur saat 12 worker berjalan bersamaan di Intel Core Ultra 7 165U (12 core, 14 thread,
  sebagian besar E-core). Nilainya beberapa kali angka satu core tanpa beban (±0,6 s dan ±13 s), jadi bukan biaya per
  citra untuk tabel efisiensi.
- `seconds_per_unit` = 180,1 di `whu_prior_stats.json` **tidak sah** sebagai ukuran waktu. Laptop masuk *sleep* pukul
  17.42–19.12 saat statistik WHU dihitung, dan pengukur waktu tetap berjalan selama *sleep*. Nilai q, mean, dan std
  tidak terpengaruh, karena proses hanya berhenti lalu lanjut. Waktu prior WHU yang sah ada di tabel cache.
- Layar mati (*Modern Standby*) beberapa kali antara 19.26 dan 23.26 tidak menghentikan proses: file prior WHU
  tertulis di setiap jendela 10 menit (146–209 tile per jendela).

Pemeriksaan sesudah run (skrip sekali jalan, di luar repo):
- Lengkap: di setiap split jumlah file = jumlah unit, tanpa file hilang atau berlebih. Ukuran setiap PNG sama dengan
  citra sumbernya, dan kanal B = 0.
- Hitung ulang: 5 unit acak per split (35 unit) dihitung ulang dengan statistik beku, dan hasilnya identik byte demi
  byte dengan cache.
- Sampel 200 unit per split: piksel jenuh (255) di `train_dedup` 0,69% / 0,55% (CrowdAI, MBI / celah) dan
  0,56% / 0,49% (WHU), dekat 0,5% yang diharapkan dari q = persentil 99,5.

**Status kriteria "Selesai jika" Fase 2 (30 Okt 2026): terpenuhi.**
- Tes lolos (265).
- Gambar QA (`outputs/figures/qa/{crowdai,whu}_{labels,gap_mode}.png`) sudah diperiksa dan disetujui.
- Cache prior lengkap untuk semua split, beserta waktu prosesnya (`splits/<ds>_prior_cache.json`).
- Fraksi piksel kontak dan fraksi piksel ber-θ valid per dataset dilaporkan (tabel statistik label di atas).

**Fase 2 disetujui dan ditutup 30 Okt 2026.** Berikutnya Fase 3 (pipeline evaluasi).

## Fase 3 — Pipeline evaluasi (sedang berjalan)

**Keputusan (30 Okt 2026, disetujui bersama rencana Fase 3):**
1. Ring GT berluas nol **tidak diubah** (protokol resmi, sama dengan evaluasi HiSup). Uji §6.1 dijalankan dengan dan
   tanpa ring itu (`docs/EVALUATION.md` §6.1 diperjelas).
2. HiSup dijalankan di `crowdai/val_clean`, bukan `crowdai/test` (`docs/EVALUATION.md` §6.4 dikoreksi). Kalimat lama
   adalah sisa dari sebelum opsi E: prediksi HiSup hanya ada untuk val, dan `test` kita diambil dari train yang dipakai
   melatih HiSup.
3. Kriteria `docs/EXPERIMENTS.md` §8 dikonfirmasi tanpa perubahan, jadi sejak Fase 3 dimulai terkunci.

**Temuan saat menyiapkan rencana** (30 Okt 2026, pycocotools 2.0.11):
- Ring berluas nol mendapat IoU 0 bahkan dengan dirinya sendiri, jadi tidak bisa dicocokkan oleh metode apa pun.
- Poligon dengan tepat 4 koordinat dibaca `frPyObjects` sebagai bbox lalu crash (`TypeError`), jadi pemeriksa
  prediksi harus menolaknya.
- GT WHU (opsi R): id anotasi mulai dari 0. Anotasi negatif (`segmentation` kosong) ada di 10.366 / 3.765 / 6.153
  patch train / val / test (GT utama). Patch tanpa anotasi sama sekali di test: 246 (GT utama) dan 227 (`p2p`). GT
  WHU test tidak punya ring ≤ 4 koordinat atau berluas nol.

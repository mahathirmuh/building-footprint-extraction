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

## Fase 3 — Pipeline evaluasi (selesai dan disetujui 30 Oktober 2026)

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
- GT val CrowdAI: id anotasi juga mulai dari 0; satu kategori (100); satu ring per anotasi; tidak ada ring yang
  membuat pycocotools atau shapely 2 gagal. Ada 3.644 anotasi ber-`area` 0, dan himpunannya persis sama dengan
  anotasi bermask kosong (3.488 ring berluas nol, sisanya ring setipis ≤ 1 px²). Dikoreksi setelah langkah 2;
  semula ditulis bahwa 156 di antaranya bangunan nyata.

**Langkah 1 — vendoring skrip metrik Pix2Poly (30 Okt 2026)** (`pytest -q` → 279 tes lolos):
- `third_party/pix2poly/{polis,angle_eval,cIoU}.py`: salinan persis (commit `4e4c003`, sha256 di
  `third_party/README.md`). `third_party/pix2poly_metrics/`: versi pembungkus, diimpor sebagai `pix2poly_metrics`
  (`third_party/` jadi akar paket kedua di `pyproject.toml`). Setiap perubahan ditandai `[wrapper N]` di kode dan
  dijelaskan di header file.
- `tests/test_pix2poly_metrics.py` (14 tes): di kasus yang tidak terkena perubahan, versi pembungkus memberi angka
  **identik** dengan aslinya (C-IoU, PoLiS, MTA). Kasus yang terkena perubahan berperilaku sesuai catatan, dan uji
  sintetis §6.1 untuk fungsi vendor (PoLiS persegi geser = 0,4·d; MTA rotasi ≥ α − 0,5°) lolos.
- Paket baru di dependensi inti: `multiprocess` 0.70.19 (diimpor `angle_eval.py`) dan `pyarrow` 26.0.0 (parquet).

**Temuan saat vendoring** (dokumen terkait sudah dikoreksi):
- `cIoU.py`: `N_GT` **sudah** di-reset per citra; yang basi hanya `mask_gti`. Jadi reset hanya mengubah IoU citra
  berprediksi tanpa GT, sedangkan C-IoU-nya ≈ 0 di kedua versi (EVALUATION §2.2 semula menulis keduanya tidak
  di-reset). `cIoU.py` juga menangani anotasi negatif WHU sebagai mask kosong.
- `polis.py` menimpa `segmentation` GT yang kosong di objek COCO-nya, jadi setiap metrik vendor memakai objek sendiri.
- `angle_eval.py` crash di shapely 2.2.0 / GEOS 3.14.1 untuk citra ber-GT kosong yang punya prediksi (misalnya
  sampel negatif WHU). Pembungkus 4 mengembalikan hasil kosong untuk citra itu, sesuai maksud filter presisinya.
- MTA didominasi artefak di sudut poligon: persegi panjang yang diputar 1° memberi 9,65°, dan yang diputar 30° memberi
  60°. Poligon identik memberi 0,0027°. Ini perlu diingat saat menafsirkan N1.
- `loadRes` pycocotools hanya menerima prediksi poligon yang punya `bbox`, dan memakai luas bbox sebagai `area`
  deteksi.
- Asal kode metrik: `angle_eval.py` dari Frame Field Learning (BSD-3-Clause), `cIoU.py` dari PolyWorld (lisensi
  BlackShark.ai: hanya riset akademik, tanpa sublisensi), `polis.py` dari `spgriffin/polis` (repo tidak ditemukan,
  lisensi tidak diketahui).

**boundary-iou-api (commit `37d2558`).** Keputusan 30 Okt 2026 (opsi a, disetujui): bagian COCO-nya di-vendor persis ke
`third_party/boundary_iou/` (sha256 dikunci di `tests/test_third_party.py`), dan extra `[eval]` dihapus. Alasannya:
- Lisensinya BSD-2-Clause (`license.txt`), jadi boleh dipasang maupun disalin dengan pemberitahuan lisensinya.
- `pip install git+…`, yaitu cara pertama di README-nya dan cara yang dipakai extra `[eval]` kita, hanya memasang
  `boundary_iou/__init__.py`, karena `setup.py`-nya tidak mendaftarkan subpaket. Paket yang terpasang rusak, jadi
  sudah dicopot lagi. `setup.py` juga menarik `cityscapesscripts`, `fvcore`, dan `panopticapi` dari cabang `master`
  (tidak terkunci), padahal modul COCO yang kita pakai tidak mengimpor satu pun dari ketiganya.
- `COCOeval.accumulate()`-nya memakai `np.float`, yang dihapus sejak numpy 1.24. AP akan dihitung dengan
  `accumulate` kita sendiri dari `evalImgs` (memang diperlukan untuk bootstrap §4), jadi fungsi itu tidak dipakai.
- AP_boundary memakai IoU := min(IoU mask, IoU batas), dengan lebar batas 2% diagonal citra (8 px untuk 300², 6 px
  untuk 224²).

**Langkah 2 — GT per split dan pemeriksa prediksi (30 Okt 2026)** (`pytest -q` → 313 tes lolos):
- `src/densepoly/eval/coco_io.py`:
  - GT per split: semua citra split dipertahankan; id anotasi +1 bila file sumbernya mulai dari 0; sampel negatif
    WHU dibuang (citranya tetap); selebihnya apa adanya. Field `info` mencatat sumber, offset, dan jumlah kasus
    khusus.
  - `check_predictions`: ketat untuk output kita, longgar untuk prediksi rilisan (hanya ring dengan < 3 titik
    berbeda yang dibuang; aturan longgar diubah di langkah 7).
  - Helper: `to_result` (format prediksi kita, ring tertutup), `gt_as_predictions`, `empty_mask_ids`, dan
    `load_results`, yang aman untuk daftar prediksi kosong (`loadRes` pycocotools crash di kasus itu).
- `scripts/prepare_gt.py` → `outputs/gt/<ds>/<split>[_p2p].json` (CrowdAI 64 s, WHU 29 s).
- `tests/test_coco_io.py` (34 tes), antara lain:
  - bukti bug id 0 di COCOeval: AP GT-sebagai-prediksi ≤ 0,5 dengan id asli, dan 1,0 setelah id digeser;
  - AP 100 hanya tercapai setelah GT bermask kosong dikeluarkan;
  - setiap pelanggaran aturan prediksi adalah error di mode ketat dan hanya dihitung di mode longgar.

GT per split (dari `info` file GT; WHU: GT utama / `p2p`):

| Dataset / split | Citra | Tanpa anotasi | Anotasi | Sampel negatif dibuang | GT bermask kosong |
|---|---|---|---|---|---|
| CrowdAI `val_tune` | 2.928 | 0 | 24.972 | 0 | 170 |
| CrowdAI `val_tune_sub` | 1.000 | 0 | 8.499 | 0 | 55 |
| CrowdAI `val_select` | 1.928 | 0 | 16.473 | 0 | 115 |
| CrowdAI `test` | 2.928 | 0 | 25.145 | 0 | 181 |
| CrowdAI `val_clean` | 212 | 0 | 1.786 | 0 | 17 |
| CrowdAI `val_full` | 60.317 | 0 | 515.364 | 0 | 3.644 |
| WHU `val_tune` | 13.761 | 3.886 / 3.873 | 40.888 / 40.685 | 3.765 / 3.753 | 0 |
| WHU `val_tune_sub` | 1.017 | 233 / 232 | 3.170 / 3.160 | 227 / 226 | 0 |
| WHU `val_select` | 12.744 | 3.653 / 3.641 | 37.718 / 37.525 | 3.538 / 3.527 | 0 |
| WHU `test` | 35.487 | 6.399 / 6.360 | 140.689 / 139.962 | 6.153 / 6.133 | 0 |

- Semua file sumber (CrowdAI train dan val, WHU) memberi id anotasi mulai dari 0, jadi offset +1 berlaku di semua
  split. Semua ring GT tertutup, jadi prediksi kita juga ditulis tertutup.
- Di CrowdAI, himpunan anotasi ber-`area` 0 sama persis dengan GT bermask kosong di setiap split (test: 174 ring
  berluas nol + 7 ring setipis ≤ 2 px²).

**Langkah 3 — metrik piksel dan instance (30 Okt 2026)** (`pytest -q` → 348 tes lolos):
- boundary-iou-api di-vendor (opsi a, lihat di atas); `tests/test_third_party.py` mengunci sha256 semua salinan persis.
- `eval/pixel.py`: mask gabungan per citra → IoU (ΣI/ΣU dan rata-rata per citra), F1, Boundary IoU (pita
  `mask_to_boundary`), dan `CIoU` utama di semua citra. Tes membuktikan nilai C-IoU per citra **identik** dengan
  `cIoU.py` vendor pada citra yang punya prediksi.
- `eval/instance.py`: pencocokan satu-satu ala COCO (IoU mask ≥ 0,5, tanpa batas 100 deteksi), lalu per pasangan
  PoLiS, MTA pasangan itu saja, galat θ, dan kecocokan verteks; per GT recall, merge, dan split. Bisa paralel
  (spawn pool); tes membuktikan hasilnya sama dengan versi serial.
- `eval/vertex.py` (Hungarian, ≤ 2 px) dan `eval/orientation.py` (θ_GT dengan aturan label; `points_theta` kini ada
  di `labels/orientation.py`).

**Pratinjau GT-sebagai-prediksi di data asli** (skrip sementara di luar repo; angka resmi §6.1 menyusul di langkah 6):
- CrowdAI `val_clean` dan `test`, serta WHU test: IoU, F1, dan Boundary IoU = 1; PoLiS_paired = 0; Vertex F1 = 1;
  MTA_inst ≈ 0,003°. Tidak ada crash.
- GT tak ter-match = GT bermask kosong, persis (17, 181, dan 0), sesuai §6.1 (ii).
- C-IoU CrowdAI 0,997–0,998, karena ring GT dengan < 3 titik berbeda dibuang saat dipakai sebagai prediksi (159 di
  test). Ini alasan §6.1 (i) mengeluarkan GT bermask kosong dari GT **dan** prediksi.
- CrowdAI test: 20 GT terhitung merge dan 20 split, diduga karena poligon GT saling tumpang tindih. Jumlah pastinya
  dihitung di langkah 6.
- GT CrowdAI sendiri punya poligon tidak valid (memotong dirinya sendiri; 442 di test) dan ring terbuka (306). Ini
  sifat GT resmi, tercatat oleh pemeriksa longgar.
- Waktu dengan 12 worker untuk instance: CrowdAI test 8 s (piksel) + 25 s (instance); WHU test 28 s + 107 s.

**Langkah 4 — `run_eval` dan `scripts/evaluate.py` (30 Okt 2026)** (`pytest -q` → 357 tes lolos):
- `eval/run_eval.py` menjalankan, berurutan: pemeriksaan prediksi; AP (pycocotools) dan AP_boundary (boundary-iou-api
  vendor, batas ditambahkan dengan fungsi single-core-nya); metrik piksel, instance, dan vendor. Hasilnya JSON §5,
  tabel parquet per citra dan per pasangan, serta cache `evalImgs`.
- `eval/stats.py` menjalankan `accumulate`/`summarize` asli pycocotools pada `evalImgs` yang diindeks ulang untuk daftar
  citra apa pun. Tes membuktikan:
  - hasilnya identik dengan COCOeval, baik di semua citra maupun di subset;
  - citra ganda setara dengan salinannya, sesuai kebutuhan bootstrap;
  - AP_boundary identik dengan `accumulate` vendor (`np.float` ditambal hanya di dalam tes).
- Strata memakai indeks utama dan sekunder dari file kepadatan beku. Metrik vendor per strata adalah nilai per citra
  yang di-pool ulang dengan aturan skripnya, dan di semua citra hasilnya diperiksa otomatis terhadap angka skrip vendor.
- Uji asap di data asli (GT sebagai prediksi, mode longgar, folder output sementara):
  - CrowdAI `val_clean` selesai dalam 22 s dan `test` dalam 153 s;
  - AP = AP_boundary = 0,989, karena GT bermask kosong tidak bisa cocok; PoLiS `null`; MTA 0,003°;
  - ekstrapolasi untuk val penuh: ±50–60 menit, dengan AP_boundary single-core dan metrik vendor sebagai bagian
    terlama.

**Langkah 5 — bootstrap (30 Okt 2026)** (`pytest -q` → 388 tes lolos):
- `eval/stats.py`:
  - `load_run` membaca tabel dan cache sebuah run;
  - `metric_values` menghitung metrik untuk setiap baris bobot resampel;
  - `compare` memberi Δ per seed, rata-ratanya, dan CI 95%;
  - `compare_strata` memberi Δpadat − Δjarang dengan bootstrap berstrata;
  - `seed_summary` memberi rata-rata ± SD.
- Tes (`tests/test_bootstrap.py`, 32 tes):
  - bobot identitas mereproduksi nilai `run_eval`, baik total maupun per strata (AP identik persis);
  - metode dibanding dirinya sendiri memberi Δ = 0 dengan CI [0, 0];
  - GT mengalahkan prediksi adegan uji pada AP dengan CI di atas 0;
  - metrik berpasangan membuang GT yang tidak ter-match oleh salah satu metode;
  - selisih berstrata tepat pada kasus yang dirancang;
  - replikasi dengan penyebut kosong dihitung, tetapi tidak dipakai untuk CI.
- Waktu di CrowdAI test (diukur pada output uji asap langkah 4): 1.000 replikasi metrik rasio < 0,1 s; AP ±55 ms per
  replikasi, sehingga satu perbandingan AP 3 seed × 2 metode makan ±5,5 menit (±2,5 menit di strata padat).

**Langkah 6 — uji kewarasan di data asli (30 Okt 2026)**: `scripts/sanity_checks.py` pada commit `1ddd931`, hasilnya
**52 dari 52 uji lolos** dan 0 tes unit gagal. Laporan `docs/sanity_report.md` dihasilkan skrip dari
`outputs/metrics/sanity_report.json`.
- §6.1, di CrowdAI `val_clean` dan `test` serta WHU test (GT utama dan `p2p`):
  - (i) tanpa GT bermask kosong: AP = AP_boundary = 1, C-IoU = CIoU_p2p = 1, PoLiS vendor `None` (0 citra valid),
    PoLiS_paired = 0, MTA_inst dan MTA ±0,0028°;
  - (ii) GT apa adanya: GT tak ter-match persis sama dengan GT bermask kosong (17, 181, 0, 0), dan IoU pasangan
    terkecil = 1;
  - merge/split di CrowdAI test: 20 dan 20 GT, persis sama dengan yang diramalkan dari tumpang tindih antar-GT.
- §6.2: enam tes unit kasus tepi lolos, dan evaluasi yang sama dua kali memberi angka identik.
- §6.6, prediksi buatan di `val_clean` (AP 0,518):
  - resampel identitas = `stats[0]` persis, dan CI 95% [0,500; 0,542] memuat nilai titik;
  - cara naif (`imgIds` duplikat langsung ke COCOeval) hanya menyisakan 63,3% citra unik dan memberi CI 29% lebih
    sempit (rasio lebar 0,71), sesuai teori (±1 − 1/e citra unik).
- Run pertama berhenti di WHU test: `cIoU.py` vendor hanya membaca satu ring per prediksi, sedangkan sebagian GT WHU
  berlubang. GT-sebagai-prediksi kini memakai ring pertama saja (mask terbukti identik), dan prediksi dengan banyak ring
  ditolak `run_eval` dengan pesan jelas. Ditambahkan juga `IoU_p2p`, IoU yang dilaporkan HiSup.
- Waktu dengan 12 worker: CrowdAI `val_clean` 1 menit, `test` 4 menit, WHU test ±9,5 menit per GT; total 26 menit.

**Langkah 7 — reproduksi HiSup (§6.4) dan HiSup di `val_clean` (30 Okt 2026)** (`pytest -q` → 408 tes lolos):
- Aturan longgar diubah sebelum evaluasi (commit `2726f61`, `docs/EVALUATION.md` §1):
  - 0,57% rilisan HiSup adalah ring dua titik seperti `[a, b, a]`. Bagi pycocotools ring ini mask kosong, jadi di
    evaluasi HiSup sendiri ring itu ikut AP sebagai positif palsu.
  - Aturan lama membuangnya. Itu berarti protokol yang berbeda, dan AP bisa naik.
  - Sekarang hanya ring ≤ 4 angka yang dibuang. Ring dua titik tetap ikut AP, AP_boundary, IoU, dan C-IoU, dan
    hanya dikeluarkan dari `polis.py` dan `angle_eval.py`.
- `sanity.hisup_reproduction` (commit `3c65809`) membaca hasil evaluasi `val_full` dan membandingkannya dengan
  `configs/paper_numbers.yaml` (arXiv:2208.00609v1, Tabel 1–2, hlm. 9). Ujinya: |ΔAP| ≤ 0,3.
- `crowdai/val_full`: 60.317 citra dan 506.978 prediksi. Tidak ada prediksi yang dibuang; ada 2.890 ring dua titik,
  5.288 poligon tidak valid, dan 305 ring tertutup.

| Metrik | Paper (dilaporkan di paper) | Pipeline ini | Selisih |
|---|---|---|---|
| AP | 79,4 | 79,43 | +0,03 |
| AP50 | 92,7 | 92,68 | −0,02 |
| AP75 | 85,3 | 85,31 | +0,01 |
| AR (AR100) | 81,5 | 81,55 | +0,05 |
| AP_boundary | 66,5 | 66,51 | +0,01 |
| PoLiS | 0,726 | 0,726 | +0,000 |
| C-IoU (`CIoU_p2p`) | 89,6 | 89,67 | +0,07 |
| IoU (`IoU_p2p`) | 94,3 | 94,28 | −0,02 |

- **Uji §6.4 lolos** (ΔAP = +0,03).
  - Setelah dibulatkan ke presisi paper, 7 dari 8 angka sama persis. Hanya C-IoU yang berbeda: 89,67 dibulatkan
    menjadi 89,7, sedangkan paper 89,6 (selisih minimal 0,02). Penyebabnya belum ditelusuri.
  - AR50 dan AR75 paper (93,1 dan 86,7) tidak dibandingkan, karena ringkasan standar pycocotools yang kita pakai tidak
    menghitungnya.
  - Hasil ini sekaligus memvalidasi bagian pipeline yang kita tulis sendiri: AP_boundary dihitung dengan `accumulate`
    kita (versi vendor rusak di numpy ≥ 1.24), PoLiS dengan `polis.py` pembungkus tanpa ring dua titik, serta IoU dan
    C-IoU dengan `cIoU.py` pembungkus.
- Varian lain di `val_full`: CIoU semua citra 89,45; IoU ΣI/ΣU 95,20; MTA 31,94°; PoLiS_paired 0,720; MTA_inst
  31,93°; merge 1,5%; split 1,1%.
- Waktu `val_full` dengan 12 worker: 70,6 menit, lebih lama dari proyeksi benchmark 4.000 citra (±56 menit).
  - Per tahap: pemeriksaan 60 s, AP 129 s, AP_boundary 468 s, piksel 79 s, instance 341 s, vendor 3.066 s, ringkasan
    56 s.
  - Puncak memori proses utama 5,72 GB (working set) / 6,54 GB (private), ditambah ±0,95 GB per worker MTA.
  - Selama tahap MTA, RAM laptop tersisa ±0,3 GB (aplikasi lain memakai ±25 GB), jadi sebagian memori dipindah ke
    pagefile. Ini kemungkinan penyebab run lebih lama; belum dibuktikan.
  - Cache `evalImgs` `val_full` tidak disimpan, karena besar dan tidak ada bootstrap yang direncanakan di sana.
- `crowdai/val_clean`: 212 tile, 1.786 GT, 1.774 prediksi (9 ring dua titik); selesai dalam 23 s. Semua tile punya
  prediksi, jadi `CIoU` = `CIoU_p2p`. PoLiS dan MTA di tabel adalah versi vendor.

| Subset | Tile | GT | AP | AP_boundary | C-IoU | PoLiS | MTA (°) | merge | split |
|---|---|---|---|---|---|---|---|---|---|
| semua | 212 | 1.786 | 65,60 | 43,97 | 81,25 | 1,364 | 34,54 | 1,5% | 1,6% |
| jarang | 91 | 303 | 65,59 | 41,53 | 77,37 | 1,592 | 34,33 | 0,7% | 2,3% |
| sedang | 60 | 554 | 69,99 | 45,89 | 84,67 | 1,245 | 35,15 | 1,3% | 2,2% |
| padat | 61 | 929 | 63,39 | 45,17 | 83,66 | 1,152 | 34,22 | 1,8% | 1,0% |
| bersentuhan | 10 | 144 | 64,22 | 48,78 | 84,81 | 1,126 | 36,24 | 4,2% | 2,8% |
| tidak bersentuhan | 202 | 1.642 | 65,79 | 43,73 | 81,07 | 1,376 | 34,39 | 1,2% | 1,5% |

- AP HiSup di `val_clean` 13,8 poin di bawah `val_full`.
  - Arahnya sesuai dugaan kebocoran, karena `val_full` berisi duplikat data latih HiSup.
  - Besarnya tidak bisa seluruhnya dikaitkan ke kebocoran, karena `val_clean` juga kumpulan tile yang berbeda (wakil
    klaster tanpa duplikat).
  - Dengan 212 tile (61 padat, 10 bersentuhan), angka per strata punya ketidakpastian besar. Selang kepercayaan
    bootstrap dihitung saat perbandingan dengan model kita.
- Uji kewarasan dijalankan ulang pada commit `3c65809` (aturan longgar baru, ditambah §6.4): **53 dari 53 uji lolos**,
  dan 0 tes unit gagal.
  - Angka §6.1, §6.2, dan §6.6 tidak berubah. Yang berubah hanya keterangan pemeriksaan prediksi: ring dua titik di GT
    (16 di `val_clean`, 159 di `test`) kini ikut AP sebagai mask kosong, tidak lagi dibuang.
  - Waktunya 48 menit, sedangkan run pertama 26 menit. Selama kasus CrowdAI `test` dan WHU, RAM hampir penuh dan
    laptop memakai baterai.
  - Tabel §6.4 kemudian dirender ulang dari JSON yang sama dengan opsi baru `--render-only`, supaya PoLiS ditulis
    0,7261, bukan 0,73. Aturannya: satu desimal lebih banyak daripada paper.

**Status kriteria "Selesai jika" Fase 3 (30 Okt 2026): terpenuhi.**
- Uji kewarasan §6.1, §6.2, §6.4, dan §6.6 lolos: 53 dari 53 (`docs/sanity_report.md`). Tes lolos (408).
- AP HiSup di val penuh 79,43, dalam ±0,3 dari angka paper (79,4).
- Metrik HiSup di `crowdai/val_clean`, total dan per strata, tersedia di
  `outputs/metrics/crowdai/hisup_release/ref/val_clean.json`.
- Perubahan aturan longgar di langkah 7 disampaikan bersama laporan penutupan, dan tidak ada keberatan.

**Fase 3 disetujui dan ditutup 30 Okt 2026.** Berikutnya Fase 4 (post-processing dan baseline klasik B0).

## Fase 4 — Post-processing dan baseline klasik B0

**Keputusan (30 Okt 2026, disetujui bersama rencana Fase 4):**
1. Parameter post-processing:
   - nilai default (`docs/METHOD.md` §10) masuk ke bagian `postproc` di `configs/base.yaml`;
   - hasil tuning disimpan di `configs/postproc/<dataset>/<exp>.yaml` sebagai lapisan config keempat;
   - grid tuning ada di `configs/postproc/grid.yaml`.

   Bagian `postproc` semula hanya berisi `instance` dan `polygon`.
2. B0 membaca MBI dari cache prior Fase 2, yaitu nilai yang sudah diskalakan ke [0, 1] (§2.4). Waktu proses B0 diukur
   dari RGB, termasuk menghitung MBI, pada sampel tile.
3. Klarifikasi spesifikasi (bukan perubahan definisi):
   - a. Di B0, komponen foreground tanpa marker h-maxima tetap jadi instance sendiri, seperti §6.2 langkah 5.
     Pelabelan komponen B0 memakai konektivitas 4.
   - b. Setelah klip dan pembulatan 2 desimal, poligon dicek ulang. Kalau tidak valid, poligon masuk rantai fallback
     §7.3.
   - c. Oracle §6.3 (b) mode `theta` memakai θ dari poligon GT.
   - d. Oracle §6.3 (c) memakai mode `raw`, supaya yang diuji hanya marker dan watershed.
4. Tuning memakai evaluasi ringan, yaitu AP dan PoLiS_paired (`docs/EXPERIMENTS.md` §4). Kombinasi terpilih lalu
   dievaluasi penuh.
5. Fase 4 hanya untuk CrowdAI. Default WHU (METHOD §10) dibuat di Fase 8.

**Langkah 1 — kontur marching squares (30 Okt 2026):** `postproc/contours.py` (`mask_to_exterior`).
- Bolak-balik mask → poligon → raster pycocotools memberi mask yang identik per piksel pada 300 bentuk acak
  terhubung-4. Sebagian bentuk memotong tepi tile.
- Kotak 10 × 10 hanya kehilangan 4 × 1/8 px² di sudut-sudutnya.

**Langkah 2 — regularisasi (30 Okt 2026):** `postproc/regularize.py`, logika di-port dari prototipe. Tes kriteria
"Selesai jika" lolos:
- Tanpa derau, di 8 sudut rotasi: kotak 70 × 40 jadi 4 verteks, L 6, T 8, dan dinding miring 5, semuanya dengan IoU
  ≥ 0,95 terhadap poligon asli.
- Dengan derau tepi (25% piksel gradien dibalik), kotak tetap 4 verteks, dan IoU rata-rata hasil ≥ kontur mentah
  untuk keempat bentuk.
- Fallback: hasil snap yang tidak valid diganti DP, lalu kontur. Input yang sengaja sulit (40 kotak acak, L/T
  berderau, piksel tunggal, garis 1 px, semuanya juga dengan θ yang salah) selalu memberi poligon valid dengan IoU
  ≥ 0,7 terhadap kontur.
- Kotak kecil 12 × 9 diuji terpisah dengan θ yang benar: hasilnya 4 verteks, dan IoU-nya tidak lebih rendah dari
  kontur rasternya. Patokan IoU ≥ 0,95 tidak dipakai, karena tepinya jatuh tepat di pusat piksel sehingga rasternya
  sendiri hanya mencapai IoU ±0,89 terhadap poligon.

**Temuan langkah 2 (uji sintetis, skrip diagnostik di luar repo):** θ minAreaRect pada raster kotak 12 × 9 meleset
10° dan 5° pada sudut 10° dan 85°, karena θ_mar keluar 0°.
- Akibatnya IoU mode `mar` hanya 0,85 dan 0,92, padahal dengan θ yang benar 0,97–0,98.
- Pada kotak ≥ 20 × 14, galatnya ≤ 0,7°.
- Ini batas θ minAreaRect untuk bangunan kecil. Batas yang sama berlaku untuk label θ (§3.3), dan ini bagian dari
  alasan N1.

**Langkah 3 — marker, watershed, dan instance (30 Okt 2026):** `postproc/markers.py` dan `postproc/watershed.py`
(METHOD §6.2–6.3). Tes sintetis (10):
- marker dipotong ke F sebelum diberi label, sehingga marker di celah tidak menyatukan dua rumah;
- marker < `m_min` dibuang, tetapi komponennya tetap jadi instance;
- pita kontak memisahkan dua rumah berdempetan tepat di dindingnya;
- komponen F tanpa marker tetap jadi instance;
- instance dipecah di sentuhan diagonal;
- skor = rata-rata per instance;
- model biner memakai komponen terhubung-4;
- probabilitas sempurna dari 5 adegan sintetis memberi kembali setiap rumah satu-satu dengan IoU ≥ 0,99.

Yang terakhir tidak identik per piksel. Di pita kontak yang datar pada dinding bersama yang miring, watershed membagi
piksel menurut jarak ke marker, sehingga 1–2 piksel tangga raster bisa masuk ke tetangga.

**Klarifikasi `A_min` (METHOD §6.2 langkah 6 dan §6.3):** `A_min` dibuang dari setiap instance, bukan hanya dari
bagian hasil pemecahan. Kalimat lama hanya menyebut bagian pecahan. Tafsiran ini mengikuti grid §10 dan
EXPERIMENTS §4, yang menyebut `A_min` sebagai satu-satunya parameter model biner.

**Langkah 4 — poligonisasi (30 Okt 2026):**
- Yang dibuat:
  - `postproc/polygonize.py`: mode `raw`/`mar`/`theta`, θ dari head (§6.4), lalu klip, pembulatan 2 desimal, dan cek
    ulang;
  - default parameter post-processing di `configs/base.yaml` (keputusan 1);
  - field `fallback` di format prediksi, yang dijumlahkan menjadi `n_fallback`.
- Tes (9):
  - poligon dari 3 adegan sintetis lolos pemeriksa ketat Fase 3 di ketiga mode, dengan IoU ≥ 0,85 terhadap rumahnya;
  - rumah miring menjadi kotak 4 verteks di mode `mar`;
  - kotak yang keluar tile diklip;
  - θ head adalah rata-rata sirkular berbobot, benar melewati batas 0°/90°, dan κ = 0 untuk arah yang saling
    meniadakan;
  - poligon yang rusak oleh pembulatan ditolak.
- Klarifikasi (b) diterapkan: poligon yang rusak setelah klip atau pembulatan langsung diganti kontur, tanpa mencoba
  DP dulu, karena DP bisa rusak oleh pembulatan yang sama. Penggantian ini dihitung sebagai fallback (METHOD §7
  langkah 4).

**Langkah 6a — B0 (30 Okt 2026):** `classical/b0.py` (METHOD §8). Tes sintetis (11):
- ambang Otsu dengan batas bawah `t_min`, termasuk tile tanpa bangunan dan tile konstan;
- opening, buang bintik, dan isi lubang;
- rumah memanjang di sudut 0°, 10°, 45°, dan 80° masing-masing satu instance;
- dua lingkaran yang bertumpuk terpisah di lehernya;
- tiga rumah terpisah menjadi tiga instance;
- dari RGB adegan sintetis (MBI, lalu B0), ≥ 70% rumah terisolasi ditemukan dengan IoU ≥ 0,5 di tiga nilai `t_min`.

**Temuan langkah 6a (marker h-maxima):** versi pertama memakai `skimage.morphology.h_maxima` dan memecah 3 rumah
menjadi 8 instance. Fungsi itu hanya menyisakan piksel paling atas tiap kubah jarak. Di punggung rumah memanjang, jarak
naik-turun sepersekian piksel (6,083 lawan 6,0), sehingga rumah 40 × 12 px yang miring 80° mendapat 5 marker, berapa
pun nilai h-nya. Marker sekarang dihitung dengan definisi buku teks, yaitu maksimum regional rekonstruksi dilasi
`dist − h` di bawah `dist` (*extended maxima*), dan dicatat di METHOD §8. Konektivitas 4 dari klarifikasi (a) tetap
berlaku.

**Langkah 6b — alat prediksi dan tuning (30 Okt 2026):**
- `infer.py` dan `scripts/predict.py`: prediksi B0 per split, paralel per potongan citra dengan urutan tetap.
  - Prediksi di `test` ditolak tanpa tuning beku, dan tidak menerima override `postproc`.
  - `--time-sample N` mengukur detik per citra dari RGB, termasuk menghitung MBI, sekaligus memeriksa bahwa MBI yang
    dihitung ulang sama dengan cache.
- `tuning.py` dan `scripts/tune_postproc.py`: grid dua tahap EXPERIMENTS §4 dari `configs/postproc/grid.yaml`.
  - Evaluasi ringan `run_eval.tuning_metrics` memberi AP dan PoLiS_paired yang identik dengan evaluasi penuh.
  - Hasilnya dibekukan ke `configs/postproc/<dataset>/<exp>.yaml`, berisi nilai terpilih dan skor setiap titik grid.
  - File beku tidak ditimpa tanpa `--force`.
- `load_config` memuat tuning beku sebagai lapisan keempat, hanya bagian `postproc`-nya.
- `postproc/oracle.py`: prediksi oracle §6.3 dari GT.
- Pengukuran awal B0 di 100 citra `val_tune_sub` (`--dry-run`, tidak ada yang ditulis): tuning penuh 1.000 citra
  diperkirakan ±29 menit dengan 12 worker, dan AP B0 di sampel itu ±0,1 poin.
- Diagnosis B0 di 30 citra (skrip di luar repo):
  - IoU piksel foreground hanya 0,25;
  - 54,6 instance per citra, padahal GT 7,3;
  - 18% rumah ditemukan dengan IoU ≥ 0,5.
- Pemeriksaan visual: MBI hanya menangkap sisi atap yang terkena matahari dan melewatkan atap datar yang gelap. Kepingan
  kecil muncul di pinggir atap. Jadi B0 memang lemah, bukan bug.

**Keputusan tambahan (30 Okt 2026, disetujui): oracle §6.3 (c) memakai kontur persis, tidak lagi mode `raw`
(keputusan 3d diubah).**
- Pada pratinjau, DP ε = 1 sendiri sudah menurunkan AP mask GT dari 98,09 (kontur persis) menjadi 93,41 (`raw`).
  Patokan "AP ≥ 95" EVALUATION §6.3 karena itu tidak bisa dipakai untuk `raw`, sebaik apa pun watershed-nya.
- Varian `raw` tetap dilaporkan, dan baris kontur persis ditambahkan di (b) (EVALUATION §6.3 dikoreksi).
- Angka resmi §6.3 berasal dari `scripts/sanity_checks.py --section 6.3`; angka pratinjau tidak dipakai.

**Langkah 6c — tuning B0 di `val_tune_sub` (30 Okt 2026), dibekukan ke `configs/postproc/crowdai/b0.yaml`:**
- Setelan terpilih: `t_min` 0,15, ρ 0,85, k 0,06, δ 20.
- AP hampir sama di seluruh grid. Tahap 1 (empat `t_min`): AP 0,088–0,090 poin. Tahap 2 (36 titik): AP 0,078–0,086
  poin. Pilihan ρ 0,85 jatuh pada aturan seri: AP-nya dalam 0,2 poin dari yang terbaik (0,0861 lawan 0,0862), dan
  PoLiS_paired-nya terkecil (2,84).
- Commit di file beku tercatat `028dd2b-dirty`, hanya karena `progress.md` belum di-commit; kodenya bersih.
- Waktu dengan 12 worker: tahap 1 208 s, tahap 2 6.484 s, total 111,5 menit. Ini jauh di atas proyeksi ±29 menit dari
  sampel 100 citra.
  - Selama run ada pekerjaan berat lain di laptop yang sama (±7 inti), dan RAM kosong hanya ±3,6 GB.
  - Puncak memori proses utama 3,28 GB (working set) / 4,07 GB (private). Hasil 36 titik grid disimpan bersamaan di
    memori.
  - Penyebab pastinya belum diukur.

**Temuan langkah 6d — MTA vendor memberi NaN (30 Okt 2026), diperbaiki dengan pembungkus 5 (disetujui):**
- Evaluasi pertama B0 di `test` berhenti di pemeriksaan pooling `run_eval`: MTA hasil pooling kita 41,04, sedangkan
  skrip vendor memberi NaN.
- Penyebabnya 3 dari 39.820 nilai MTA (3 citra). Ketiganya prediksi kecil (luas 10–14 px²) yang semua tepi
  tersisanya sejajar persis dengan kontur GT. Pembulatan memberi cosinus 1 + 2,2·10⁻¹⁶, sehingga `arccos`-nya NaN,
  bukan 0°. Satu NaN sudah cukup membuat `np.mean` vendor NaN untuk seluruh split, sedangkan jumlah per citra milik
  kita (pandas) melewatinya.
- Perbaikan (pembungkus 5, EVALUATION §2.3): cosinus dipotong ke 1 sebelum `arccos`. Nilai berhingga tidak berubah.
  Tes sintetis membandingkan dengan salinan asli: belah ketupat kecil di dalam belah ketupat GT 45° memberi NaN di
  kode asli dan 0 di versi pembungkus, sedangkan nilai lain identik.
- Dampak pada angka yang sudah ada:
  - HiSup `val_clean` dievaluasi ulang: 211 angkanya identik.
  - HiSup `val_full`: MTA vendor-nya berhingga, jadi tidak berubah. Ke-44 pasangan tanpa nilai `MTA_inst` dihitung
    ulang dengan pembungkus 5 (skrip di luar repo, 90 s). Semuanya tetap tanpa nilai, karena semua tepinya
    tersaring, bukan NaN. Jadi MTA_inst 31,93° tidak berubah, dan evaluasi ulang 70 menit tidak diperlukan.
  - B0 `val_tune_sub` tidak terdampak, karena MTA vendor-nya berhingga dan `n_pairs_without_mta` = 0.
  - §6.1 laporan kewarasan dihitung sebelum perbaikan. MTA_inst-nya hanya bisa turun atau tetap, karena pasangan yang
    dulu NaN kini bernilai 0. Ujinya berbentuk batas atas, jadi tetap lolos.
- Catatan pooling: sebelum perbaikan, MTA pooling kita (41,04) juga keliru. Jumlah per citra yang NaN dilewati
  pandas, tetapi banyaknya nilai citra itu tetap masuk penyebut. Setelah perbaikan, nilainya 41,12 dan sama dengan
  vendor.

**Langkah 6d — B0 di `test` (30 Okt 2026), dengan setelan beku `configs/postproc/crowdai/b0.yaml`:**
- Prediksi:
  - 2.928 tile, 180.937 poligon (61,8 per tile; GT 8,6 per tile), dan 39.793 fallback (22,0%);
  - waktu dinding 102,8 s dengan 12 worker, dari MBI cache.
- Waktu per citra dari RGB (50 citra, 1 proses): median 0,491 s (rata-rata 0,513 s). Dari jumlah itu, 0,390 s untuk
  MBI. MBI yang dihitung ulang sama dengan cache di 50 dari 50 citra.
- Evaluasi: 6,1 menit dengan 12 worker, pemeriksaan ketat, tidak ada prediksi yang dibuang. File metrik mencatat git
  `c4bfe3b-dirty`, hanya karena `progress.md` belum di-commit; kodenya bersih.
- PoLiS dan MTA di tabel adalah versi vendor. AP dan AP_boundary dalam poin persen.

| Subset | Tile | GT | Prediksi | AP | AP_boundary | IoU | C-IoU | PoLiS | MTA (°) | PoLiS_paired | MTA_inst (°) | recall@0,5 | split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semua | 2.928 | 25.145 | 180.937 | 0,076 | 0,047 | 22,49 | 6,93 | 3,525 | 41,12 | 2,775 | 34,66 | 13,2% | 49,2% |
| jarang | 1.158 | 3.904 | 49.605 | 0,045 | 0,027 | 16,06 | 4,47 | 3,482 | 39,48 | 2,826 | 33,06 | 12,2% | 52,4% |
| sedang | 909 | 8.278 | 61.976 | 0,089 | 0,044 | 23,73 | 7,45 | 3,535 | 41,52 | 2,773 | 33,57 | 12,8% | 50,7% |
| padat | 857 | 12.959 | 69.276 | 0,118 | 0,073 | 27,18 | 9,72 | 3,540 | 41,69 | 2,763 | 35,73 | 13,8% | 47,2% |
| bersentuhan | 146 | 1.847 | 10.450 | 0,579 | 0,547 | 24,20 | 8,34 | 3,066 | 41,61 | 2,517 | 36,89 | 12,6% | 44,2% |
| tidak bersentuhan | 2.778 | 23.294 | 170.407 | 0,071 | 0,041 | 22,39 | 6,86 | 3,558 | 41,08 | 2,795 | 34,49 | 13,3% | 49,6% |

- 4 tile kosong (tanpa GT) tidak punya baris AP.
- Merge hampir nol (0,016% total; 0,22% di bersentuhan), dan satu pasangan tidak punya nilai MTA_inst.
- Hasilnya konsisten dengan diagnosis langkah 6b: B0 memecah atap menjadi kepingan MBI kecil (split 49%), dan hanya
  13% rumah yang ditemukan.
- AP B0 di bawah 0,6 poin di semua subset, jadi selisih antar-strata tidak bermakna. Fungsi B0 adalah batas bawah
  klasik, bukan pembanding per strata.
- Pembanding dari `val_tune_sub` (1.000 citra, evaluasi penuh setelah tuning):
  - AP 0,086 dan AP_boundary 0,049;
  - IoU 22,03 dan C-IoU 6,72;
  - PoLiS 3,673, MTA 40,83°, PoLiS_paired 2,837, MTA_inst 34,88°;
  - 60.559 prediksi, 13.191 fallback (21,8%).

  Angka `test` dekat dengan ini, jadi tidak ada tanda tuning berlebih.

**Langkah 6e — oracle §6.3, angka resmi (30 Okt 2026):** `scripts/sanity_checks.py --section 6.3 --workers 12` pada
commit `550f81e`, 9,3 menit. Hasilnya di `docs/sanity_report.md` dan `outputs/metrics/crowdai/oracle_*/ref/`.
Laporan kewarasan kini **55 dari 55 uji lolos**. Angkanya sama dengan pratinjau.
- (a) Bolak-balik di `val_tune_sub`:
  - 8.444 anotasi bermask tidak kosong, IoU rata-rata 0,997561 (patokan ≥ 0,99);
  - 101 anotasi di bawah 1, dan semuanya terjelaskan (skrip di luar repo). 99 mask terdiri dari beberapa bagian
    terhubung-4, sehingga hanya ring terbesar yang disimpan, sama seperti instance. 4 mask berlubang, padahal
    poligon COCO tidak punya lubang. Dua anotasi masuk kedua kategori.
  - IoU terkecil 0,295 berasal dari mask 298 px yang terpecah menjadi 5 bagian.
- (b) dan (c), AP dalam poin persen:

| Run | AP | AP_boundary | C-IoU | PoLiS_paired | MTA_inst (°) | recall@0,5 | fallback |
|---|---|---|---|---|---|---|---|
| (b) mask GT → kontur persis | 98,09 | 98,07 | 48,32 | 0,284 | 27,28 | 99,27% | 0 |
| (b) mask GT → `raw` | 93,41 | 92,34 | 95,24 | 0,293 | 14,72 | 98,25% | 0 |
| (b) mask GT → `mar` | 92,41 | 81,52 | 86,66 | 0,544 | 21,13 | 99,24% | 181 |
| (b) mask GT → `theta` (θ poligon GT) | 92,80 | 82,31 | 86,94 | 0,532 | 20,32 | 99,24% | 166 |
| (c) 3 kelas GT → watershed → kontur persis | 96,53 | 96,51 | 48,28 | 0,283 | 27,33 | 98,56% | 0 |
| (c) 3 kelas GT → watershed → `raw` | 92,54 | 91,66 | 95,38 | 0,293 | 14,74 | 97,78% | 0 |

- Uji (c) lolos: AP 96,53 (patokan ≥ 95). Marker dan watershed dari probabilitas sempurna kehilangan 1,56 poin AP
  dibanding mask GT (98,09).
- Catatan untuk fase berikutnya:
  - Kontur persis punya C-IoU rendah (48,3), karena jumlah verteks tangga raster jauh di atas GT. MTA_inst-nya juga
    paling tinggi (27,3°).
  - Di oracle, `raw` memberi MTA_inst terendah (14,7°), lebih rendah dari `mar` (21,1°) dan `theta` (20,3°). Sebabnya
    `raw` hanyalah GT yang disederhanakan. Ini batas atas tahap poligon, bukan uji N1: N1 diuji pada prediksi model
    (B5 → B5′ → FULL-P0).
  - θ dari poligon GT (`theta`) hanya sedikit di atas `mar`: AP +0,39, PoLiS_paired −0,013, dan fallback 166 lawan
    181.

**Status kriteria "Selesai jika" Fase 4 (30 Okt 2026): terpenuhi.**
- Tes unit lolos: 472 tes, termasuk tes kriteria regularisasi (langkah 2) dan tes pembungkus 5.
- Uji oracle §6.3 lolos: (a) 0,997561 ≥ 0,99 dan (c) 96,53 ≥ 95. Laporan kewarasan 55 dari 55.
- Metrik B0 di `crowdai/test`, total dan per strata, tersedia di `outputs/metrics/crowdai/b0/ref/test.json`.
  Waktu proses per citra dari RGB: median 0,491 s (langkah 6d).

**Fase 4 disetujui dan ditutup 31 Okt 2026.** Berikutnya Fase 5 (infrastruktur model, B4 seed 0, dan B5).

## Fase 5 — Infrastruktur model, B4 (seed 0), dan B5 (sedang berjalan)

**Pengukuran sebelum rencana (31 Okt 2026, skrip di luar repo, tanpa menyentuh `test`):**
- Radius kontak r, pada 3.000 tile acak `train_dedup` (seed 0) dan oracle (c) di `val_tune_sub` (kontur persis,
  parameter default):

| r | Celah tercakup | Kontak / piksel bangunan | Tile berkontak | Bangunan berkontak | AP oracle | merge oracle |
|---|---|---|---|---|---|---|
| tanpa kontak (CC) | – | – | – | – | 96,44 | 0,26% |
| 2 | ≤ 1 px | 0,015% | 4,6% | 1,2% | 96,53 | 0,09% |
| 3 | ≤ 2 px | 0,033% | 11,1% | 3,0% | 96,59 | 0,09% |
| 4 | ≤ 3 px | 0,067% | 18,3% | 5,7% | 96,63 | 0,09% |
| 5 | ≤ 4 px | 0,130% | 26,0% | 8,8% | 96,64 | 0,09% |

  - r besar tidak merugikan tahap watershed, dan piksel kontak naik ±9 kali dari r = 2 ke r = 5.
  - Dengan probabilitas sempurna, kelas kontak hanya menambah 0,09 poin AP di atas CC, dan merge turun dari 0,26%
    ke 0,09%. Manfaatnya baru muncul bila model mengaburkan celah sempit, jadi efeknya di CrowdAI kemungkinan terbatas.
- Biaya loader per tile, satu proses di laptop saat CPU menganggur (300 tile `train_dedup`, median): baca RGB 3,1 ms,
  instance 1,8 ms, prior 6,2 ms; augmentasi dan label P2 32,3 ms. Di dalamnya bobot tepi 11,5 ms, kelas kontak 7,7 ms,
  orientasi 5,8 ms. Dengan 2 worker, batasnya ±2,9 iterasi per detik pada bs 16.
- Model U-Net++ EfficientNet-B4 5 kanal keluaran: 20,81 juta parameter (83 MB). Inferensi CPU laptop (12 thread): 0,60 s
  per tile 320².
- Data training untuk Colab (`train_dedup` + `val_tune`: citra, instance, prior): ±3,4 GB.
- Laptop tidak punya GPU. Repo GitHub ternyata publik, padahal dokumen menulis privat.

**Keputusan (31 Okt 2026, disetujui bersama rencana Fase 5):**
1. Radius r dan bobot kontak diputuskan sekali dengan pilot B4 r = 2 lawan r = 5 di `val_select`. Aturannya dibekukan
   sebelum ada model (`docs/EXPERIMENTS.md` §4.1): r = 5 dipakai bila `merge_rate` lebih rendah dan AP tidak turun
   lebih dari 0,5 poin.
2. Ambang "F1 kontak rendah" (pertanyaan terbuka sejak Fase 0): F1 piksel kelas kontak < 0,30 di `val_select`. Bila
   rendah, satu pilot lagi dengan bobot 5. METHOD semula menulis `val_tune`; `val_select` dipakai karena `val_tune_sub`
   khusus untuk tuning post-processing.
3. Label dipercepat dengan OpenCV (`edge_weight`, `contact_mask`). Definisinya tidak berubah, dan tes membuktikan
   hasilnya sama dengan implementasi lama.
4. Colab hanya untuk training. `best.pt` dibawa ke laptop; inferensi, tuning, dan evaluasi berjalan di laptop, jadi data
   `test` tidak perlu diunggah (`docs/COLAB.md`).
5. Colab T4 gratis, ruang Drive cukup untuk CrowdAI, dan repo tetap publik, jadi Colab tidak butuh token. Saran privat
   sampai tugas akhir dinilai sudah disampaikan; repo tetap publik atas pilihan pemilik proyek.

**Langkah 1 — model (31 Okt 2026):** `models/first_conv.py` dan `models/factory.py`. Tes (17): model 5 kanal sama dengan
model RGB untuk input prior apa pun (toleransi 1e-5), bobot RGB tidak berubah, bobot kanal prior mulai dari nol tetapi
mendapat gradien, dan setiap eksperimen mendapat arsitektur serta jumlah kanalnya.

**Langkah 2 — loss (31 Okt 2026):** `losses.py` untuk model biner, 3 kelas, dan FULL. Tes (19): nilai CE, BCE, Dice, dan
L_θ yang dihitung tangan; piksel padding tidak menyumbang nilai maupun gradien; bobot tepi dan bobot kontak hanya
mengubah suku yang dimaksud; loss turun di adegan sintetis; loss dihitung dalam float32 di dalam autocast.

**Langkah 3 — label lebih cepat (31 Okt 2026):** `contact_mask` memakai dilasi dan erosi OpenCV, `edge_weight` memakai
`distanceTransform` presisi. Tes membandingkan keduanya dengan implementasi scipy lama: mask kontak identik untuk
r = 1–8, termasuk di tepi tile dan label > 255, dan bobot tepi sama dengan toleransi 1e-5. Biaya tile P2 di laptop:

| Tahap | Sebelum | Sesudah |
|---|---|---|
| Kelas kontak | 7,7 ms | 0,9 ms |
| Bobot tepi | 11,5 ms | 2,3 ms |
| Augmentasi dan semua label | 32,3 ms | 13,3 ms |
| Total per tile (dengan baca file) | ±43 ms | ±24 ms |

Dengan 2 worker, batas loader naik dari ±2,9 menjadi ±5,3 iterasi per detik pada kecepatan CPU laptop; CPU Colab diukur
saat benchmark.

**Langkah 4 — dataset dan training (31 Okt 2026):** `data/dataset.py`, `train.py`, dan `scripts/train.py`.
- Augmentasi tiap sampel adalah fungsi murni dari (seed, epoch, indeks), sehingga batch tidak bergantung pada jumlah
  worker, dan run yang dilanjutkan melihat data yang sama. `EpochSampler` memberikan (epoch, indeks) ke dataset, karena
  worker persisten tidak melihat perubahan epoch pada salinan dataset mereka.
- DataLoader punya generator sendiri, karena membuat iterator mengambil angka dari RNG global yang juga dipakai
  *stochastic depth* EfficientNet.
- `last.pt` ditulis atomik tiap epoch dan tiap 20 menit. Log CSV baris-baris sesudah checkpoint dibuang saat resume.
  Melanjutkan run dengan config yang berbeda ditolak.
- Tambahan config: `train.accumulate` dan `train.log_every_steps`.
- `--benchmark N` mencatat iterasi per detik, porsi waktu menunggu data, memori GPU puncak, dan rata-rata CE, Dice, L_θ.

**Temuan langkah 4 (uji resume):** dua run tanpa henti dari seed yang sama di CPU **tidak identik per bit**, bahkan
dengan 1 thread dan tanpa oneDNN. Bobot akhirnya berbeda sampai 2·10⁻³, sebesar selisih dengan run yang di-resume.
Selisih kurva loss-nya sangat kecil (tabel di bawah), jadi kemungkinan besar ini pembulatan yang diperbesar Adam pada parameter
bergradien hampir nol; sumber persisnya belum dicari. Karena itu tes resume membandingkan kurva loss, bukan bobot.
Daya tangkapnya diuji dengan sengaja merusak resume (8 tile, potong di langkah 1, 2, 4, 6; skrip di luar repo):

| Resume | Selisih loss terbesar |
|---|---|
| utuh | ≤ 8,3·10⁻⁴ |
| RNG tidak dipulihkan | 1,7·10⁻²–5,9·10⁻² |
| optimizer tidak dipulihkan | 3,7·10⁻³–4,3·10⁻² |
| BatchNorm tidak dipulihkan | 2,6·10⁻²–3,5·10⁻² |
| sampler tidak melewati data yang sudah dipakai | 1,5·10⁻¹–6,8·10⁻¹ (di batas epoch memang tidak ada yang dilewati) |

Ambang tes 2·10⁻³. Dua kerusakan (RNG dan optimizer) ikut menjadi tes, supaya pemeriksaannya tetap tajam.

**Langkah 5 — inferensi dan tuning model (31 Okt 2026):** `infer.py`, `scripts/predict.py`, `tuning.py`,
`scripts/tune_postproc.py`, dan opsi `--run` di `scripts/evaluate.py`.
- Jaringan memberi kanal per tile: input di-pad, softmax (sigmoid di model biner), crop kembali ke tile, dengan
  tanh(ŝ, ĉ) di belakang probabilitas pada FULL.
- Kanal selalu melewati float16, baik disimpan maupun tidak, sehingga prediksi dari kanal tersimpan sama dengan prediksi
  langsung (ada tesnya). B5 memakai kanal B4 lewat `--from-channels`, jadi di CPU laptop jaringan tidak dijalankan dua
  kali.
- Tuning model berjalan dari kanal tersimpan: tahap 1 untuk semua model, tahap 2 hanya untuk poligon `mar`/`theta`.
- `choose` kini memilih AP terbaik pertama bila tidak ada baris yang punya pasangan. Kasus ini muncul pada model yang
  baru dilatih satu epoch di tes; pada model sungguhan tidak diharapkan.

**Langkah 7 — bahan Colab (31 Okt 2026):** `scripts/pack_colab.py` membuat satu tar (citra, cache instance, dan cache
prior `train_dedup` + `val_tune`) dengan manifest jumlah file dan sha256, dan `--verify` memeriksa salinan di Colab.
`notebooks/colab_train.ipynb` hanya memanggil skrip, tanpa token karena repo publik, dan `TASK` memilih `benchmark` atau
`train`. Tes (5): data test tidak ikut, dataset bisa membaca hasil ekstrak, salinan rusak ditolak, file hilang
menghentikan pembuatan tar, dan notebook tersimpan tanpa output.

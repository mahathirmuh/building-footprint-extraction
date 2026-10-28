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

## Fase 1 — Data (sedang berjalan: bagian CrowdAI selesai, menunggu data WHU)

**Dibuat** (diuji di data sintetis; `pytest -q` → 188 tes lolos):
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

**Menunggu:** data WHU 0,2 m di `data/whu/raw_0.2/{train,val,test}/{image,label}/`, lalu konversi protokol A, cek
jumlah patch, dan strata WHU (ambangnya minta persetujuan).

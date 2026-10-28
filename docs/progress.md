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

## Fase 0 — Setup repo (dikerjakan 28 Oktober 2026, menunggu persetujuan)

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

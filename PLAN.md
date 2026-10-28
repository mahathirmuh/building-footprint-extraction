# PLAN.md — Rencana Implementasi

Kerjakan **berurutan**. Setiap fase berisi tugas, output, dan kriteria "Selesai jika".
Setelah satu fase selesai: perbarui `docs/progress.md`, commit, lalu tunggu persetujuan sebelum lanjut.
Semua keputusan metodologi mengikuti `CLAUDE.md`; detail teknis ada di `docs/`.

Urutan sengaja dibuat: **data → label → evaluasi → post-processing → model**. Pipeline evaluasi dan
post-processing diuji dulu dengan GT ("oracle") sebelum ada model, supaya angka model nanti bisa dipercaya.
Kriteria keputusan di `docs/EXPERIMENTS.md` §8 sudah dibekukan sejak 28 Okt 2026 dan tidak diubah setelah ada hasil.

---

## Fase 0 — Setup repo

**Tugas**
- Buat struktur folder sesuai `CLAUDE.md`, `.gitignore` (`data/`, `outputs/`, `*.pt`, `*.npz`, `.env`,
  `.ipynb_checkpoints`), `pyproject.toml` dengan extras `[dev]` (pytest, ruff, nbstripout, nbconvert), `[eval]` (boundary-iou-api dari
  GitHub, multiprocess), `[yolo]` (ultralytics).
- `src/densepoly/config.py`: muat YAML dengan OmegaConf, gabungkan `base` ← `data/<ds>` ← `exp/<exp>` ← argumen CLI
  (`key=value`). Simpan config final ke folder run.
- `src/densepoly/paths.py`: semua path dari `DENSEPOLY_DATA` dan `DENSEPOLY_OUT`.
- Pasang `nbstripout` sebagai filter git, supaya output notebook tidak ikut ter-commit.
- `tests/fixtures/synthetic.py`: generator adegan sintetis ber-seed (ide dari `reference/demo/before_after.py`):
  rumah persegi yang diputar acak, deret rumah bersentuhan, rumah bentuk L, celah gelap sempit, latar bertekstur.
  Sumber kebenaran adalah **poligon GT** (koordinat COCO). Peta instance dibuat dengan merasterisasi poligon itu
  memakai pycocotools (bukan `cv2.fillPoly` seperti di demo), lalu RGB digambar dari peta instance.

**Output:** repo kosong yang bisa di-install; fixture sintetis.

**Selesai jika:** `pip install -e ".[dev]"` dan `pytest -q` lolos di CPU tanpa internet; peta instance fixture sama
persis dengan rasterisasi pycocotools dari poligon GT-nya.

---

## Fase 1 — Data: unduh, deduplikasi, split, protokol WHU, strata kepadatan

**Tugas**
- Saya mengunduh data secara manual (`docs/DATA.md` §1). Kamu menyiapkan skrip dan mengecek strukturnya.
- `src/densepoly/data/crowdai.py`: baca anotasi COCO, rasterisasi peta instance per citra (`docs/DATA.md` §2.3).
  Catat statistik tumpang tindih dan jumlah citra tanpa anotasi.
- `scripts/dedup_crowdai.py --data crowdai` (`docs/DATA.md` §2.2): p-hash 6 varian, klaster union-find,
  `train_dedup`, `val_tune` (beserta `val_tune_sub` dan `val_select`), test bersih, dan `splits/dedup_report.json`.
  Pekerjaan CPU ±0,6 jam-inti untuk hashing saja (tambah waktu baca disk); lihat `docs/COLAB.md` §8 untuk tempat
  menjalankannya. Minta konfirmasi sebelum menjalankan di seluruh data.
- Cek konvensi titik penutup poligon dan arah putaran di GT CrowdAI; catat di `docs/progress.md`.
- **WHU:** diskusikan protokol A/B dengan saya (`CLAUDE.md`, "Yang perlu saya putuskan"). Untuk A: siapkan
  `scripts/prepare_whu_pix2poly.py` yang mengikuti `data_preprocess/whu_buildings_to_coco.py` Pix2Poly **tanpa mengubah
  logika** (cantumkan atribusi MIT). Cek juga anotasi COCO resmi WHU 0,2 m: apakah bangunan yang bersentuhan
  terpisah (hitung pasangan poligon yang bersinggungan).
- `scripts/make_density.py --data crowdai --split test` (dan `val_tune`, `whu/test`): strata kepadatan
  (`docs/DATA.md` §4) → `splits/density_<dataset>_<split>.json`.

**Output:** `splits/*.txt`, `splits/dedup_report.json`, `splits/density_*.json`, statistik dataset.

**Selesai jika:** jumlah citra dilaporkan dan dibandingkan dengan angka terverifikasi di `CLAUDE.md` (280.741 /
60.317 / 114.548 / 3.949; README repo: ±30 rb unik setelah augmentasi), dan selisih dijelaskan; tes membuktikan
tidak ada citra atau klaster yang muncul di dua split; definisi kepadatan dan ambang tertilnya sudah saya setujui
lalu dibekukan.

---

## Fase 2 — Label dan prior morfologi (N3)

**Tugas**
- `labels/instances.py`, `labels/contact.py` (filter min/max, tanpa loop per pasangan), `labels/orientation.py`,
  `labels/boundary_weight.py` sesuai `docs/METHOD.md` §3.
- `data/transforms.py`: augmentasi geometri (flip, rot90, transpose) pada citra + prior + peta instance, lalu label
  diturunkan **sesudahnya**; augmentasi fotometrik ringan hanya pada RGB; padding 300 → 320.
- `priors/mbi.py`, `priors/gap.py` (`docs/METHOD.md` §2), `priors/cache.py`, `scripts/precompute_priors.py`
  (multiprocess, simpan `uint8`), statistik prior dari train → `configs/data/<ds>_prior_stats.json`.
- **Ukur dulu di 50 tile.** Saat dokumen disusun, MBI by reconstruction (32 opening) butuh ±1,4 s per tile 320² dan
  peta celah ±20 ms, jadi ±35 rb tile ≈ 14 jam-inti CPU. Laporkan angka di mesinmu dan minta konfirmasi sebelum
  menjalankan semuanya.
- Unit test (data sintetis): kontak hanya di pita selebar r sepanjang sisi yang bersentuhan; rumah terisolasi tanpa
  kontak; θ rumah yang diputar 30° kembali 30° ± 1°; sin 4θ berganti tanda saat flip; bobot tepi maksimum = 1 + α
  di tepi; MBI atap terang > latar; peta celah menyala di garis gelap antara dua atap terang.
- Gambar QA (`scripts/figures.py --qa-labels`): 12 tile per dataset dengan RGB, kelas, θ (warna HSV), bobot tepi,
  MBI, peta celah → `outputs/figures/qa/`.
- Pilih mode peta celah (`max` atau `contrast`) secara visual di 20 tile `val_tune`; catat keputusannya.

**Output:** modul label dan prior, tes, cache prior, statistik, gambar QA.

**Selesai jika:** tes lolos; gambar QA sudah saya periksa; cache prior lengkap untuk semua split beserta waktu
prosesnya; fraksi piksel kontak dan fraksi piksel ber-θ valid per dataset dilaporkan.

---

## Fase 3 — Pipeline evaluasi (sebelum ada model)

**Tugas**
- Vendor `polis.py`, `angle_eval.py`, `cIoU.py` dari `Pix2Poly/eval/hisup_eval_utils/metrics/` ke
  `third_party/pix2poly_metrics/` (MIT). Hanya perubahan **pembungkus** yang diizinkan (`docs/EVALUATION.md` §2.3):
  me-return nilai, output per citra, subset `imgIds`, hapus impor `descartes`, reset `mask_gti`/`N_GT` per citra.
  Pencocokan, ambang, sampling, dan cara pooling tidak boleh diubah. Catat semua perubahan di header file.
- `eval/coco_io.py`, `eval/pixel.py`, `eval/instance.py` (pencocokan satu-satu, merge/split/recall, metrik
  berpasangan `PoLiS_paired`, `MTA_inst`, galat θ), `eval/run_eval.py` (skema JSON `docs/EVALUATION.md` §5),
  `eval/stats.py` (bootstrap per citra **dan** bootstrap AP dari `evalImgs` yang di-cache, §4).
- Uji kewarasan `docs/EVALUATION.md` §6.1, §6.2, §6.4, dan §6.6. Uji §6.3 dikerjakan di Fase 4, §6.5 di Fase 8.

**Output:** paket evaluasi; `outputs/metrics/crowdai/hisup_release/ref/{val_full,test}.json`; `docs/sanity_report.md`.

**Selesai jika:** uji kewarasan §6.1, §6.2, §6.4, §6.6 lolos; AP HiSup hasil reproduksi di val penuh berada dalam
±0,3 poin dari paper (79,4), atau selisihnya dijelaskan; metrik HiSup di `crowdai/test` (total + per strata)
tersedia.

---

## Fase 4 — Post-processing dan baseline klasik B0

**Tugas**
- `postproc/contours.py` (marching squares, `docs/METHOD.md` §6.1), `postproc/markers.py`, `postproc/watershed.py`
  (marker dipotong ke foreground, komponen tanpa marker, konektivitas 4), `postproc/regularize.py` (kotak momen dan
  jalur DP + pelurusan, cek validitas dan fallback), `postproc/polygonize.py` (mode `raw`, `mar`, `theta`).
- Uji **oracle** (`docs/EVALUATION.md` §6.3): bolak-balik poligon; mask instance GT → `polygonize`; peta 3 kelas GT →
  marker → watershed → poligon. Hasilnya adalah batas atas yang bisa dicapai post-processing kita.
- `classical/b0.py`: MBI → Otsu (batas bawah `t_min`) → morfologi → watershed → `polygonize(mar)`. Setel di
  `val_tune_sub`, lalu jalankan di test.

**Output:** modul post-processing; `outputs/metrics/crowdai/oracle_*/ref/`; metrik B0.

**Selesai jika:** tes unit lolos, yaitu:
- bentuk tanpa derau di 8 sudut rotasi → kotak 4, L 6, T 8, satu dinding miring 5 verteks, IoU terhadap poligon asli
  ≥ 0,95;
- dengan derau tepi → kotak tetap 4 verteks, dan IoU rata-rata hasil ≥ IoU kontur mentah;
- poligon tidak valid → fallback valid (patokan: `reference/prototypes/regularize_proto.py`).

Selain itu: uji oracle §6.3 lolos; metrik B0 di `crowdai/test` (total + per strata) dan waktu proses per citra
tersedia.

---

## Fase 5 — Infrastruktur model, B4 (seed 0), dan B5

**Tugas**
- `models/first_conv.py` + tes kesetaraan (model 5 kanal == model 3 kanal untuk input sembarang).
- `models/factory.py`: config eksperimen → model (U-Net atau U-Net++, kanal input, kanal output).
- `losses.py` (`docs/METHOD.md` §5) + tes: loss turun di data sintetis; gradien bobot kanal prior ≠ 0; piksel padding
  tidak menyumbang loss.
- `data/dataset.py`, `train.py` (AdamW, cosine + warmup, AMP, checkpoint tiap epoch dan tiap 20 menit, resume,
  log CSV, batas waktu per sesi), `infer.py` (pad/crop, output COCO JSON dengan field `theta_deg` dan `kappa`).
- Smoke test CPU: 2 iterasi di fixture sintetis dengan `encoder_weights=None`.
- `notebooks/colab_train.ipynb` sesuai `docs/COLAB.md`. Ukur it/s dan memori di Colab (micro-batch ≥ 8), isi tabel
  biaya di `docs/EXPERIMENTS.md` §5, lalu **minta persetujuan** sebelum training panjang.
- Latih **B4 seed 0** saja. Putuskan bobot kelas kontak di sini (`docs/METHOD.md` §5), lalu berlakukan untuk semua model
  3 kelas. B5 = model B4 + `polygonize(mar)`. Setel post-processing di `val_tune_sub`, bekukan, evaluasi test.

**Output:** infrastruktur training/inferensi; checkpoint dan metrik B4, B5 (seed 0).

**Selesai jika:** smoke test lolos; training di Colab terbukti bisa dilanjutkan setelah runtime diputus; metrik B0, B4,
B5 (seed 0) di `crowdai/test` tersedia, total dan per strata.

---

## Fase 6 — Tier 1: N1 dan N3

**Tugas**
- Latih B4 seed 1–2, FULL-P0 ×3 seed, FULL-P2 ×3 seed (prioritas Tier 1, `docs/EXPERIMENTS.md` §5).
- Sebelum seed 1–2 FULL: cek skala `L_θ` terhadap loss lain di log seed 0; ubah λ hanya bila jelas mendominasi (catat).
- B5′ = model FULL-P0 + `polygonize(mar)`. Diagnostik θ: galat sudut θ instance (head dan minAreaRect) vs θ GT.
- Uji N1 (B5 vs B5′ vs FULL-P0, semua input RGB) dan N3 gabungan (FULL-P2 vs FULL-P0) dengan kriteria
  `docs/EXPERIMENTS.md` §8.
- Pilih varian FULL final dengan aturan §8 (AP di `val_select`), catat alasannya.

**Output:** tabel N1 (B5 vs B5′ vs FULL-P0) dan tabel N3 (P0 vs P2), dengan CI bootstrap; keputusan FULL final.

**Selesai jika:** semua run Tier 1 selesai dengan 3 seed; uji §8 untuk N1 dan N3 selesai; keputusan tercatat.

---

## Fase 7 — Tier 2: atribusi prior, rantai ablasi lengkap, WHU

**Tugas**
- FULL-P1 ×3 seed → atribusi: P2 vs P1 (sumbangan peta celah), P1 vs P0 (sumbangan MBI).
- B1, B2, B3 ×1 seed → rantai ablasi B1 → B4 lengkap (deskriptif).
- WHU (protokol yang disepakati): latih FULL final ×1 seed.

**Output:** tabel ablasi lengkap; tabel atribusi prior; checkpoint WHU.

**Selesai jika:** run Tier 2 selesai atau dibatalkan dengan alasan tertulis (misalnya kuota habis).

---

## Fase 8 — Perbandingan SOTA dan analisis kepadatan (N2)

**Tugas**
- `crowdai/test`: FULL final vs HiSup (prediksi rilisan) — total dan per strata, `CIoU` dan `CIoU_p2p`. Tabel terpisah
  untuk angka paper (PolyR-CNN, Pix2Poly, FFL, HiSup di val bocor), ditandai "dilaporkan di paper".
- WHU: evaluasi dengan skrip Pix2Poly yang di-vendor; bandingkan dengan angka paper. Opsional: uji kewarasan §6.5
  (checkpoint Pix2Poly rilisan pada test yang sama).
- Efisiensi: jumlah parameter, FPS, waktu post-processing di GPU/CPU yang sama.
- Analisis N2: efek per strata dan Δpadat − Δjarang (bootstrap berstrata).

**Output:** tabel dan grafik di `outputs/tables/` dan `outputs/figures/`.

**Selesai jika:** semua tabel dibuat ulang oleh skrip; setiap angka bisa ditelusuri ke file JSON.

---

## Fase 9 — (Opsional) FULL+Y: YOLOv8n-OBB sebagai seed

**Tugas**
- Label OBB dari peta instance (minAreaRect → 4 titik ternormalisasi). Latih `yolov8n-obb.pt` di train dedup.
- Gabungkan seed sesuai `docs/METHOD.md` §8 (seed disisipkan setelah filter `m_min`); setel `c_y` di `val_tune_sub`.
- Bandingkan FULL vs FULL+Y per strata; laporkan tambahan waktu inferensi.

**Selesai jika:** baris FULL+Y ada di tabel beserta biayanya, dan sudah diputuskan masuk laporan atau tidak.

---

## Fase 10 — Bahan laporan dan notebook demo

**Tugas**
- `scripts/tables.py` dan `scripts/figures.py`: tabel ablasi, N1, N3, SOTA, per strata, efisiensi; gambar
  sebelum–sesudah dari tile **asli** strata padat (pengganti gambar sintetis di slide 16 dan 19); contoh kegagalan.
- `notebooks/demo_hasil.ipynb`: notebook untuk dibaca, dipresentasikan, dan dikumpulkan ke dosen. Semua fungsi
  di-import dari `densepoly`; notebook hanya berisi pemanggilan, penjelasan, dan plot. Penjelasan dalam Bahasa
  Indonesia. Urutan isinya:
  1. Masalah dan gambaran pipeline.
  2. Data: contoh tile per strata kepadatan dan ringkasan deduplikasi (dari `splits/dedup_report.json`).
  3. Pengolahan citra klasik: brightness → MBI → peta celah → baseline B0, pada 3 tile.
  4. Label dari mask: kelas kontak, θ, dan bobot tepi.
  5. Inferensi FULL final pada 3–5 tile padat: P_b, P_k, θ → marker → watershed → poligon `raw` vs teregularisasi
     (overlay di atas citra).
  6. B4, B5, dan FULL pada tile yang sama.
  7. Tabel dan grafik hasil, **dibaca** dari `outputs/metrics/` (tidak dihitung ulang), termasuk per strata.
  8. Keterbatasan.
- Notebook harus berjalan dari atas ke bawah di runtime Colab baru dalam ≤ 10 menit (GPU atau CPU). Ia memakai
  checkpoint FULL final dan subset kecil tile yang dikemas di `data/demo/`.
- Di git, notebook disimpan **tanpa output**. Versi berisi output diekspor dengan
  `jupyter nbconvert --execute --to html` ke `outputs/report/demo_hasil.html`, dan juga ke `.ipynb` ter-eksekusi bila
  dosen memintanya.
- Opsional lebih awal: setelah Fase 4, bagian 1–4 (tanpa model) sudah bisa dibuat sebagai bahan laporan kemajuan.
- Tulis temuan dan keterbatasan di `docs/progress.md`: GT WHU dari CC, deduplikasi berbasis kesamaan p-hash,
  lubang poligon diabaikan, θ kurang bermakna untuk bangunan non-siku, perilaku metrik vendor (C-IoU hanya citra
  berprediksi, PoLiS cocok via bbox, MTA menggabungkan poligon per citra).

**Selesai jika:** satu perintah membuat ulang semua tabel dan gambar; `notebooks/demo_hasil.ipynb` lolos "Restart &
Run all" di runtime baru dan ekspor HTML-nya ada; `docs/progress.md` lengkap.

---

## Risiko dan cadangan

| Risiko | Cadangan |
|---|---|
| Kuota GPU Colab tidak cukup | Selesaikan Tier 1 dulu; kurangi E (sama untuk semua run) atau batasi tile per epoch; jangan kurangi seed Tier 1; diskusikan |
| Hitung prior terlalu lama (±14 jam-inti) | Jalankan paralel di laptop multi-core atau beberapa sesi CPU Colab; cadangan: opening biasa (dicatat) |
| Hasil deduplikasi jauh dari ±30 rb | Laporkan angka kita apa adanya; periksa 50 klaster acak secara visual |
| Reproduksi HiSup tidak cocok | Cek versi pycocotools, file anotasi (`annotation.json` vs `-small`), pemakaian skor; laporkan selisihnya |
| Head θ tidak membantu | Tetap laporkan; analisis per bentuk bangunan (IoU kotak momen) untuk melihat kapan θ membantu |
| Unduhan WHU lambat atau gagal | Fokus ke CrowdAI; WHU menjadi opsional |
| `boundary-iou-api` gagal di-install | Pin commit tertentu; cadangan: implementasi sendiri sesuai paper + tes |
| Ultralytics bermasalah | FULL+Y memang opsional; lewati |
| Ruang Drive penuh | Prior `uint8` PNG; data di-tar; hapus probabilitas mentah setelah evaluasi (konfirmasi dulu) |

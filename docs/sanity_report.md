# Laporan uji kewarasan pipeline evaluasi

Dibuat otomatis oleh `scripts/sanity_checks.py` dari `outputs/metrics/sanity_report.json`; jangan diedit tangan. Uji dan targetnya: `docs/EVALUATION.md` §6.

- Dibuat: 2026-10-30T14:13:47+08:00, commit `3c65809fc85a99516100f977c0aa1a4b01e7e7cf`.
- Mesin: Windows-11-10.0.26200-SP0, prosesor Intel64 Family 6 Model 170 Stepping 4, GenuineIntel.
- Versi: python 3.12.0, geos 3.14.1, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0, scipy 1.18.1, shapely 2.2.0, pycocotools 2.0.11, opencv-python-headless 4.14.0.94, multiprocess 0.70.19.
- Dijalankan ulang tersendiri: §6.3 pada 2026-10-30T22:10:38+08:00, commit `550f81ee6679aec42f883d26de7dbebed9ed5e73`.
- **Hasil: 55 dari 55 uji lolos.**

## §6.1 GT sebagai prediksi

(i) GT bermask kosong dikeluarkan dari GT dan prediksi; (ii) GT apa adanya.

### crowdai/val_clean

212 citra, 1786 anotasi GT, 17 GT bermask kosong. Pemeriksaan prediksi (ii): 16 ring dengan < 3 titik berbeda (mask kosong bagi pycocotools, tidak dipakai PoLiS/MTA), 0 ring tak terbaca dibuang.

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| (i) AP | 1 | 1 ± 0,001 | lolos |
| (i) AP_boundary | 1 | 1 ± 0,001 | lolos |
| (i) CIoU | 1 | 1 (± 1e-6) | lolos |
| (i) CIoU_p2p | 1 | 1 (± 1e-6) | lolos |
| (i) PoLiS vendor | None | None, 0 citra valid | lolos |
| (i) PoLiS_paired | 0 | 0 | lolos |
| (i) MTA_inst (°) | 0.0028111 | ≈ 0 (≤ 0,01) | lolos |
| (i) MTA vendor (°) | 0.00281683 | ≤ 0,5 | lolos |
| (i) GT merged | 0 | = 0 (GT tumpang tindih) | lolos |
| (i) GT split | 0 | = 0 (GT tumpang tindih) | lolos |
| (ii) GT tak ter-match | 17 | = 17 (GT bermask kosong) | lolos |
| (ii) IoU pasangan terkecil | 1 | 1 | lolos |

### crowdai/test

2928 citra, 25145 anotasi GT, 181 GT bermask kosong. Pemeriksaan prediksi (ii): 159 ring dengan < 3 titik berbeda (mask kosong bagi pycocotools, tidak dipakai PoLiS/MTA), 0 ring tak terbaca dibuang.

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| (i) AP | 1 | 1 ± 0,001 | lolos |
| (i) AP_boundary | 1 | 1 ± 0,001 | lolos |
| (i) CIoU | 1 | 1 (± 1e-6) | lolos |
| (i) CIoU_p2p | 1 | 1 (± 1e-6) | lolos |
| (i) PoLiS vendor | None | None, 0 citra valid | lolos |
| (i) PoLiS_paired | 0 | 0 | lolos |
| (i) MTA_inst (°) | 0.00283218 | ≈ 0 (≤ 0,01) | lolos |
| (i) MTA vendor (°) | 0.00284302 | ≤ 0,5 | lolos |
| (i) GT merged | 20 | = 20 (GT tumpang tindih) | lolos |
| (i) GT split | 20 | = 20 (GT tumpang tindih) | lolos |
| (ii) GT tak ter-match | 181 | = 181 (GT bermask kosong) | lolos |
| (ii) IoU pasangan terkecil | 1 | 1 | lolos |

### whu/test

35487 citra, 140689 anotasi GT, 0 GT bermask kosong. Pemeriksaan prediksi (ii): 0 ring dengan < 3 titik berbeda (mask kosong bagi pycocotools, tidak dipakai PoLiS/MTA), 0 ring tak terbaca dibuang.

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| (i) AP | 1 | 1 ± 0,001 | lolos |
| (i) AP_boundary | 1 | 1 ± 0,001 | lolos |
| (i) CIoU | 1 | 1 (± 1e-6) | lolos |
| (i) CIoU_p2p | 1 | 1 (± 1e-6) | lolos |
| (i) PoLiS vendor | None | None, 0 citra valid | lolos |
| (i) PoLiS_paired | 0 | 0 | lolos |
| (i) MTA_inst (°) | 0.00282488 | ≈ 0 (≤ 0,01) | lolos |
| (i) MTA vendor (°) | 0.00282741 | ≤ 0,5 | lolos |
| (i) GT merged | 0 | = 0 (GT tumpang tindih) | lolos |
| (i) GT split | 0 | = 0 (GT tumpang tindih) | lolos |
| (ii) GT tak ter-match | 0 | = 0 (GT bermask kosong) | lolos |
| (ii) IoU pasangan terkecil | 1 | 1 | lolos |

### whu/test (p2p)

35487 citra, 139962 anotasi GT, 0 GT bermask kosong. Pemeriksaan prediksi (ii): 0 ring dengan < 3 titik berbeda (mask kosong bagi pycocotools, tidak dipakai PoLiS/MTA), 0 ring tak terbaca dibuang.

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| (i) AP | 1 | 1 ± 0,001 | lolos |
| (i) AP_boundary | 1 | 1 ± 0,001 | lolos |
| (i) CIoU | 1 | 1 (± 1e-6) | lolos |
| (i) CIoU_p2p | 1 | 1 (± 1e-6) | lolos |
| (i) PoLiS vendor | None | None, 0 citra valid | lolos |
| (i) PoLiS_paired | 0 | 0 | lolos |
| (i) MTA_inst (°) | 0.00284747 | ≈ 0 (≤ 0,01) | lolos |
| (i) MTA vendor (°) | 0.00285352 | ≤ 0,5 | lolos |
| (i) GT merged | 0 | = 0 (GT tumpang tindih) | lolos |
| (i) GT split | 0 | = 0 (GT tumpang tindih) | lolos |
| (ii) GT tak ter-match | 0 | = 0 (GT bermask kosong) | lolos |
| (ii) IoU pasangan terkecil | 1 | 1 | lolos |

## §6.2 Kasus tepi

Tes unit (data sintetis) yang dijalankan skrip ini:

- `tests/test_pixel.py::test_edge_cases_of_docs_evaluation_6_2`: lolos
- `tests/test_run_eval.py::test_no_prediction_at_all`: lolos
- `tests/test_pix2poly_metrics.py::test_ciou_reset_changes_only_the_iou_of_an_image_without_gt`: lolos
- `tests/test_coco_io.py::test_each_problem_is_an_error_in_strict_mode`: lolos
- `tests/test_coco_io.py::test_lenient_mode_drops_only_rings_it_cannot_read`: lolos
- `tests/test_run_eval.py::test_evaluating_twice_gives_identical_numbers`: lolos

- crowdai/val_clean: evaluasi dua kali identik: lolos

## §6.3 Oracle post-processing

Di crowdai/val_tune_sub (1000 citra, 8499 anotasi GT), semua prediksi berskor 1.

(a) Bolak-balik poligon GT → raster → kontur marching squares → raster: 8444 anotasi bermask tidak kosong, IoU rata-rata 0.997561, terkecil 0.295302; 101 di bawah 1, 86 di bawah 0,99.

(b) Peta instance GT (label training) dijadikan poligon: kontur marching squares persis (batas atas tahap poligon) dan ketiga mode. (c) Peta 3 kelas GT dijadikan probabilitas sempurna → marker → watershed → kontur persis, sehingga yang diuji hanya tahap instance; varian `raw` ditampilkan sebagai informasi, karena DP-nya sendiri sudah menurunkan AP (bandingkan dua baris `raw`). Parameter: contact_radius_px 2, tau 0.4, marker_min_px 8, instance_min_px 10, raw_eps_px 1.0, rect_iou 0.9, dp_k 0.04, snap_deg 15.0, oblique_min_px 6.0, edge_min_px 2.0.

| Run | AP | AP50 | AP75 | AP_boundary | C-IoU | PoLiS_paired | MTA_inst (°) | recall@0,5 | merge | fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| (b) mask GT → kontur persis | 0.980878 | 0.98937 | 0.976566 | 0.980679 | 0.483181 | 0.283649 | 27.2833 | 0.992705 | 0.000705965 | 0 |
| (b) mask GT → `raw` | 0.934114 | 0.971293 | 0.954238 | 0.923413 | 0.952363 | 0.293192 | 14.7207 | 0.982469 | 0.000705965 | 0 |
| (b) mask GT → `mar` | 0.924113 | 0.989117 | 0.974052 | 0.815193 | 0.866551 | 0.544365 | 21.1317 | 0.992352 | 0.000705965 | 181 |
| (b) mask GT → `theta` (θ poligon GT) | 0.928044 | 0.989117 | 0.973809 | 0.823059 | 0.869373 | 0.531586 | 20.3154 | 0.992352 | 0.000705965 | 166 |
| (c) 3 kelas GT → watershed → kontur persis | 0.965314 | 0.970681 | 0.967889 | 0.965085 | 0.4828 | 0.282676 | 27.3283 | 0.985645 | 0.000941287 | 0 |
| (c) 3 kelas GT → watershed → `raw` | 0.925372 | 0.954242 | 0.948462 | 0.916577 | 0.953817 | 0.292992 | 14.738 | 0.977762 | 0.000941287 | 0 |

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| (a) IoU bolak-balik rata-rata | 0.997561 | ≥ 0.99 | lolos |
| (c) AP marker + watershed dari probabilitas sempurna (kontur persis) | 0.965314 | ≥ 0.95 | lolos |

## §6.4 Reproduksi HiSup

Prediksi rilisan HiSup dievaluasi ulang dengan pipeline ini (`metrics/crowdai/hisup_release/ref/val_full.json`, dibuat 2026-10-30T14:10:17+08:00, commit `3c65809fc85a99516100f977c0aa1a4b01e7e7cf`). Pemeriksaan longgar: 506978 prediksi, 0 dibuang, 2890 ring dua titik (ikut AP sebagai mask kosong, dikeluarkan dari PoLiS/MTA). C-IoU dan IoU paper mengikuti aturan `cIoU.py` (citra yang punya prediksi), jadi dibandingkan dengan `CIoU_p2p` dan `IoU_p2p`. Angka pipeline ini ditulis dengan satu desimal lebih banyak daripada paper.

Angka paper (`configs/paper_numbers.yaml`): Xu, Xu, Xue & Xia, "Accurate Polygonal Mapping of Buildings in Satellite Imagery" (HiSup), arXiv:2208.00609v1, 1 Aug 2022; Table 1 (MS-COCO metrics) and Table 2 (AP_boundary, PoLiS, C-IoU, IoU); hlm. 9 (page of the arXiv PDF).

| Metrik | Paper | Pipeline ini | Selisih |
|---|---|---|---|
| AP | 79.4 | 79.43 | +0.03 |
| AP50 | 92.7 | 92.68 | -0.02 |
| AP75 | 85.3 | 85.31 | +0.01 |
| AR | 81.5 | 81.55 | +0.05 |
| AP_boundary | 66.5 | 66.51 | +0.01 |
| PoLiS | 0.726 | 0.7261 | +0.0001 |
| CIoU | 89.6 | 89.67 | +0.07 |
| IoU | 94.3 | 94.28 | -0.02 |

Varian lain dari pipeline ini: CIoU (semua citra) 89.45, IoU (ΣI/ΣU) 95.20, MTA (°) 31.94.

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| AP HiSup: rilisan dievaluasi ulang vs paper | 79.432 | 79.4 ± 0.3 | lolos |

## §6.6 Bootstrap AP

Prediksi buatan di crowdai/val_clean: GT digeser acak ±2 px, 10% dibuang, skor acak. AP titik 0.517563; 1000 replikasi: CI 95% [0.499681, 0.542429], SD 0.0107799.
Cara naif (imgIds duplikat langsung ke COCOeval, 200 replikasi pertama): rata-rata hanya 63.3% citra unik yang tersisa, CI 95% [0.504364, 0.537798] versus [0.497541, 0.544322] dengan cara yang benar (rasio lebar 0.71).

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| resampel identitas = stats[0] | 0.517563 | = 0.5175628780638957 | lolos |
| CI 95% memuat nilai titik | [0.499681, 0.542429] | ∋ 0.5176 | lolos |
| lebar CI cara naif / cara benar | 0.714693 | ≠ 1 (naif membuang duplikat) | lolos |

# Laporan uji kewarasan pipeline evaluasi

Dibuat otomatis oleh `scripts/sanity_checks.py` dari `outputs/metrics/sanity_report.json`; jangan diedit tangan. Uji dan targetnya: `docs/EVALUATION.md` §6.

- Dibuat: 2026-10-30T12:07:51+08:00, commit `1ddd931a0ddce033bfe365c473b93786df69965a`.
- Mesin: Windows-11-10.0.26200-SP0, prosesor Intel64 Family 6 Model 170 Stepping 4, GenuineIntel.
- Versi: python 3.12.0, geos 3.14.1, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0, scipy 1.18.1, shapely 2.2.0, pycocotools 2.0.11, opencv-python-headless 4.14.0.94, multiprocess 0.70.19.
- **Hasil: 52 dari 52 uji lolos.**

## §6.1 GT sebagai prediksi

(i) GT bermask kosong dikeluarkan dari GT dan prediksi; (ii) GT apa adanya.

### crowdai/val_clean

212 citra, 1786 anotasi GT, 17 GT bermask kosong. Pemeriksaan prediksi (ii): 16 ring < 3 titik dibuang.

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

2928 citra, 25145 anotasi GT, 181 GT bermask kosong. Pemeriksaan prediksi (ii): 159 ring < 3 titik dibuang.

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

35487 citra, 140689 anotasi GT, 0 GT bermask kosong. Pemeriksaan prediksi (ii): 0 ring < 3 titik dibuang.

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

35487 citra, 139962 anotasi GT, 0 GT bermask kosong. Pemeriksaan prediksi (ii): 0 ring < 3 titik dibuang.

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

## §6.6 Bootstrap AP

Prediksi buatan di crowdai/val_clean: GT digeser acak ±2 px, 10% dibuang, skor acak. AP titik 0.517563; 1000 replikasi: CI 95% [0.499681, 0.542429], SD 0.0107799.
Cara naif (imgIds duplikat langsung ke COCOeval, 200 replikasi pertama): rata-rata hanya 63.3% citra unik yang tersisa, CI 95% [0.504364, 0.537798] versus [0.497541, 0.544322] dengan cara yang benar (rasio lebar 0.71).

| Uji | Nilai | Target | Hasil |
|---|---|---|---|
| resampel identitas = stats[0] | 0.517563 | = 0.5175628780638957 | lolos |
| CI 95% memuat nilai titik | [0.499681, 0.542429] | ∋ 0.5176 | lolos |
| lebar CI cara naif / cara benar | 0.714693 | ≠ 1 (naif membuang duplikat) | lolos |

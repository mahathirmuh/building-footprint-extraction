# third_party — kode pihak ketiga

| Path | Sumber | Versi | Lisensi | Perubahan |
|---|---|---|---|---|
| `pix2poly/whu_buildings_to_coco.py` | [Pix2Poly](https://github.com/yeshwanth95/Pix2Poly) `data_preprocess/whu_buildings_to_coco.py`; Pix2Poly sendiri meminjamnya dari [HiSup](https://github.com/SarahwXU/HiSup) `tools/inria_to_coco.py` | commit `4e4c003` (cabang `master`), diambil 28 Okt 2026 | MIT: `licenses/Pix2Poly-LICENSE.txt`, `licenses/HiSup-LICENSE.txt` | tidak ada (salinan persis, sha256 `c9930d04abce…`) |

`pix2poly/whu_buildings_to_coco.py` tidak dipakai oleh kode utama. File ini hanya menjadi pembanding di
`tests/test_whu.py`, yang membuktikan bahwa port di `src/densepoly/data/whu.py` menghasilkan poligon yang identik.

Skrip metrik Pix2Poly (`pix2poly_metrics/`) ditambahkan di Fase 3, dengan catatan perubahan pembungkus di header tiap
file (`docs/EVALUATION.md` §2.3).

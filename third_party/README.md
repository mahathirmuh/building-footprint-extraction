# third_party — kode pihak ketiga

| Path | Sumber | Versi | Lisensi | Perubahan |
|---|---|---|---|---|
| `pix2poly/whu_buildings_to_coco.py` | [Pix2Poly](https://github.com/yeshwanth95/Pix2Poly) `data_preprocess/whu_buildings_to_coco.py`; Pix2Poly sendiri meminjamnya dari [HiSup](https://github.com/SarahwXU/HiSup) `tools/inria_to_coco.py` | commit `4e4c003` (cabang `master`), diambil 28 Okt 2026 | MIT: `licenses/Pix2Poly-LICENSE.txt`, `licenses/HiSup-LICENSE.txt` | tidak ada (salinan persis, sha256 `c9930d04abce…`) |
| `pix2poly/{polis,angle_eval,cIoU}.py` | Pix2Poly `eval/hisup_eval_utils/metrics/` | commit `4e4c00359b41f31fe80f7fbdab0fa50eb7a46ab8`, diambil 30 Okt 2026 | lihat catatan lisensi di bawah | tidak ada (salinan persis; sha256 `ab845034a848…`, `1007612195147…`, `134f4f29af84…`) |
| `pix2poly_metrics/{polis,angle_eval,cIoU}.py` | salinan di atas | sama | sama | hanya pembungkus (`docs/EVALUATION.md` §2.3), dicatat di header tiap file dan ditandai `[wrapper N]` |

`pix2poly/` berisi salinan persis yang tidak dipakai kode utama. File-file itu hanya menjadi pembanding di tes:
- `tests/test_whu.py` membuktikan port di `src/densepoly/data/whu.py` menghasilkan poligon yang identik.
- `tests/test_pix2poly_metrics.py` membuktikan versi `pix2poly_metrics/` memberi angka identik dengan aslinya di
  semua kasus yang tidak terkena perubahan pembungkus.

`pix2poly_metrics/` adalah paket Python (`import pix2poly_metrics`; `pyproject.toml` memakai `third_party/` sebagai
akar paket kedua) dan dipakai oleh `src/densepoly/eval/`.

**Catatan lisensi skrip metrik** (dicek 30 Okt 2026). Pix2Poly (MIT) menyebut asal setiap file di docstring-nya:
- `angle_eval.py`: [Polygonization-by-Frame-Field-Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning),
  BSD-3-Clause.
- `cIoU.py`: [PolyWorldPretrainedNetwork](https://github.com/zorzi-s/PolyWorldPretrainedNetwork). Lisensinya khusus
  (BlackShark.ai): hanya untuk riset akademik, tanpa hak sublisensi. Pemakaian di tugas akhir ini termasuk riset
  akademik, tetapi file ini tidak boleh ikut dipublikasikan di bawah lisensi lain.
- `polis.py`: `spgriffin/polis`. Repo itu tidak ditemukan lagi di GitHub, jadi lisensinya tidak diketahui.

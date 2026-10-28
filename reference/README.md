# reference/ — Kode Demo Tahap ETS

Kode ini dibuat untuk gambar slide ETS. **Hanya referensi**: logikanya boleh di-port ke `src/densepoly/` mengikuti
`docs/METHOD.md`, tetapi jangan diimpor dari kode utama.

| File | Isi | Hasil di slide | Beda dengan spesifikasi final |
|---|---|---|---|
| `demo/before_after.py` | adegan sintetis 25 rumah, output jaringan disimulasikan, `split_instances` (marker + watershed), `regularize` | slide 19: 5 poligon (±72 verteks) → 25 poligon (4 verteks) | kontak r = 3 dan ikut mencakup celah latar; ambang persegi 0,75; kontur `cv2.findContours` (menyusut, lihat METHOD §6.1) |
| `demo/label_demo.py` | panel mask biner, mask instance, θ, anotasi poligon | slide 16 | ilustrasi saja |
| `demo/prior_demo.py` | MBI sederhana, peta celah, B0 pada adegan sintetis | B0 menemukan 14 dari 25 rumah | MBI disederhanakan (bukan opening by reconstruction + DMP); peta celah mode `max` |
| `demo/real_prior.py` | SpaceNet 6 Rotterdam: pansharpen high-pass, MBI, peta celah, B0 | slide 17: MBI menyalakan jalan, atap gelap terlewat | pansharpen hanya untuk demo; skala MBI 21–81 px pada citra 0,5 m |

## prototypes/

Dibuat saat menyusun dokumen Claude Code untuk membuktikan spesifikasi bisa dicapai. Bukan kode produksi.

| File | Isi |
|---|---|
| `prototypes/method_checks.py` | bolak-balik kontur marching squares (IoU 1,000 vs `cv2` 0,84), filter kontak, tanda sin 4θ saat flip |
| `prototypes/regularize_proto.py` | regularisasi `docs/METHOD.md` §7: kotak momen orde-2, pelurusan dengan peleburan sisi pendek; uji bentuk kotak, L, T, dinding miring, kotak kecil |

Kebutuhan: `numpy scipy scikit-image opencv-python-headless shapely pycocotools`.

## Cara menjalankan demo

Cara menjalankan (opsional): `python reference/demo/before_after.py <folder_output>`;
`real_prior.py` butuh `<folder_sumber_spacenet6> <folder_output>`.

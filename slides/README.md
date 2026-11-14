# Slides (gaya template ITS Teknik Informatika)

- `densepoly.pptx`: deck presentasi (18 slide, 16:9). Font yang dipakai: Raleway dan Raleway Black (Google Fonts, gratis).
- `build.js`: isi deck. Tidak ada angka hasil yang diketik: semua dibaca dari `numbers.json`.
- `numbers.json`: dibuat `python scripts/paper_tables.py` dari tabel hasil, sama dengan angka di paper.
- `its_kit.js`: kit tata letak (pptxgenjs).

Bangun ulang:

    python scripts/figures.py --data crowdai --results --qualitative --lang id
    python scripts/paper_tables.py
    NODE_PATH=<folder node_modules yang berisi pptxgenjs> node slides/build.js

Logo resmi ITS (header, footer, logo IF di sampul, ornamen penutup) sudah ada di `densepoly.pptx`. File gambarnya dibongkar dari
skill `ppt-its-assets` ke `slides/its_assets/` (tidak masuk git). Tanpa folder itu, `build.js` tetap berjalan dengan header dan
footer teks. Foto latar slide penutup tidak tersedia, jadi slide penutup memakai warna biru ITS polos.

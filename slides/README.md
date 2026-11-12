# Slides (gaya template ITS Teknik Informatika)

- `densepoly.pptx`: deck presentasi (18 slide, 16:9). Font yang dipakai: Raleway dan Raleway Black (Google Fonts, gratis).
- `build.js`: isi deck. Tidak ada angka hasil yang diketik: semua dibaca dari `numbers.json`.
- `numbers.json`: dibuat `python scripts/paper_tables.py` dari tabel hasil, sama dengan angka di paper.
- `its_kit.js`: kit tata letak (pptxgenjs).

Bangun ulang:

    python scripts/figures.py --data crowdai --results --qualitative --lang id
    python scripts/paper_tables.py
    NODE_PATH=<folder node_modules yang berisi pptxgenjs> node slides/build.js

Logo resmi ITS tidak disertakan. Taruh file logo di `slides/its_assets/` (nama file mengikuti slot di `its_kit.js`, misalnya
`header_left.png`), lalu bangun ulang; tanpa itu header dan footer memakai teks.

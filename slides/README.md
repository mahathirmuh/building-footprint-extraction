# Slides (gaya template ITS Teknik Informatika)

- `densepoly.pptx`: deck presentasi (19 slide, 16:9). Font yang dipakai: Raleway dan Raleway Black (Google Fonts, gratis).
- `densepoly.pdf`: deck yang sama sebagai PDF (19 halaman), untuk dilihat tanpa PowerPoint. Dikonversi dengan LibreOffice
  di GitHub Actions (`.github/workflows/slides.yml`), dengan font Raleway terpasang. Ini hasil render LibreOffice, jadi
  detail kecil bisa berbeda dari PowerPoint (misalnya huruf θ, yang tidak ada di Raleway dan memakai font pengganti).
- `build.js`: isi deck. Tidak ada angka hasil yang diketik: semua dibaca dari `numbers.json`.
- `naskah.md`: naskah ucapan per slide, untuk presentasi ke dosen dan rekan mahasiswa. Ditulis `build.js` dari `say(...)`
  di samping tiap slide, dengan angka dari `numbers.json`; build berhenti bila ada slide tanpa naskah.
- `numbers.json`: dibuat `python scripts/paper_tables.py` dari tabel hasil, sama dengan angka di paper.
- `its_kit.js`: kit tata letak (pptxgenjs).

Bangun ulang:

    python scripts/figures.py --data crowdai --results --qualitative --lang id
    python scripts/paper_tables.py
    NODE_PATH=<folder node_modules yang berisi pptxgenjs> node slides/build.js

Setiap push yang mengubah `densepoly.pptx` dikonversi ulang di GitHub; hasilnya ada di cabang `slides-pdf` (hanya keluaran
build). Salin PDF-nya ke sini:

    git fetch origin slides-pdf
    git show FETCH_HEAD:exit_code.txt          # 0 = berhasil; fonts.txt memuat font yang tertanam di PDF
    git show FETCH_HEAD:densepoly.pdf > slides/densepoly.pdf

Logo resmi ITS (header, footer, logo IF di sampul, ornamen penutup) sudah ada di `densepoly.pptx`. File gambarnya dibongkar dari
skill `ppt-its-assets` ke `slides/its_assets/` (tidak masuk git). Tanpa folder itu, `build.js` tetap berjalan dengan header dan
footer teks. Foto latar slide penutup tidak ada di paket itu: `its_assets/closing_bg.jpg` dipulihkan dari tangkapan layar halaman
penutup template kiriman pemilik proyek (19 Nov 2026; logo, ornamen, dan teks di atas foto dihapus dengan *inpainting*), jadi
lebih lunak dari foto aslinya. Tanpa file itu slide penutup memakai warna biru ITS polos.

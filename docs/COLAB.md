# COLAB.md — Training di Google Colab dan Pekerjaan CPU Berat

Claude Code menulis dan menguji kode di lokal (CPU, data sintetis). **Colab hanya untuk training** (T4 gratis;
diputuskan 31 Okt 2026), lewat `notebooks/colab_train.ipynb`. Notebook hanya berisi sel tipis yang memanggil skrip di
repo. Logika tidak boleh ditulis di notebook. `best.pt` lalu dibawa ke laptop, dan inferensi, tuning, serta evaluasi
berjalan di sana (§5). Pekerjaan CPU berat (hash, prior, evaluasi besar, tuning) dibahas di §8.

## 1. Tata letak Google Drive

```
MyDrive/densepoly/
├── data_tar/
│   ├── crowdai_train_dedup.tar      # citra train_dedup + val_tune + cache instance + cache prior (±3,4 GB)
│   └── whu_pix2poly_224.tar         # tile resmi + annotation*.json + patches.csv (dibuat di Fase 7)
└── outputs/                         # = DENSEPOLY_OUT di Colab (runs/)
```

Tar dibuat sekali di lokal atau di sesi Colab CPU. Membaca ribuan file kecil langsung dari Drive sangat lambat, jadi
data selalu disalin ke disk lokal `/content/data` dulu.

## 2. Urutan sel notebook

1. **GPU:** `!nvidia-smi` (catat nama GPU ke log run).
2. **Mount Drive:** `from google.colab import drive; drive.mount('/content/drive')`.
3. **Clone repo:** repo publik (31 Okt 2026), jadi `git clone` tanpa token.
4. **Install:** `pip install -e .`. Skrip metrik (Pix2Poly dan boundary-iou-api) sudah di-vendor di `third_party/`,
   jadi tidak perlu instalasi tambahan. Jangan meng-install ulang `torch`; pakai versi bawaan Colab dan catat versinya.
5. **Data:** tar dibuat sekali di laptop dengan `python scripts/pack_colab.py --data crowdai`, lalu tar dan manifest
   `.json`-nya diunggah ke `MyDrive/densepoly/data_tar/`. Notebook menyalinnya ke disk lokal, memeriksa salinannya
   (`--verify`), lalu mengekstrak ke `/content/data`. Perintah `!` yang gagal tidak menghentikan Run all, jadi sebuah
   `assert` menghentikan notebook bila tar belum ada atau rusak. Salinan yang rusak dihapus, dan Run all berikutnya
   menyalinnya lagi.
6. **Environment:** `DENSEPOLY_DATA=/content/data`, `DENSEPOLY_OUT=/content/drive/MyDrive/densepoly/outputs`.
7. **Benchmark dulu:** `python scripts/train.py --data crowdai --exp full_p2 --benchmark 200` → it/s dan memori
   puncak. Isi tabel biaya di `docs/EXPERIMENTS.md` §5.
8. **Training:** `python scripts/train.py --data crowdai --exp <exp> --seed <k> --resume --max-hours <H>
   train.epochs=<E>`. Colab gratis tidak menjamin lama sesi, jadi `H` dibuat pendek dan run dilanjutkan di sesi berikutnya.
9. **Salin `best.pt` ke laptop** dari `outputs/runs/<data>/<exp>/<run>/` di Drive. Prediksi dan evaluasi berjalan di
   laptop (§5).

## 3. Resume dan putus koneksi

- `last.pt` disimpan ke Drive tiap epoch **dan** tiap 20 menit. Isinya: model, optimizer, scheduler, GradScaler, epoch,
  iterasi, status RNG, metrik terbaik, dan config. Simpan secara atomik (tulis file sementara, lalu rename).
- `--resume` melanjutkan dari `last.pt` bila ada. `--max-hours` menghentikan training dengan rapi sebelum batas sesi.
- Uji sekali di Fase 5: hentikan runtime di tengah epoch, sambung lagi, lalu pastikan kurva loss berlanjut mulus.

## 4. Kinerja

- AMP fp16, `cudnn.benchmark=True`, `num_workers=2`, `pin_memory=True`, `persistent_workers=True`.
- Label diturunkan on-the-fly dari cache instance (murah). Prior dibaca dari cache, tidak dihitung saat training.
- Batch 16. Bila memori kurang, pakai gradient accumulation (misalnya 8 × 2). Micro-batch tetap minimal 8, karena
  akumulasi tidak memperbesar batch untuk statistik BatchNorm.
- Validasi tiap epoch hanya IoU foreground (cepat). Metrik poligon dihitung setelah training.

## 5. Inferensi dan tuning post-processing

- Semuanya berjalan di laptop, jadi data `test` tidak perlu diunggah ke Drive. Inferensi di CPU laptop ±0,60 s per
  tile (12 thread, diukur 31 Okt 2026): `val_tune_sub` ±10 menit, `test` ±30 menit.
- `scripts/predict.py --save-channels` menyimpan keluaran model `float16` per citra di `outputs/channels/`, untuk
  `val_tune_sub` (1.000 citra, ±1 GB), sehingga tuning tidak perlu menjalankan model berulang kali. B5 dan B5′ memakai
  ulang kanal model sumbernya lewat `--from-channels`.
- Grid tuning dijalankan paralel dengan `multiprocessing`. Ukur dulu waktunya di sampel (§8).
- Hapus probabilitas mentah setelah tuning dan evaluasi selesai (minta konfirmasi dulu).

## 6. FULL+Y (Ultralytics)

Jalankan di runtime terpisah, karena `pip install ultralytics` bisa mengubah versi paket lain. Bobot `yolov8n-obb.pt`
diunduh otomatis oleh Ultralytics. Simpan hasil deteksi ke JSON, lalu penggabungan seed dijalankan di lingkungan
utama.

## 7. Daftar cek sebelum training panjang (minta persetujuan saya)

Daftar ini juga berlaku, sejauh relevan, untuk pekerjaan CPU berat di §8.

- [ ] `pytest -q` lolos di commit yang akan dijalankan; hash git dicatat.
- [ ] Config final dicetak dan sesuai `docs/EXPERIMENTS.md` (eksperimen, seed, E, batch, lr).
- [ ] Benchmark it/s sudah ada, estimasi jam GPU masuk kuota.
- [ ] Split dan cache prior yang dipakai sesuai versi di `splits/` dan `configs/data/`.
- [ ] Resume sudah teruji.

## 8. Pekerjaan CPU berat (tanpa GPU)

| Pekerjaan | Fase | Estimasi (1 inti, diukur saat dokumen disusun) | Tempat yang disarankan |
|---|---|---|---|
| p-hash 6 varian CrowdAI (341 rb citra) | 1 | ±0,6 jam + waktu baca disk | laptop (data lokal) atau sesi CPU Colab |
| Prior MBI + peta celah (±35 rb tile) | 2 | ±1,4 s/tile → ±14 jam | laptop multi-core (`multiprocessing`), atau dibagi ke beberapa sesi CPU Colab per split |
| Evaluasi val penuh HiSup (60 rb citra; AP, AP_boundary, PoLiS, MTA) | 3 | `TODO` (ukur di 1.000 citra) | laptop atau sesi CPU Colab |
| Tuning post-processing (72 kombinasi × 1.000 citra per model) | 4–6 | `TODO` (ukur) | laptop atau sesi CPU Colab, paralel |

Aturan:
- Ukur dulu di sampel kecil (50 tile / 1.000 citra), laporkan estimasi totalnya, lalu minta persetujuan.
- Skrip harus bisa dilanjutkan bila terputus (lewati file output yang sudah ada) dan bisa dipecah per split atau per
  rentang indeks (`--shard i/n`).
- Hasil ditulis ke `DENSEPOLY_DATA` (cache) atau `DENSEPOLY_OUT` (metrik), lalu di-tar untuk Colab.

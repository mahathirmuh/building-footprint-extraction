# COLAB.md — Training di Google Colab dan Pekerjaan CPU Berat

Claude Code menulis dan menguji kode di lokal (CPU, data sintetis). Training dan inferensi penuh dijalankan di Colab
lewat `notebooks/colab_train.ipynb`. Notebook hanya berisi sel tipis yang memanggil skrip di repo. Logika tidak
boleh ditulis di notebook. Pekerjaan CPU berat (hash, prior, evaluasi besar, tuning) dibahas di §8.

## 1. Tata letak Google Drive

```
MyDrive/densepoly/
├── data_tar/
│   ├── crowdai_train_dedup.tar      # citra train_dedup + val_tune + cache instance + cache prior
│   ├── crowdai_clean_test.tar       # citra test bersih + cache
│   ├── crowdai_gt.tar               # anotasi GT, file GT subset, prediksi HiSup
│   └── whu_pix2poly_224.tar         # bila protokol A dipakai
└── outputs/                         # = DENSEPOLY_OUT (runs/, preds/, metrics/, tables/, figures/)
```

Tar dibuat sekali di lokal atau di sesi Colab CPU. Membaca ribuan file kecil langsung dari Drive sangat lambat, jadi
data selalu disalin ke disk lokal `/content/data` dulu.

## 2. Urutan sel notebook

1. **GPU:** `!nvidia-smi` (catat nama GPU ke log run).
2. **Mount Drive:** `from google.colab import drive; drive.mount('/content/drive')`.
3. **Clone repo privat:** token GitHub disimpan di Colab Secrets (`GITHUB_TOKEN`), dibaca dengan
   `google.colab.userdata.get`. Jangan pernah mencetak token atau menyimpannya di notebook.
4. **Install:** `pip install -e ".[eval]"`. Jangan meng-install ulang `torch`; pakai versi bawaan Colab dan catat
   versinya.
5. **Data:** salin tar dari Drive ke `/content/data`, lalu ekstrak.
6. **Environment:** `DENSEPOLY_DATA=/content/data`, `DENSEPOLY_OUT=/content/drive/MyDrive/densepoly/outputs`.
7. **Benchmark dulu:** `python scripts/train.py --data crowdai --exp full_p2 --benchmark 200` → it/s dan memori
   puncak. Isi tabel biaya di `docs/EXPERIMENTS.md` §5.
8. **Training:** `python scripts/train.py --data crowdai --exp <exp> --seed <k> --resume --max-hours 11`.
9. **Prediksi dan evaluasi:** `scripts/predict.py` (GPU), lalu `scripts/evaluate.py` (CPU, bisa di sesi lain).

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

- `scripts/predict.py --save-probs` menyimpan probabilitas `float16` (`.npz`) untuk `val_tune_sub` (1.000 citra,
  ±1 GB), sehingga tuning post-processing (CPU) tidak perlu menjalankan model berulang kali.
- Grid tuning dijalankan paralel dengan `multiprocessing`. Colab gratis hanya punya ±2 vCPU; ukur dulu waktunya (§8).
  Bisa juga dijalankan di laptop.
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

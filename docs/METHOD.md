# METHOD.md — Spesifikasi Metode

Dokumen ini adalah **spesifikasi** yang diikuti kode. Nilai default bisa disetel di `val_tune` sesuai §10. Perubahan
definisi (bukan sekadar nilai parameter) harus didiskusikan dulu dan dicatat di `docs/progress.md`.

## 0. Notasi dan konvensi

- Citra berukuran H × W. Indeks piksel numpy `(r, c)`. Koordinat kontinu (COCO/shapely) `x = c`, `y = r`;
  piksel `(r, c)` menempati kotak `[c, c+1] × [r, r+1]`, pusatnya `(c + 0,5, r + 0,5)`.
- Kelas: `0` latar, `1` bangunan (bukan kontak), `2` kontak; `255` diabaikan (padding).
- θ: arah sisi utama bangunan di koordinat citra (sumbu y ke bawah), dalam rentang `[0°, 90°)`. Karena bangunan
  simetris setiap 90°, jaringan memprediksi `(sin 4θ, cos 4θ)`.
- `P_bg, P_b, P_k`: probabilitas softmax latar, bangunan, kontak.

## 1. Gambaran pipeline

```
RGB ──┬──────────────────────────────────────────────┐
      └─► brightness ─► MBI (kanal 4) ─┐              │
                     └─► peta celah (kanal 5) ─┐      │
                                               ▼      ▼
                         U-Net++ EfficientNet-B4 (5 kanal, kanal 4–5 bobot awal 0)
                                               │
                        ┌──────────────────────┴───────────────┐
                 softmax 3 kelas (P_bg, P_b, P_k)        tanh (sin 4θ, cos 4θ)
                        │                                      │
     marker = P_b − P_k > τ ; foreground = 1 − P_bg > 0,5      │
                        ▼                                      │
                watershed(−P_b) → instance                     │
                        ▼                                      ▼
         kontur marching squares ──► θ_i (rata-rata sirkular per instance)
                        ▼
   IoU(kotak momen, kontur) ≥ ρ ? ──ya──► kotak 4 verteks searah θ_i
                        └─tidak─► DP (ε = k·√A) + pelurusan sisi ke θ_i
                        ▼
         cek validitas / fallback → klip ke batas tile → COCO JSON / GeoJSON
```

## 2. Prior morfologi (kanal 4–5, kontribusi N3)

### 2.1 Brightness
`b = max(R, G, B) / 255` (float32, [0, 1]). Mengikuti Huang & Zhang (2011): nilai maksimum band tampak.

### 2.2 MBI (Huang & Zhang, 2011)
- Structuring element (SE) garis dengan panjang `s` piksel dan arah `d ∈ {0°, 45°, 90°, 135°}` (buat dengan
  `skimage.draw.line`, panjang ganjil, berpusat di tengah).
- White top-hat by reconstruction: `WTH(s, d) = b − γ_R(b; s, d)`, dengan `γ_R` = opening by reconstruction:
  `reconstruction(seed=erosion(b, SE), mask=b, method="dilation")` dari `skimage.morphology`.
- Differential morphological profile: `DMP(s, d) = |WTH(s + Δs, d) − WTH(s, d)|`.
- `MBI = mean_{d, s} DMP(s, d)`.
- Skala didefinisikan dalam **meter** (±3–25 m, sesuai ukuran rumah) lalu dikonversi ke piksel per dataset (§10):
  CrowdAI (±0,3 m) `s ∈ {11, 21, …, 81}` px (Δs = 10); WHU 0,2 m `s ∈ {15, 31, 45, 61, 75, 91, 105, 121}` px (15·k,
  dibulatkan ke ganjil karena SE garis harus berpusat). Disetujui setelah QA visual Fase 2 dan **dibekukan 29 Okt 2026**.
- Waktu terukur saat dokumen disusun: ±1,4 s per tile 320² untuk 32 opening by reconstruction. Kalau di mesinmu jauh
  lebih lambat (> 3 s per tile), boleh memakai opening biasa (seperti demo ETS), **asal dicatat**.

### 2.3 Peta celah (black top-hat multi-arah)
- `BTH(d) = closing(b; SE garis L_g, d) − b`, dengan `L_g` ±2,7 m (CrowdAI 9 px, WHU 0,2 m 13 px): lebih panjang dari
  lebar celah antar-atap, lebih pendek dari lebar atap.
- Dua mode (pilih di Fase 2, lalu bekukan):
  - `max`: `gap = max_d BTH(d)` (sama dengan demo ETS).
  - `contrast`: `gap = max_d BTH(d) − min_d BTH(d)`. Garis gelap tipis hanya terisi oleh SE yang tegak lurus
    terhadapnya, sedangkan noda gelap bulat (bayangan pohon, tekstur vegetasi) terisi oleh semua arah. Jadi mode ini
    menekan derau isotropik dan mempertahankan celah memanjang.

### 2.4 Penyimpanan dan normalisasi
- Hitung sekali (`scripts/precompute_priors.py`) dari citra asli (sebelum augmentasi). Simpan
  `uint8(clip(v / q, 0, 1) · 255)`, dengan `q` = persentil 99,5 dari 500 tile train acak (per kanal), disimpan di
  `configs/data/<ds>_prior_stats.json` bersama mean/std hasil skala.
- Saat training: RGB → `/255` lalu normalisasi ImageNet (mean 0,485/0,456/0,406; std 0,229/0,224/0,225). Kanal prior
  → `(v/255 − μ)/σ` dari statistik train.
- Augmentasi geometri diterapkan bersama ke RGB dan prior. Augmentasi fotometrik hanya ke RGB dan dibuat ringan,
  karena prior dihitung dari citra asli.

### 2.5 Perluasan konvolusi pertama dengan bobot nol

```python
def expand_first_conv(model, n_in: int) -> None:
    """Extend the first 3-channel conv of model.encoder to n_in channels.
    RGB weights are kept unchanged; extra channels start at zero, so the
    expanded model is functionally identical to the RGB model at init."""
    conv = next(m for m in model.encoder.modules()
                if isinstance(m, nn.Conv2d) and m.in_channels == 3)
    w = conv.weight.detach()
    new_w = torch.zeros(w.shape[0], n_in, *w.shape[2:], dtype=w.dtype)
    new_w[:, :3] = w
    conv.in_channels = n_in
    conv.weight = nn.Parameter(new_w)
    model.encoder._in_channels = n_in
    model.encoder._out_channels = [n_in] + list(model.encoder._out_channels[1:])
```

Modul diganti bobotnya di tempat karena stem EfficientNet di smp adalah subclass `nn.Conv2d` dengan padding
"same" yang dihitung di `forward`. Tes wajib: untuk input `x` acak 5 kanal,
`model5(x) == model3(x[:, :3])` (toleransi 1e-5, mode eval).

**Alasan:** Maniyar dkk. (2025) mengganti RGB dengan komposit fitur sehingga fitur pretrained tidak cocok lagi dan
IoU turun. Dengan bobot nol, model mulai persis sama dengan model RGB, lalu belajar sendiri seberapa besar memakai
prior. Gradien bobot kanal prior tetap tidak nol karena inputnya tidak nol.

## 3. Label (diturunkan otomatis dari peta instance)

### 3.1 Peta instance `L` (`uint16`, 0 = latar)
- CrowdAI: rasterisasi tiap anotasi (`pycocotools` `annToMask`), digambar urut **luas menurun** sehingga bangunan
  kecil yang tumpang tindih tetap terlihat. Catat jumlah piksel yang tumpang tindih.
- WHU (opsi R): rasterisasi poligon **GT utama** (`annotation.json`) dengan `annToMask`, sama seperti CrowdAI,
  supaya label training konsisten dengan GT evaluasi. Poligon itu berasal dari mask per bangunan COCO resmi yang
  menjalani langkah poligon Pix2Poly, sehingga bangunan yang bersentuhan tetap terpisah (`docs/DATA.md` §3.2).
  GT `p2p` (CC dari label raster) tidak dipakai untuk training.
- WHU protokol B (0,3 m): CC dari mask biner dengan konektivitas penuh (8-tetangga, default `skimage.measure.label`).

### 3.2 Kelas 3 (kontak)
Piksel bangunan `p` dengan label `l` disebut kontak bila dalam radius `r` ada piksel berlabel lain selain 0 dan `l`.
Implementasi tanpa loop per pasangan:

```python
fp = disk(r)
big = np.iinfo(L.dtype).max
lmax = ndi.maximum_filter(L, footprint=fp)
lmin = ndi.minimum_filter(np.where(L > 0, L, big), footprint=fp)
contact = (L > 0) & ((lmax != L) | (lmin != L))
cls = np.where(L > 0, 1, 0); cls[contact] = 2
```

Default `r = 2`. Kontak dibatasi ke piksel bangunan, sehingga celah latar di antara rumah tetap kelas 0 dan
foreground (`1 − P_bg`) tetap sama dengan gabungan bangunan.

### 3.3 Orientasi θ (kontribusi N1)
Untuk setiap instance dengan luas ≥ `A_θ` (default 20 px):

```python
pts = np.column_stack([cols, rows]).astype(np.float32)   # (x, y)
box = cv2.boxPoints(cv2.minAreaRect(pts))
ex, ey = box[1] - box[0]
theta = np.arctan2(ey, ex) % (np.pi / 2)
```

Isi `S = sin 4θ`, `C = cos 4θ` di piksel instance tersebut, dan mask valid `V` = piksel instance dengan luas ≥ `A_θ`.
Nilai sudut bawaan `minAreaRect` **tidak dipakai**, karena konvensinya berubah antar-versi OpenCV. Vektor sisi dari
`boxPoints` tidak bergantung versi, dan faktor 4 membuat pilihan sisi mana pun memberi hasil yang sama.

### 3.4 Bobot tepi
Piksel batas `B` = piksel yang labelnya berbeda dari minimal satu tetangga-4 (termasuk transisi ke latar).
`d = distance_transform_edt(~B)`, lalu `w = 1 + α·exp(−d² / (2σ²))`, default `α = 5`, `σ = 3` px; `w = 0` di
piksel 255.

### 3.5 Urutan augmentasi
1. Augmentasi geometri acak (flip H, flip V, rot90 k, transpose) pada RGB, prior, dan `L` (nearest).
2. Turunkan kelas, θ, `V`, dan `w` dari `L` hasil augmentasi.
3. Padding ke kelipatan 32 (citra: reflect; `L` dan kelas: 255; `w = 0`; `V = 0`).

Tabel berikut dipakai sebagai **tes invarian** (label hasil turunan ulang harus sama dengan transformasi label):

| Transformasi | θ menjadi | sin 4θ | cos 4θ |
|---|---|---|---|
| rot90 (k berapa pun) | θ ± 90° | tetap | tetap |
| flip horizontal | 180° − θ | −sin | tetap |
| flip vertikal | −θ | −sin | tetap |
| transpose | 90° − θ | −sin | tetap |

### 3.6 Label biner (B1–B3)
`y = (L > 0)`, tanpa kelas kontak. B3 memakai bobot `w` dari §3.4.

## 4. Model

- B1: `smp.Unet("efficientnet-b4", encoder_weights="imagenet", classes=1)`.
- B2, B3: `smp.UnetPlusPlus(... classes=1)`.
- B4, B5: `smp.UnetPlusPlus(... classes=3)`.
- FULL-P*: `smp.UnetPlusPlus(... classes=5)`; `logits = out[:, :3]`, `orient = tanh(out[:, 3:5])` → `(ŝ, ĉ)`.
- P1/P2: buat model 3 kanal pretrained, lalu `expand_first_conv(model, 4 atau 5)`.
- Semua model memakai encoder yang sama supaya perbandingan B1 → FULL hanya mengubah satu hal per langkah.
- Satu konvolusi head 5 kanal setara dengan dua head konvolusi terpisah (3 + 2 kanal) yang berbagi decoder. Cara ini
  dipilih karena tidak bergantung pada API decoder smp yang berubah antar-versi.

## 5. Loss (dihitung dalam float32)

- **Semua model:** piksel padding (label 255) tidak boleh menyumbang loss. `w = 0` di piksel 255; rata-rata loss
  piksel dihitung hanya atas piksel non-255.
- **Biner (B1–B3):** `L = Σ_{non-255} w · BCE(logit, y) / jumlah piksel non-255 + DiceLoss(mode="binary",
  ignore_index=255)`, dengan `w = 1` (selain 255) untuk B1–B2. Target BCE dibentuk dari `y == 1`, karena nilai 255
  tidak valid untuk BCE.
- **3 kelas (B4):** `L = mean_{piksel non-255}(w · CE(logits, cls)) + Dice(mode="multiclass", classes=[1, 2],
  ignore_index=255)`. Bobot kelas kontak default 1. Naikkan ke 2–5 **hanya** bila F1 kontak di `val_tune` rendah;
  putuskan sekali di Fase 5, lalu berlaku untuk semua model 3 kelas.
- **FULL:** `L_B4 + λ · L_θ`, dengan `L_θ = Σ_{p∈V} [(ŝ − S)² + (ĉ − C)²] / max(|V|, 1)`, default `λ = 1`.
- Dice dari `smp.losses.DiceLoss`; CE dari `F.cross_entropy(..., ignore_index=255, reduction="none")`.

## 6. Inferensi dan pemisahan instance

### 6.1 Ekstraksi kontur (dipakai semua mode)

```python
def mask_to_exterior(mask: np.ndarray) -> np.ndarray:
    """Exterior ring of a binary instance mask in continuous (x, y) coordinates."""
    padded = np.pad(mask.astype(np.uint8), 1)
    contours = skimage.measure.find_contours(padded, 0.5)
    ring = max(contours, key=lambda c: abs(shoelace_area(c)))   # (row, col), index space
    xy = ring[:, ::-1] - 1 + 0.5                                  # unpad, then pixel-center -> edge coords
    return xy
```

Batas marching squares di level 0,5 jatuh di tengah antara pusat piksel tepi dan pusat piksel latar, sehingga di
koordinat kontinu tepat berada di tepi piksel. Hasilnya, rasterisasi ulang dengan pycocotools kembali ke mask
semula. `cv2.findContours` tidak dipakai untuk membuat poligon keluaran karena menelusuri pusat piksel tepi
(poligon menyusut). Lubang diabaikan, karena format poligon COCO tidak mendukung lubang.

Sudah dicek saat menyusun dokumen ini (pycocotools 2.0.11, scikit-image 0.26): pada 300 bentuk acak (persegi,
persegi miring, bentuk L), IoU bolak-balik cara di atas = **1,000**, sedangkan koordinat indeks `cv2.findContours`
apa adanya rata-rata **0,84** (minimum 0,69). Untuk rumah 10 × 10 px: 1,00 vs 0,81.

### 6.2 Model 3 kelas (B4, B5, B5′, FULL)
Semua pelabelan komponen di sisi prediksi memakai **konektivitas 4** (`skimage.measure.label(..., connectivity=1)`,
dan `watershed` dengan konektivitas bawaannya, 1).
1. Pad ke kelipatan 32 → softmax → crop kembali ke ukuran asli.
2. Foreground: `F = (1 − P_bg) > 0,5`.
3. Marker: `M = (P_b − P_k > τ) & F`. Pemotongan ke `F` wajib: tanpa itu, satu komponen marker bisa menjembatani dua
   komponen `F`, sehingga satu label menjadi instance dua bagian. Buang komponen marker < `m_min` px (default 8), lalu
   beri label.
4. `inst = watershed(−P_b, markers=M_label, mask=F)` (`skimage.segmentation.watershed`).
5. Komponen `F` yang tidak terjangkau marker (`F & (inst == 0)`) → diberi label sebagai instance baru.
6. Setiap instance harus terhubung-4. Bila tidak, pecah per komponen. Lalu buang setiap instance < `A_min` px
   (default 10), termasuk instance yang tidak dipecah. `A_min` adalah luas minimum instance di semua model, sesuai
   grid §10 dan `docs/EXPERIMENTS.md` §4, yang menyebut `A_min` sebagai satu-satunya parameter model biner.
   (Diperjelas 30 Okt 2026; kalimat lama hanya menyebut bagian hasil pemecahan.)
7. Skor instance = rata-rata `1 − P_bg` di piksel instance.

### 6.3 Model biner (B1–B3)
Instance = komponen terhubung-4 dari `σ(logit) > 0,5`, tanpa instance < `A_min` px; skor = rata-rata probabilitas di
instance.

### 6.4 Orientasi instance dari head (FULL)
Normalisasi dulu vektor tiap piksel: `u_p = (ŝ, ĉ)_p / ‖(ŝ, ĉ)_p‖`, karena keluaran tanh dua kanal bukan vektor
satuan. Lalu rata-rata sirkular berbobot di piksel instance `I`, dengan bobot `ω = 1 − P_bg`:
`θ_i = ¼ · atan2(Σ ω u_s, Σ ω u_c) mod 90°`, dan konsistensi `κ_i = ‖Σ ω u‖ / Σ ω ∈ [0, 1]`.
Simpan `theta_deg` dan `kappa` di setiap objek hasil COCO (`docs/EVALUATION.md` §1). Untuk mode `mar`, simpan θ dari
minAreaRect dengan `kappa = null`. FULL **selalu** memakai θ dari head (tanpa fallback ke minAreaRect), supaya
perbandingan B5′ vs FULL-P0 bersih.

## 7. Poligonisasi dan regularisasi

| Mode | Dipakai oleh | Sumber θ | Isi |
|---|---|---|---|
| `raw` | B1–B4 | — | kontur §6.1 → DP dengan ε tetap 1,0 px |
| `mar` | B0, B5, B5′ | minAreaRect instance (§3.3) | regularisasi di bawah |
| `theta` | FULL, FULL+Y | head (§6.4) | regularisasi di bawah |

Regularisasi untuk instance `i` dengan mask `M_i`, kontur `P` (§6.1), luas `A`, dan orientasi `θ_i`:

1. **Jalur kotak (momen orde-2).** Ambil koordinat pusat piksel `M_i`, putar −θ_i terhadap centroid. Ukuran kotak:
   `w = √(12·var(x′) + 1)`, `h = √(12·var(y′) + 1)` (+1 mengoreksi lebar satu piksel), berpusat di centroid, lalu
   diputar balik. Jika `IoU(kotak, P) ≥ ρ` (default 0,90), output = kotak 4 verteks.
   **Jangan memakai bounding box**: satu tonjolan derau 1 px langsung memperbesar kotak. Pada prototipe dengan tepi
   berderau, kotak bounding box hanya mencapai IoU ±0,89 terhadap bangunan asli, sedangkan kotak momen 0,98–0,99
   (kontur mentah 0,95–0,97).
2. **Jalur DP + pelurusan** (selain itu), dikerjakan di kerangka yang diputar −θ_i:
   1. DP dengan `ε = k·√A` (default `k = 0,04`), boleh `shapely.simplify(ε, preserve_topology=True)`.
   2. Klasifikasi sisi: H bila ≤ δ dari 0°/180°, V bila ≤ δ dari 90° (default `δ = 15°`). Sisi lain yang lebih pendek
      dari `ℓ_O = max(6 px, 4ε)` dimasukkan ke H atau V terdekat (anak tangga rasterisasi atau sudut terpotong).
      Sisanya O (dinding miring sungguhan).
   3. Gabungkan sisi berurutan berkelas sama (H–H, V–V) menjadi satu garis dengan offset rata-rata berbobot panjang:
      H → garis `y = ȳ`, V → `x = x̄`, O → garis melalui kedua ujungnya.
   4. Selama jumlah sisi > 4: hapus sisi terpendek bila panjangnya < `ℓ_min = max(2 px, 2ε)`, lalu ulangi langkah 3.
   5. Verteks baru = perpotongan garis berurutan. Bila dua garis sejajar, atau titik potong lebih jauh dari
      `2ε + 2` px dari titik awal sisinya, pakai titik awal itu.
   6. Putar balik.
   Implementasi acuan: `reference/prototypes/regularize_proto.py`. Pada bentuk tanpa derau (8 sudut rotasi) hasilnya
   tepat: kotak 4, L 6, T 8, satu dinding miring 5 verteks. Dengan derau tepi berat, jumlah verteks L/T kadang lebih
   2–4, tetapi IoU rata-rata tetap lebih tinggi daripada kontur mentah.
3. **Validitas dan fallback.** Hasil diterima bila poligon valid (`is_valid`), luas > 0, dan
   `IoU(hasil, kontur §6.1) ≥ 0,7`. Jika gagal: coba hasil DP saja, lalu kontur mentah (`make_valid`, ambil
   poligon terbesar). Catat jumlah fallback per run.
4. **Klip** ke `box(0, 0, W, H)` (kotak yang diputar bisa keluar tile), lalu koordinat dibulatkan 2 desimal. Hasilnya
   dicek ulang: harus valid, berluas > 0, dan punya ≥ 3 titik unik. Bila gagal (misalnya bagian tipis yang jadi
   memotong dirinya sendiri), poligon diganti kontur §6.1 dan dihitung sebagai fallback. Koordinat kontur kelipatan
   0,5, jadi tidak berubah oleh pembulatan; hasil DP tidak punya jaminan itu. (Diperjelas 30 Okt 2026.)
5. **Format.** Hanya cincin luar, minimal 3 titik unik, konvensi titik penutup mengikuti GT (cek di Fase 1).

## 8. Baseline klasik B0 dan varian FULL+Y

**B0** (tanpa deep learning): `b` → MBI (§2.2, nilai sudah diskalakan ke [0, 1] seperti §2.4) → ambang
`max(Otsu per tile, t_min)`; `t_min` disetel di `val_tune_sub`, karena Otsu selalu membelah dua kelas, termasuk di
tile tanpa bangunan → opening (disk 2) → buang objek < 30 px → isi lubang → distance transform → marker h-maxima
(h = 2) → `watershed(−dist, marker, mask)` → `polygonize(mar)` → skor 1.

Rincian B0 (diperjelas 30 Okt 2026):
- **Marker h-maxima** adalah maksimum regional dari transformasi h-maxima, yaitu rekonstruksi dilasi `dist − h` di
  bawah `dist` (*extended maxima*, Soille). Hasilnya satu marker per kubah jarak yang lebih tinggi dari h.
  Keluaran `skimage.morphology.h_maxima` tidak dipakai, karena fungsi itu hanya menyisakan piksel paling atas tiap
  kubah. Di sepanjang punggung jarak rumah memanjang, jarak naik-turun sepersekian piksel, sehingga punggung itu pecah
  menjadi beberapa marker. Pada uji sintetis, rumah 40 × 12 px yang miring 80° mendapat 5 marker.
- Foreground, objek, marker, dan instance terhubung-4, seperti sisi model. Lubang adalah komponen latar terhubung-8
  (dual dari foreground terhubung-4) yang tidak menyentuh tepi tile.
- Komponen foreground tanpa marker tetap menjadi instance sendiri, seperti §6.2 langkah 5.

**FULL+Y** (opsional): YOLOv8n-OBB memberi kotak miring dengan confidence ≥ `c_y` (disetel di `val_tune_sub`).
Penggabungan dilakukan **setelah** langkah 3 §6.2 (marker sudah dipotong ke `F` dan disaring `m_min`), supaya seed YOLO
(disk radius 2 = 13 px) tidak terbuang oleh filter `m_min` yang grid-nya sampai 16 px.
1. Pusat kotak di dalam `F` → seed disk radius 2 px.
2. Komponen marker yang memuat ≥ 2 seed YOLO → diganti oleh seed-seed itu (rumah yang menyatu dipecah).
3. Komponen marker dengan 0 atau 1 seed → dipertahankan.
4. Seed YOLO di komponen `F` tanpa marker → ditambahkan.
5. Lanjut ke langkah 4 §6.2 (watershed). Hanya langkah marker yang berbeda dari FULL.

Label OBB dibuat dari peta instance: `cv2.boxPoints(cv2.minAreaRect(...))` → 4 titik dinormalisasi →
`0 x1 y1 x2 y2 x3 y3 x4 y4` per baris. Tetap tanpa anotasi poligon.

## 9. Ringkasan alasan desain

- **Kontak ⊂ bangunan:** foreground tetap sama dengan gabungan bangunan, sehingga celah tidak ikut ke poligon.
- **θ dari vektor sisi + 4θ:** tidak bergantung konvensi sudut OpenCV dan kontinu melewati 0°/90°.
- **Label diturunkan setelah augmentasi:** menghapus seluruh kelas bug tanda θ.
- **Marching squares:** poligon tidak menyusut; penting untuk rumah kecil di CrowdAI.
- **Kotak dari momen orde-2:** tahan terhadap tonjolan derau di tepi, berbeda dengan bounding box.
- **Satu head 5 kanal:** setara dua head, kebal terhadap perubahan API decoder smp.

## 10. Parameter

Nilai berpiksel di bawah untuk CrowdAI (GSD ±0,3 m). Untuk WHU 0,2 m, kalikan 1,5 lalu bulatkan (ke bilangan ganjil
untuk panjang SE): r = 3, σ = 4,5, A_θ = 45, s = 15–121 (15·k dibulatkan ke ganjil), L_g = 13; parameter luas (m_min, A_min, beserta
grid-nya) dikali 2,25.

| Parameter | Simbol | Default (CrowdAI) | Grid tuning | Disetel di | Catatan |
|---|---|---|---|---|---|
| Radius kontak | r | 2 px | — | tetap | ubah hanya lewat diskusi |
| Luas min. untuk θ | A_θ | 20 px | — | tetap | |
| Bobot tepi | α, σ | 5, 3 px | — | tetap | |
| Bobot loss θ | λ | 1,0 | {0,5; 1; 2} hanya bila L_θ mendominasi | awal Fase 6, sebelum FULL seed 0, sekali | dari run pendek; sama untuk semua seed |
| Skala MBI | s | 11–81 px, Δs 10 | visual | Fase 2, sekali | dibekukan 29 Okt 2026 (WHU 15–121, ganjil) |
| Panjang SE celah | L_g | 9 px | {7, 9, 11} visual + AUC `val_tune` | Fase 2, sekali | dibekukan 29 Okt 2026: 9 (WHU 13) |
| Mode celah | — | `contrast` | {`max`, `contrast`} visual + AUC `val_tune` | Fase 2, sekali | dibekukan 29 Okt 2026: `contrast` |
| Batas bawah ambang B0 | t_min | 0,15 | {0,05; 0,10; 0,15; 0,20} | `val_tune_sub` | B0 saja; skala MBI [0, 1] |
| Ambang marker | τ | 0,4 | {0,3; 0,4; 0,5; 0,6} | `val_tune_sub`, per model | |
| Luas min. marker | m_min | 8 px | {4, 8, 16} | `val_tune_sub`, per model | |
| Luas min. instance | A_min | 10 px | {5, 10, 20} | `val_tune_sub`, per model | |
| Ambang persegi (IoU kotak momen vs kontur) | ρ | 0,90 | {0,85; 0,90; 0,93} | `val_tune_sub`, per model | demo ETS memakai rasio luas 0,75 |
| Panjang sisi miring minimum | ℓ_O | max(6 px, 4ε) | — | tetap | lebih pendek → dianggap H/V |
| Panjang sisi minimum | ℓ_min | max(2 px, 2ε) | — | tetap | |
| Faktor ε DP | k | 0,04 | {0,02; 0,03; 0,04; 0,06} | `val_tune_sub`, per model | |
| Toleransi pelurusan | δ | 15° | {10, 15, 20} | `val_tune_sub`, per model | |
| Confidence YOLO | c_y | 0,25 | {0,15; 0,25; 0,4} | `val_tune_sub` | FULL+Y saja |

Protokol pemilihan grid ada di `docs/EXPERIMENTS.md` §4. Grid sama untuk semua model yang dibandingkan.

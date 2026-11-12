/* Presentation of the project, in the ITS Teknik Informatika lecture style (its_kit.js).
 *
 *   python scripts/paper_tables.py            # writes slides/numbers.json from the result tables
 *   NODE_PATH=<node_modules with pptxgenjs> node slides/build.js
 *
 * No result number is typed here: n("...") reads slides/numbers.json, the same numbers the paper quotes, and stops the
 * build when one is missing. Figures are the Indonesian-labelled ones of scripts/figures.py (--lang id), read from
 * $DENSEPOLY_OUT/figures/results/. Text is Bahasa Indonesia; English technical terms are in italics.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const { ITSDeck } = require("./its_kit");

const HERE = __dirname;
const numbers = JSON.parse(fs.readFileSync(path.join(HERE, "numbers.json"), "utf8"));
const out = process.env.DENSEPOLY_OUT || path.join(HERE, "..", "outputs");
const figure = (name) => {
  const file = path.join(out, "figures", "results", name);
  if (!fs.existsSync(file)) throw new Error(`${file} does not exist: run scripts/figures.py --lang id first`);
  return file;
};

/** A number of slides/numbers.json the Indonesian way: decimal comma, thousands point, true minus sign. */
function n(name) {
  if (!(name in numbers)) throw new Error(`number ${name} is not in slides/numbers.json: run scripts/paper_tables.py`);
  return numbers[name].replace(/, /g, "; ").replace(/[.,]/g, (c) => (c === "." ? "," : ".")).replace(/-/g, "−");
}

(async () => {
  const d = new ITSDeck({
    assets: path.join(HERE, "its_assets"),
    title: "Ekstraksi Poligon Bangunan pada Permukiman Padat",
    author: "Mahathir Muhammad",
  });

  d.cover({
    title: ["Ekstraksi Poligon Bangunan", "pada Permukiman Padat"],
    pill: "Final Project – Pengolahan Citra Digital",
    name: "Mahathir Muhammad – 6025252008",
  });

  let s = d.slide("Mengapa poligon, dan mengapa permukiman padat?", "Peta butuh satu poligon rapi per rumah, bukan *mask* piksel.");
  d.cards(s, [
    { head: "Poligon, bukan *mask*", body: "Peta dan SIG memakai poligon dengan sedikit verteks dan dinding lurus. *Mask* hasil segmentasi bergerigi dan boros verteks." },
    { head: "Atap berdempetan", body: "Di permukiman padat, beberapa rumah mudah terbaca sebagai satu gumpalan. Satu poligon gabungan menyembunyikan banyak rumah." },
    { head: "Tanpa anotasi poligon", body: "Metode terbaru dilatih dengan anotasi poligon. Banyak dataset daerah hanya punya *mask* per bangunan." },
  ], { callout: "Seberapa baik poligon yang dihasilkan hanya dari *mask*, bila dievaluasi tanpa kebocoran data?" });

  s = d.slide("Tiga gagasan yang diuji, dengan aturan yang dibekukan", "Cara menilai tiap gagasan ditulis sebelum ada model yang dilatih.");
  d.numbered(s, [
    { head: "N1: *head* orientasi", body: "Sudut bangunan θ dipelajari dari *mask*, lalu memandu perapian poligon." },
    { head: "N2: evaluasi per kepadatan", body: "CrowdAI bebas duplikat, dibagi menurut jumlah bangunan per *tile*." },
    { head: "N3: prior morfologi", body: "MBI dan peta celah ditambahkan sebagai kanal input." },
    { head: "Aturan keputusan", body: "Tiga *seed*, selang kepercayaan *bootstrap*, hasil negatif tetap dilaporkan." },
  ], { labelW: 3.6 });

  s = d.slide("Alur metode: dari citra ke poligon", "Jaringan segmentasi, pemisahan *instance*, lalu perapian geometris.");
  d.pipeline(s, [
    { head: "Citra RGB", body: "*Tile* 300 × 300 piksel, tanpa diubah ukurannya." },
    { head: "U-Net++", body: "*Encoder* EfficientNet-B4, satu *head* dengan lima keluaran." },
    { head: "Tiga kelas + θ", body: "Latar, bangunan, kontak, dan orientasi (sin 4θ, cos 4θ)." },
    { head: "Marker + *watershed*", body: "Memisahkan gumpalan menjadi satu *instance* per rumah." },
    { head: "Regularisasi", body: "Kontur dirapikan searah θ menjadi kotak atau poligon siku." },
    { head: "Poligon", body: "Satu poligon per rumah, format COCO JSON dan GeoJSON." },
  ], { highlight: 5, callout: "Semua label, termasuk θ, diturunkan dari *mask*. Tidak ada anotasi poligon saat *training*." });

  s = d.slide("Label diturunkan otomatis dari *mask*", "Tiga jenis label dibuat dari peta *instance* yang sama.");
  d.cards(s, [
    { head: "Kelas kontak", body: "Piksel bangunan yang berjarak ≤ 2 piksel dari bangunan lain. Tujuannya menandai tempat dua atap bertemu." },
    { head: "Orientasi θ", body: "Arah sisi kotak terkecil (*minAreaRect*) tiap bangunan, dikodekan sebagai (sin 4θ, cos 4θ) agar keempat sisi setara." },
    { head: "Bobot tepi", body: "Piksel dekat batas bangunan diberi bobot *loss* lebih besar, supaya celah sempit antar-rumah tidak diabaikan." },
  ]);

  s = d.slide("Evaluasi tanpa kebocoran data", "Val CrowdAI tumpang-tindih dengan train, jadi *split* dibuat ulang.");
  d.numbered(s, [
    { head: "Deduplikasi", body: `${n("NTrainImages")} *tile* train menjadi ${n("NTrainClusters")} klaster; satu wakil per klaster.` },
    { head: "*Split* bersih", body: `Train ${n("NTrainDedup")}, tuning ${n("NTuneTiles")} + ${n("NSelectTiles")}, *test* ${n("NTestTiles")} *tile*.` },
    { head: "Strata kepadatan", body: "Jarang, sedang, padat; dibekukan dari *ground truth* sebelum evaluasi." },
    { head: "Statistik", body: "Tiga *seed*, selisih berpasangan, *bootstrap* 1.000 replikasi per *tile*." },
    { head: "*Test* tidak disetel", body: "Semua parameter disetel di data tuning, lalu dibekukan." },
  ], { labelW: 2.9 });

  s = d.slide("Hasil utama di CrowdAI *test*", "Rata-rata tiga *seed*. MTA dan *merge rate*: makin kecil makin baik.");
  const row = (label, code) => [label, n(`NAp${code}`), n(`NCiou${code}`), n(`NMta${code}`), n(`NMerge${code}`)];
  d.table(s, ["Model", "AP", "C-IoU", "MTA (°)", "*Merge rate* (%)"], [
    row("B3: segmentasi biner", "Bthree"),
    row("B3m: B3 + regularisasi (eksploratif)", "Bthreem"),
    row("B4: tiga kelas + *watershed*", "Bfour"),
    row("B5: B4 + regularisasi", "Bfive"),
    row("FULL-P0: + *head* θ", "Full"),
    row("FULL+Y: + *seed* detektor", "FullY"),
    row("B3m+Y (eksploratif)", "BthreemY"),
  ], { colW: [5.3, 1.5, 1.6, 1.6, 1.894], align: "center",
    callout: "Regularisasi yang merapikan poligon. Jalur biner unggul di AP; jalur tiga kelas unggul memisahkan rumah." });

  s = d.slide("N1 didukung: *head* θ merapikan poligon", `B5 → FULL-P0 di ${n("NTilesDense")} *tile* padat, selisih berpasangan dengan selang 95%.`);
  d.cards(s, [
    { head: "PoLiS", body: `Turun ${n("NNoneTotalPolis")} piksel, selang 95% ${n("NNoneTotalPolisCi")}. Batas poligon lebih dekat ke *ground truth*.` },
    { head: "MTA", body: `Galat sudut turun ${n("NNoneTotalMta")}°, selang 95% ${n("NNoneTotalMtaCi")}. Dinding lebih sejajar.` },
    { head: "AP", body: `Berubah ${n("NNoneTotalAp")} poin, jadi kerapian tidak dibayar dengan akurasi deteksi.` },
  ], { callout: "Efeknya kecil, tetapi searah di ketiga *seed*." });

  s = d.slide("Efek N1 membesar di *tile* padat", "Selisih B5 → FULL-P0 dan FULL-P0 → FULL-P2 per strata, dengan selang 95%.");
  d.mediaStack(s, { image: figure("crowdai_n2_id.png"), alt: "Selisih per strata kepadatan",
    callout: "Selisih padat − jarang signifikan untuk MTA, tidak untuk PoLiS: klaim \"terbesar di area padat\" tidak dipakai." });

  s = d.slide("Dua hasil negatif, dilaporkan apa adanya", "Keduanya dirancang untuk memisahkan rumah berdempetan.");
  d.cards(s, [
    { head: "N3: prior morfologi tidak membantu", body: `Dengan MBI dan peta celah, *merge rate* berubah ${n("NNthreeDeltaMerge")} poin, selang 95% ${n("NNthreeDeltaMergeCi")}. Arahnya salah dan tidak sama di ketiga *seed*.` },
    { head: "Kelas kontak jarang aktif", body: `Di ${n("NContactInactive")} dari ${n("NContactRuns")} *run* FULL, kelas kontak tidak pernah terpilih. Rumah terpisah oleh celah latar, bukan oleh kelas ini.` },
  ]);

  s = d.slide("Banding dengan HiSup di data tanpa kebocoran", `${n("NTilesClean")} *tile* val yang tidak punya kembaran di train; kode evaluasi yang sama.`);
  d.table(s, ["Metode", "Anotasi poligon", "AP", "AP50", "AP *boundary*", "C-IoU"], [
    ["HiSup, val penuh (bocor)", "ya", n("NLeakyApHisup"), "–", "–", "–"],
    ["HiSup, *tile* bersih", "ya", n("NCleanApHisup"), n("NCleanApfiftyHisup"), n("NCleanApbHisup"), n("NCleanCiouHisup")],
    ["FULL-P0 (kita), *tile* bersih", "tidak", n("NCleanApFull"), n("NCleanApfiftyFull"), n("NCleanApbFull"), n("NCleanCiouFull")],
  ], { colW: [4.6, 2.1, 1.2, 1.2, 1.6, 1.194], align: "center",
    callout: `HiSup turun dari ${n("NLeakyApHisup")} ke ${n("NCleanApHisup")} AP di data bersih; kita tertinggal ${n("NCleanDeltaAp")} poin.` });

  s = d.slide("Eksploratif: *baseline* sederhana dan *seed* detektor", "Ditambahkan setelah hasil ada, jadi bukan bagian dari klaim.");
  d.cards(s, [
    { head: "B3m: biner + regularisasi", body: `AP ${n("NApBthreem")} dan C-IoU ${n("NCiouBthreem")}; FULL-P0 ${n("NApFull")} dan ${n("NCiouFull")}. B3m unggul di semua metrik kecuali rumah tergabung.` },
    { head: "*Seed* dari detektor", body: `YOLOv8n-OBB, juga dilatih dari *mask*, mengubah *merge rate* FULL-P0 ${n("NSeedFullMerge")} poin dan B3m ${n("NSeedBinaryMerge")} poin, dengan AP setara.` },
    { head: "Artinya", body: "Jalur biner paling akurat dan rapi. Jalur tiga kelas paling baik memisahkan rumah berdempetan. *Seed* membantu keduanya." },
  ]);

  s = d.slide("Contoh hasil di *tile* asli", "*Ground truth* dan poligon tiap model pada *tile* *test* yang sama.");
  d.mediaStack(s, { image: figure("crowdai_qualitative_id.png"), alt: "Poligon tiap model pada tile test", split: 0.6, items: [
    { head: "B4", body: "Kontur mentah: bergelombang dan boros verteks." },
    { head: "FULL-P0, FULL+Y, B3m", body: "Setelah regularisasi: poligon siku dengan sedikit verteks." },
    { head: "Baris terbawah", body: "Contoh kegagalan: rumah berdempetan masih tergabung." },
  ] });

  s = d.slide("WHU dan biaya komputasi", "Dataset kedua (satu *seed*) dan kecepatan alur penuh.");
  d.cards(s, [
    { head: "WHU", body: `AP ${n("NWhuAp")} dan C-IoU ${n("NWhuCiouMain")}. Dengan aturan Pix2Poly C-IoU ${n("NWhuCiou")}; dilaporkan di paper: HiSup ${n("NWhuCiouHisup")}, Pix2Poly ${n("NWhuCiouPixtopoly")} (kira-kira).` },
    { head: "Titik lemah", body: `${n("NWhuMergeTouching")}% rumah tergabung di *patch* yang berisi bangunan bersentuhan.` },
    { head: "Biaya", body: `${n("NParamsFull")} juta parameter. ${n("NFpsFull")} *tile* per detik di RTX 5090, dan ${n("NFpsFullY")} dengan detektor.` },
  ]);

  s = d.slide("Keterbatasan", "Hal yang membatasi kesimpulan penelitian ini.");
  d.numbered(s, [
    { head: "Data bersih kecil", body: `Banding HiSup hanya di ${n("NTilesClean")} *tile*, dan bukan sampel acak.` },
    { head: "WHU satu *seed*", body: "Daftar *tile* uji tidak persis sama dengan Pix2Poly." },
    { head: "Efek N1 kecil", body: "Baru diuji di jalur tiga kelas, belum di jalur biner." },
    { head: "Satu detektor", body: "YOLO dilatih sekali dan dipakai untuk ketiga *seed*." },
    { head: "Tanpa lubang", body: "Poligon hanya memuat cincin luar bangunan." },
  ], { labelW: 2.9 });

  s = d.slide("Kesimpulan", "Empat hal yang bisa dibawa pulang.");
  d.numbered(s, [
    { head: "Poligon rapi bisa dari *mask* saja", body: "Regularisasi geometris bekerja tanpa anotasi poligon." },
    { head: "*Head* θ membantu sedikit", body: "Konsisten, dan makin terasa di *tile* padat." },
    { head: "Prior dan kelas kontak gagal", body: "Dilaporkan sebagai hasil negatif." },
    { head: "*Baseline* sederhana paling kuat", body: "Terlihat karena semua model dinilai dengan protokol yang sama." },
  ], { shape: "square", labelW: 4.7 });

  s = d.slide("Referensi");
  d.references(s, [
    "Girard, N., dkk. (2021). Polygonal building extraction by frame field learning. *CVPR*.",
    "Xu, B., dkk. (2023). HiSup: Accurate polygonal mapping of buildings in satellite imagery. *ISPRS JPRS*, 198, 284–296.",
    "Adimoolam, Y. K., dkk. (2025). Pix2Poly: A sequence prediction method for polygonal building footprint extraction. *WACV*.",
    "Adimoolam, Y. K., dkk. (2023). Data leakage detection and de-duplication in large scale geospatial image datasets. arXiv:2304.02296.",
    "Iglovikov, V. I., dkk. (2018). TernausNetV2: Fully convolutional network for instance segmentation. *CVPRW*.",
    "Huang, X., & Zhang, L. (2011). A multidirectional and multiscale morphological index for automatic building extraction. *PE&RS*, 77(7), 721–732.",
    "Zhou, Z., dkk. (2018). UNet++: A nested U-Net architecture for medical image segmentation. *DLMIA*.",
  ]);

  d.closing();
  const target = path.join(HERE, "densepoly.pptx");
  await d.save(target);
  console.log(target);
})().catch((error) => {
  console.error(String(error && error.message ? error.message : error));
  process.exit(1);
});

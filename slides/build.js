/* Presentation of the project, in the ITS Teknik Informatika lecture style (its_kit.js).
 *
 *   python scripts/paper_tables.py            # writes slides/numbers.json from the result tables
 *   NODE_PATH=<node_modules with pptxgenjs> node slides/build.js
 *
 * No result number is typed here: n("...") reads slides/numbers.json, the same numbers the paper quotes, and stops the
 * build when one is missing. Figures are the Indonesian-labelled ones of scripts/figures.py (--lang id), read from
 * $DENSEPOLY_OUT/figures/results/. Text is Bahasa Indonesia; English technical terms are in italics.
 *
 * say("...") after a slide holds the words spoken at it. The build writes them to slides/naskah.md, one section per
 * slide, and stops when a slide has none, so the script cannot fall behind the deck. Its numbers are n("...") too.
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

/** The words spoken at each slide, in the order of the deck: one say() per slide, one paragraph per argument. */
const script = [];
const say = (title, ...paragraphs) => script.push({ title, paragraphs });
const WORDS_PER_MINUTE = 125; // a calm pace for a talk in Bahasa Indonesia; only used for the time estimates

function writeScript(target, slides) {
  if (script.length !== slides) throw new Error(`${script.length} say() for ${slides} slides: every slide needs its words`);
  const words = (item) => item.paragraphs.join(" ").replace(/\*/g, "").split(/\s+/).filter(Boolean).length;
  const minutes = (count) => Math.max(0.5, Math.round((count / WORDS_PER_MINUTE) * 2) / 2).toString().replace(".", ",");
  const total = script.reduce((sum, item) => sum + words(item), 0);
  const lines = [
    "# Naskah presentasi: Ekstraksi Poligon Bangunan pada Permukiman Padat",
    "",
    `Naskah ucapan untuk tiap slide \`densepoly.pptx\` (${slides} slide), untuk dosen dan rekan mahasiswa. Lama bicara kira-kira`,
    `${Math.round(total / WORDS_PER_MINUTE)} menit pada ${WORDS_PER_MINUTE} kata per menit; perkiraan per slide ada di tiap judul.`,
    "",
    "- Naskah ini dibuat `slides/build.js` bersama deck, jadi angkanya sama dengan yang tampil di slide. Untuk mengubahnya,",
    "  ubah `say(...)` di `build.js` lalu bangun ulang; jangan menyunting file ini langsung.",
    "- Istilah Inggris dicetak miring. Angka di dalam kurung siku adalah selang kepercayaan 95%.",
    "- Bila waktu hanya ±10 menit: lewati slide \"Eksploratif\" dan \"WHU dan biaya komputasi\", dan sebut isinya dalam satu",
    "  kalimat di slide Kesimpulan.",
    "",
  ];
  script.forEach((item, i) => {
    lines.push(`## Slide ${i + 1}: ${item.title}`, "", `*±${minutes(words(item))} menit*`, "");
    for (const paragraph of item.paragraphs) lines.push(paragraph, "");
  });
  fs.writeFileSync(target, lines.join("\n").replace(/\n+$/, "\n"));
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
  say("Sampul",
    "Selamat pagi, Bapak/Ibu dosen dan teman-teman. Saya Mahathir Muhammad, NRP 6025252008. Saya akan mempresentasikan *final project* mata kuliah Pengolahan Citra Digital, berjudul Ekstraksi Poligon Bangunan pada Permukiman Padat.",
    "Singkatnya: dari foto udara, saya ingin komputer menggambar denah atap setiap rumah sebagai poligon yang rapi, termasuk di kawasan yang rumahnya saling berdempetan.");

  let s = d.slide("Inti presentasi dalam tiga pertanyaan", "Masalah, temuan, dan artinya; slide berikutnya menguraikan satu per satu.");
  d.cards(s, [
    { head: "Masalahnya apa?", body: "Peta butuh satu poligon rapi per rumah. Di permukiman padat, atap berdempetan terbaca sebagai satu gumpalan. Metode terbaik butuh anotasi poligon, dan data uji CrowdAI bocor ke data latih." },
    { head: "Ketemunya apa?", body: `Dari *mask* saja, regularisasi menaikkan kerapian poligon (C-IoU) dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}. *Head* θ menambah sedikit kerapian. Prior morfologi dan kelas kontak tidak membantu. Di data bersih, AP HiSup ternyata ${n("NCleanApHisup")}, bukan ${n("NLeakyApHisup")} seperti di data bocor.` },
    { head: "Kenapa itu penting?", body: "Banyak daerah hanya punya *mask*: poligon rapi tetap bisa dibuat tanpa anotasi poligon. Angka *benchmark* yang bocor terlalu optimistis, jadi metode harus dibandingkan di data bersih." },
  ]);
  say("Inti presentasi dalam tiga pertanyaan",
    "Sebelum masuk ke rincian, saya rangkum dulu seluruh presentasi ini dalam tiga pertanyaan.",
    "Pertama, masalahnya apa? Peta membutuhkan satu poligon yang rapi untuk tiap rumah. Di permukiman padat itu sulit, karena atap yang berdempetan terbaca sebagai satu gumpalan. Selain itu, metode terbaik saat ini membutuhkan anotasi poligon, dan data uji CrowdAI yang biasa dipakai ternyata bocor ke data latih.",
    `Kedua, ketemunya apa? Dari *mask* saja, langkah regularisasi sudah membuat poligon jauh lebih rapi: C-IoU, ukuran kerapian poligon yang makin tinggi makin baik, naik dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}. *Head* orientasi menambah kerapian sedikit lagi. Dua gagasan lain, yaitu prior morfologi dan kelas kontak, tidak membantu.`,
    `Lalu soal metode terbaik, HiSup. AP-nya, yang juga makin tinggi makin baik, tercatat ${n("NLeakyApHisup")} di data uji yang bocor. Di data yang bersih ternyata hanya ${n("NCleanApHisup")}. Modelnya tidak berubah; yang berubah hanya soal ujiannya, yang kali ini belum pernah dilihat model. Jadi angka yang tinggi tadi sebagian berasal dari kebocoran.`,
    "Ketiga, kenapa itu penting? Banyak daerah hanya punya data *mask*, jadi poligon yang rapi tetap bisa dibuat tanpa anotasi poligon. Dan angka *benchmark* yang bocor ternyata terlalu optimistis, sehingga metode seharusnya dibandingkan di data yang bersih.",
    "Slide-slide berikutnya menguraikan ketiga jawaban itu satu per satu.");

  s = d.slide("Mengapa poligon, dan mengapa permukiman padat?", "Peta butuh satu poligon rapi per rumah, bukan *mask* piksel.");
  d.cards(s, [
    { head: "Poligon, bukan *mask*", body: "Peta dan SIG memakai poligon dengan sedikit verteks dan dinding lurus. *Mask* hasil segmentasi bergerigi dan boros verteks." },
    { head: "Atap berdempetan", body: "Di permukiman padat, beberapa rumah mudah terbaca sebagai satu gumpalan. Satu poligon gabungan menyembunyikan banyak rumah." },
    { head: "Tanpa anotasi poligon", body: "Metode terbaru dilatih dengan anotasi poligon. Banyak dataset daerah hanya punya *mask* per bangunan." },
  ], { callout: "Seberapa baik poligon yang dihasilkan hanya dari *mask*, bila dievaluasi tanpa kebocoran data?" });
  say("Mengapa poligon, dan mengapa permukiman padat?",
    "Saya mulai dari masalahnya, yang punya tiga sisi. Pertama, peta dan sistem informasi geografis menyimpan bangunan sebagai poligon, yaitu beberapa titik sudut yang dihubungkan dinding lurus. Keluaran segmentasi citra berbeda: bentuknya *mask*, kumpulan piksel yang tepinya bergerigi. Kalau *mask* langsung dijadikan poligon, titik sudutnya bisa ratusan dan dindingnya tidak lurus.",
    "Kedua, di permukiman padat atap rumah saling menempel, sehingga beberapa rumah sering terbaca sebagai satu gumpalan besar. Untuk pemetaan itu masalah, karena satu poligon gabungan menyembunyikan banyak rumah.",
    "Ketiga, metode terbaru seperti HiSup dan Pix2Poly dilatih dengan anotasi poligon. Padahal banyak dataset, terutama data daerah, hanya punya *mask* per bangunan.",
    "Jadi pertanyaan penelitian saya: seberapa baik poligon yang bisa dihasilkan kalau model hanya belajar dari *mask*, dan kalau hasilnya dinilai di data yang bebas kebocoran?");

  s = d.slide("Tiga gagasan yang diuji, dengan aturan yang dibekukan", "Cara menilai tiap gagasan ditulis sebelum ada model yang dilatih.");
  d.numbered(s, [
    { head: "N1: *head* orientasi", body: "Sudut bangunan θ dipelajari dari *mask*, lalu memandu perapian poligon." },
    { head: "N2: evaluasi per kepadatan", body: "CrowdAI bebas duplikat, dibagi menurut jumlah bangunan per *tile*." },
    { head: "N3: prior morfologi", body: "MBI dan peta celah ditambahkan sebagai kanal input." },
    { head: "Aturan keputusan", body: "Tiga *seed*, selang kepercayaan *bootstrap*, hasil negatif tetap dilaporkan." },
  ], { labelW: 3.6 });
  say("Tiga gagasan yang diuji, dengan aturan yang dibekukan",
    "Saya menguji tiga gagasan. N1 adalah gagasan metode: jaringan diberi satu keluaran tambahan, disebut *head* orientasi, yang menebak arah hadap bangunan, yaitu sudut θ. Label sudutnya dihitung otomatis dari *mask*, lalu sudut itu dipakai untuk merapikan poligon.",
    "N2 bukan metode, melainkan cara evaluasi: dataset CrowdAI dibersihkan dari duplikat, lalu hasil dilaporkan per tingkat kepadatan bangunan.",
    "N3 adalah gagasan *preprocessing*: dua peta dari morfologi klasik, yaitu MBI dan peta celah, ditambahkan sebagai kanal input, dengan harapan celah antar-rumah lebih mudah terlihat oleh jaringan.",
    "Yang ingin saya tekankan: kriteria berhasil atau gagal untuk tiap gagasan saya tulis dan saya bekukan sebelum ada model yang dilatih. Setiap perbandingan memakai tiga *seed* dan selang kepercayaan *bootstrap*. Akibatnya hasil negatif pun tetap dilaporkan, dan itu akan terlihat di slide-slide berikutnya.");

  s = d.slide("Alur metode: dari citra ke poligon", "Jaringan segmentasi, pemisahan *instance*, lalu perapian geometris.");
  d.pipeline(s, [
    { head: "Citra RGB", body: "*Tile* 300 × 300 piksel, tanpa diubah ukurannya." },
    { head: "U-Net++", body: "*Encoder* EfficientNet-B4, satu *head* dengan lima keluaran." },
    { head: "Tiga kelas + θ", body: "Latar, bangunan, kontak, dan orientasi (sin 4θ, cos 4θ)." },
    { head: "Marker + *watershed*", body: "Memisahkan gumpalan menjadi satu *instance* per rumah." },
    { head: "Regularisasi", body: "Kontur dirapikan searah θ menjadi kotak atau poligon siku." },
    { head: "Poligon", body: "Satu poligon per rumah, format COCO JSON dan GeoJSON." },
  ], { highlight: 5, callout: "Semua label, termasuk θ, diturunkan dari *mask*. Tidak ada anotasi poligon saat *training*." });
  say("Alur metode: dari citra ke poligon",
    "Ini alur lengkapnya, dibaca dari kiri ke kanan. Inputnya citra RGB, satu *tile* berukuran 300 × 300 piksel, tanpa diubah ukurannya.",
    "Jaringannya U-Net++ dengan *encoder* EfficientNet-B4. Keluarannya lima kanal. Tiga kanal pertama adalah kelas: latar, bangunan, dan kontak, yaitu tempat dua bangunan bersentuhan. Dua kanal terakhir adalah orientasi bangunan.",
    "Dari peta kelas dibuat *marker*, yaitu inti tiap rumah. Algoritma *watershed* lalu memekarkan *marker* itu, sehingga satu gumpalan terpecah menjadi satu *instance* per rumah.",
    "Kontur tiap rumah kemudian dirapikan searah sudut θ. Bangunan yang hampir persegi diganti dengan kotak. Bangunan lain disederhanakan, lalu dindingnya diluruskan mengikuti θ.",
    "Hasil akhirnya satu poligon per rumah dalam format COCO JSON dan GeoJSON. Semua label untuk melatih jaringan ini, termasuk θ, diturunkan dari *mask*.");

  s = d.slide("Label diturunkan otomatis dari *mask*", "Tiga jenis label dibuat dari peta *instance* yang sama.");
  d.cards(s, [
    { head: "Kelas kontak", body: "Piksel bangunan yang berjarak ≤ 2 piksel dari bangunan lain. Tujuannya menandai tempat dua atap bertemu." },
    { head: "Orientasi θ", body: "Arah sisi kotak terkecil (*minAreaRect*) tiap bangunan, dikodekan sebagai (sin 4θ, cos 4θ) agar keempat sisi setara." },
    { head: "Bobot tepi", body: "Piksel dekat batas bangunan diberi bobot *loss* lebih besar, supaya celah sempit antar-rumah tidak diabaikan." },
  ]);
  say("Label diturunkan otomatis dari *mask*",
    "Karena tidak ada anotasi poligon, semua label dibuat otomatis dari peta *instance*, yaitu *mask* yang tiap bangunannya punya nomor sendiri.",
    "Label pertama adalah kelas kontak: piksel bangunan yang jaraknya paling jauh dua piksel dari bangunan lain. Tujuannya menandai tempat dua atap bertemu.",
    "Label kedua adalah orientasi. Untuk tiap bangunan dicari kotak terkecil yang membungkusnya, dengan fungsi *minAreaRect* di OpenCV, lalu diambil arah sisinya. Sudut itu dikodekan sebagai sinus dan kosinus dari 4θ. Alasannya, bangunan persegi terlihat sama bila diputar 90 derajat, jadi keempat arah sisinya harus dianggap satu arah yang sama.",
    "Yang ketiga adalah bobot tepi: piksel di dekat batas bangunan diberi bobot *loss* lebih besar, supaya jaringan tidak mengabaikan celah sempit di antara rumah.",
    "Kelas kontak dan bobot tepi saya adopsi dari penelitian sebelumnya. Yang baru di sini adalah label orientasi yang diturunkan dari *mask*.");

  s = d.slide("Evaluasi tanpa kebocoran data", "Val CrowdAI tumpang-tindih dengan train, jadi *split* dibuat ulang.");
  d.numbered(s, [
    { head: "Deduplikasi", body: `${n("NTrainImages")} *tile* train menjadi ${n("NTrainClusters")} klaster; satu wakil per klaster.` },
    { head: "*Split* bersih", body: `Train ${n("NTrainDedup")}, tuning ${n("NTuneTiles")} + ${n("NSelectTiles")}, *test* ${n("NTestTiles")} *tile*.` },
    { head: "Strata kepadatan", body: "Jarang, sedang, padat; dibekukan dari *ground truth* sebelum evaluasi." },
    { head: "Statistik", body: "Tiga *seed*, selisih berpasangan, *bootstrap* 1.000 replikasi per *tile*." },
    { head: "*Test* tidak disetel", body: "Semua parameter disetel di data tuning, lalu dibekukan." },
  ], { labelW: 2.9 });
  say("Evaluasi tanpa kebocoran data",
    "Bagian ini menurut saya yang paling penting. Dataset CrowdAI punya masalah yang sudah dilaporkan peneliti lain: banyak citra validasi yang sama persis dengan citra train, atau hanya diputar dan dicerminkan. Model seolah diuji dengan soal yang sudah pernah dilihatnya, sehingga angkanya terlalu optimistis.",
    `Karena itu saya mendeduplikasi dengan *perceptual hash*. Dari ${n("NTrainImages")} *tile* train tersisa ${n("NTrainClusters")} klaster, dan dari tiap klaster diambil satu wakil.`,
    `Klaster itu lalu dibagi: ${n("NTrainDedup")} *tile* untuk train, ${n("NTuneTiles")} untuk menyetel *post-processing*, ${n("NSelectTiles")} untuk memilih model, dan ${n("NTestTiles")} *tile* *test* yang disisihkan.`,
    "*Tile* *test* dibagi menjadi tiga kelompok kepadatan, yaitu jarang, sedang, dan padat, menurut jumlah bangunannya. Pembagian ini dibekukan dari *ground truth* sebelum ada model yang dievaluasi.",
    "Setiap selisih antarmodel dihitung berpasangan pada *tile* yang sama, dengan tiga *seed* dan *bootstrap* seribu replikasi. Dan semua parameter disetel di data tuning; data *test* tidak pernah dipakai untuk menyetel apa pun.");

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
  say("Hasil utama di CrowdAI *test*",
    "Tabel ini merangkum hasil di data *test*, rata-rata tiga *seed*. Sebelum membacanya, sedikit tentang metriknya. AP mengukur ketepatan deteksi tiap bangunan; makin besar makin baik. C-IoU adalah IoU yang dikenai penalti bila jumlah titik sudut poligon jauh dari *ground truth*, jadi ia mengukur kerapian. MTA adalah galat sudut dinding dalam derajat. *Merge rate* adalah persentase rumah yang tergabung dengan tetangganya. Untuk MTA dan *merge rate*, makin kecil makin baik.",
    `Ada tiga hal yang bisa dibaca. Pertama, regularisasi sangat menaikkan kerapian: dari B4 ke B5, C-IoU naik dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}, dan MTA turun dari ${n("NMtaBfour")} ke ${n("NMtaBfive")} derajat.`,
    `Kedua, jalur biner unggul di AP: B3m ${n("NApBthreem")}, sedangkan FULL-P0 ${n("NApFull")}.`,
    `Ketiga, jalur tiga kelas lebih baik memisahkan rumah: *merge rate* FULL-P0 ${n("NMergeFull")}%, B3m ${n("NMergeBthreem")}%. Dengan bantuan detektor, FULL+Y turun lagi ke ${n("NMergeFullY")}%.`,
    "Jadi tidak ada satu model yang menang di semua kolom. Itu saya uraikan satu per satu.");

  s = d.slide("N1 didukung: *head* θ merapikan poligon", `B5 → FULL-P0 di ${n("NTilesDense")} *tile* padat, selisih berpasangan dengan selang 95%.`);
  d.cards(s, [
    { head: "PoLiS", body: `Turun ${n("NNoneTotalPolisAbs")} piksel; selisih dan selang 95%: ${n("NNoneTotalPolis")} ${n("NNoneTotalPolisCi")}. Batas poligon lebih dekat ke *ground truth*.` },
    { head: "MTA", body: `Galat sudut turun ${n("NNoneTotalMtaAbs")}°; selisih dan selang 95%: ${n("NNoneTotalMta")} ${n("NNoneTotalMtaCi")}. Dinding lebih sejajar.` },
    { head: "AP", body: `Berubah ${n("NNoneTotalAp")} poin, jadi kerapian tidak dibayar dengan akurasi deteksi.` },
  ], { callout: "Efeknya kecil, tetapi searah di ketiga *seed*." });
  say("N1 didukung: *head* θ merapikan poligon",
    `Sekarang gagasan pertama. Yang dibandingkan adalah B5 dan FULL-P0, di ${n("NTilesDense")} *tile* terpadat. Di B5, sudut perapian diambil dari kotak terkecil kontur hasil prediksi. Di FULL-P0, jaringan dilatih dengan *head* θ dan sudutnya dibaca dari *head* itu.`,
    `PoLiS adalah jarak rata-rata antara batas poligon prediksi dan batas *ground truth*. Nilainya turun ${n("NNoneTotalPolisAbs")} piksel. MTA, galat sudut dinding, turun ${n("NNoneTotalMtaAbs")} derajat. Selang kepercayaan 95% keduanya tidak memuat nol, dan arahnya sama di ketiga *seed*.`,
    `AP berubah ${n("NNoneTotalAp")} poin, jadi kerapian itu tidak dibayar dengan akurasi deteksi.`,
    "Menurut aturan yang sudah dibekukan, N1 dinyatakan didukung. Tetapi saya harus jujur soal besarnya: efeknya kecil, jauh di bawah satu piksel. Jadi kesimpulannya, *head* θ membantu secara konsisten, bukan membantu secara besar.");

  s = d.slide("Efek N1 membesar di *tile* padat", "Selisih B5 → FULL-P0 dan FULL-P0 → FULL-P2 per strata, dengan selang 95%.");
  d.mediaStack(s, { image: figure("crowdai_n2_id.png"), alt: "Selisih per strata kepadatan",
    callout: "Selisih padat − jarang signifikan untuk MTA, tidak untuk PoLiS: klaim \"terbesar di area padat\" tidak dipakai." });
  say("Efek N1 membesar di *tile* padat",
    "Gambar ini memecah efek tadi menurut kepadatan. Tiap titik adalah selisih antara dua model, dan garis tegaknya selang kepercayaan 95%. Titik di bawah garis putus-putus berarti ada perbaikan.",
    `Di dua panel kiri, untuk *head* θ, perbaikannya membesar dari *tile* jarang ke padat. PoLiS: ${n("NDensePolisSparse")}, ${n("NDensePolisMedium")}, lalu ${n("NDensePolisDense")} piksel. MTA: ${n("NDenseMtaSparse")}, ${n("NDenseMtaMedium")}, lalu ${n("NDenseMtaDense")} derajat.`,
    `Dua titik di sebelah kanan tiap panel membagi *tile* dengan cara lain: yang berisi bangunan bersentuhan dan yang tidak. Selang kelompok bersentuhan lebar karena *tile*-nya hanya ${n("NTilesTouching")} dari ${n("NTestTiles")}.`,
    `Kembali ke kepadatan: kecenderungan belum tentu bukti. Ketika selisih padat dikurangi jarang diuji, hanya MTA yang signifikan, yaitu ${n("NDenseMta")} derajat. Untuk PoLiS selang kepercayaannya memuat nol. Karena itu klaim bahwa efek N1 paling besar di area padat tidak saya pakai.`,
    "Panel kanan adalah prior morfologi, gagasan N3. Di sana titik-titiknya berada di atas nol, artinya rumah yang tergabung tidak berkurang. Itu saya bahas di slide berikutnya.");

  s = d.slide("Dua hasil negatif, dilaporkan apa adanya", "Keduanya dirancang untuk memisahkan rumah berdempetan.");
  d.cards(s, [
    { head: "N3: prior morfologi tidak membantu", body: `Dengan MBI dan peta celah, *merge rate* berubah ${n("NNthreeDeltaMerge")} poin, selang 95% ${n("NNthreeDeltaMergeCi")}. Arahnya salah dan tidak sama di ketiga *seed*.` },
    { head: "Kelas kontak jarang aktif", body: `Di ${n("NContactInactive")} dari ${n("NContactRuns")} *run* FULL, kelas kontak tidak pernah terpilih. Rumah terpisah oleh celah latar, bukan oleh kelas ini.` },
  ]);
  say("Dua hasil negatif, dilaporkan apa adanya",
    "Dua bagian yang dirancang untuk memisahkan rumah berdempetan ternyata tidak bekerja seperti yang saya harapkan.",
    `Yang pertama N3. Dengan MBI dan peta celah sebagai input tambahan, *merge rate* berubah ${n("NNthreeDeltaMerge")} poin, dengan selang ${n("NNthreeDeltaMergeCi")}. Artinya rumah yang tergabung sedikit bertambah, bukan berkurang, dan arahnya pun tidak sama di ketiga *seed*. Jadi hipotesis bahwa prior celah membantu tidak didukung data.`,
    `Yang kedua kelas kontak. Di ${n("NContactInactive")} dari ${n("NContactRuns")} *run* FULL, jaringan tidak pernah memilih kelas kontak sama sekali. Penjelasan yang paling mungkin: piksel kontak sangat langka dibanding piksel bangunan, sehingga jaringan lebih untung mengabaikannya. Rumah-rumah tetap terpisah, tetapi karena ada celah latar di antaranya, bukan karena kelas kontak.`,
    "Keduanya saya laporkan apa adanya, karena hasil negatif yang diukur dengan benar tetap berguna bagi yang ingin mencoba gagasan serupa.");

  s = d.slide("Banding dengan HiSup di data tanpa kebocoran", `${n("NTilesClean")} *tile* val yang tidak punya kembaran di train; kode evaluasi yang sama.`);
  d.table(s, ["Metode", "Anotasi poligon", "AP", "AP50", "AP *boundary*", "C-IoU"], [
    ["HiSup, val penuh (bocor)", "ya", n("NLeakyApHisup"), "–", "–", "–"],
    ["HiSup, *tile* bersih", "ya", n("NCleanApHisup"), n("NCleanApfiftyHisup"), n("NCleanApbHisup"), n("NCleanCiouHisup")],
    ["FULL-P0 (kita), *tile* bersih", "tidak", n("NCleanApFull"), n("NCleanApfiftyFull"), n("NCleanApbFull"), n("NCleanCiouFull")],
  ], { colW: [4.6, 2.1, 1.2, 1.2, 1.6, 1.194], align: "center",
    callout: `HiSup turun dari ${n("NLeakyApHisup")} ke ${n("NCleanApHisup")} AP di data bersih; kita tertinggal ${n("NCleanDeltaApAbs")} poin.` });
  say("Banding dengan HiSup di data tanpa kebocoran",
    "Bagaimana posisi hasil ini dibanding metode terbaik? HiSup merilis file prediksinya, jadi bisa saya nilai ulang dengan kode evaluasi yang sama, tanpa menjalankan modelnya.",
    `Di data validasi penuh, yang bocor, HiSup mendapat AP ${n("NLeakyApHisup")}. Angka itu cocok dengan yang dilaporkan di papernya, dan itu sekaligus memeriksa bahwa kode evaluasi saya benar.`,
    `Tetapi di ${n("NTilesClean")} *tile* validasi yang benar-benar tidak punya kembaran di train, AP HiSup turun menjadi ${n("NCleanApHisup")}. Di *tile* yang sama, FULL-P0 mendapat ${n("NCleanApFull")}, tertinggal ${n("NCleanDeltaApAbs")} poin. Perlu diingat, HiSup dilatih dengan anotasi poligon, sedangkan model saya tidak.`,
    "Ambang \"mendekati metode terbaik\" yang saya bekukan sejak awal adalah tiga poin AP. Selisihnya lebih besar dari itu, jadi klaim tersebut tidak saya pakai.",
    "Temuan yang menurut saya lebih penting di slide ini adalah baris pertama dan kedua: angka HiSup sendiri turun jauh begitu duplikat disingkirkan. Artinya angka di *benchmark* CrowdAI perlu dibaca dengan hati-hati.");

  s = d.slide("Eksploratif: *baseline* sederhana dan *seed* detektor", "Ditambahkan setelah hasil ada, jadi bukan bagian dari klaim.");
  d.cards(s, [
    { head: "B3m: biner + regularisasi", body: `AP ${n("NApBthreem")} dan C-IoU ${n("NCiouBthreem")}; FULL-P0 ${n("NApFull")} dan ${n("NCiouFull")}. B3m unggul di semua metrik kecuali rumah tergabung.` },
    { head: "*Seed* dari detektor", body: `YOLOv8n-OBB, juga dilatih dari *mask*, menurunkan *merge rate* FULL-P0 ${n("NSeedFullMergeAbs")} poin dan B3m ${n("NSeedBinaryMergeAbs")} poin, dengan AP setara.` },
    { head: "Artinya", body: "Jalur biner paling akurat dan rapi. Jalur tiga kelas paling baik memisahkan rumah berdempetan. *Seed* membantu keduanya." },
  ]);
  say("Eksploratif: *baseline* sederhana dan *seed* detektor",
    "Dua percobaan di slide ini saya tambahkan setelah hasil utama ada. Karena itu sifatnya eksploratif dan bukan bagian dari klaim.",
    `Yang pertama B3m: segmentasi biner biasa, ditambah regularisasi yang sama. Ternyata ini konfigurasi yang paling kuat. AP-nya ${n("NApBthreem")} dan C-IoU-nya ${n("NCiouBthreem")}, sedangkan FULL-P0 ${n("NApFull")} dan ${n("NCiouFull")}. B3m unggul di semua metrik kecuali satu, yaitu rumah yang tergabung.`,
    `Yang kedua, sebuah detektor kecil, YOLOv8n-OBB, yang juga dilatih dari *mask*, dipakai untuk memberi titik benih bagi tiap rumah. Hasilnya *merge rate* turun ${n("NSeedFullMergeAbs")} poin di FULL-P0 dan ${n("NSeedBinaryMergeAbs")} poin di B3m, dengan AP yang setara. Ada harganya: rumah yang terpecah bertambah ${n("NSeedFullSplitAbs")} dan ${n("NSeedBinarySplitAbs")} poin.`,
    "Jadi gambaran besarnya: jalur biner paling akurat dan paling rapi, jalur tiga kelas paling baik memisahkan rumah berdempetan, dan benih dari detektor membantu keduanya.");

  s = d.slide("Eksploratif: *head* θ di jalur biner tidak membantu", `B3m → B3θ di ${n("NTilesDense")} *tile* padat; aturannya dibekukan sebelum training.`);
  d.cards(s, [
    { head: "Pemanduan θ tetap membantu", body: `B3θ′ → B3θ: PoLiS ${n("NBinThetaGuidePolis")} piksel dan MTA ${n("NBinThetaGuideMta")}°. Di jalur tiga kelas: ${n("NNoneGuidePolis")} piksel dan ${n("NNoneGuideMta")}°.` },
    { head: "Pelatihan θ menurunkan mutu *mask*", body: `B3m → B3θ′: AP turun ${n("NBinThetaTrainApAbs")} poin dan PoLiS naik ${n("NBinThetaTrainPolisAbs")} piksel. IoU validasi ${n("NFgIouBthreet")}% lawan ${n("NFgIouBthree")}% pada B3.` },
    { head: "Total: aturan tidak terpenuhi", body: `PoLiS ${n("NBinThetaTotalPolis")} piksel, MTA ${n("NBinThetaTotalMta")}°, AP ${n("NBinThetaTotalAp")} poin, searah di ketiga *seed*.` },
  ], { callout: "Hasil negatif: *head* θ tidak memberi perbaikan bersih pada *baseline* terkuat. B3m tetap yang terbaik." });
  say("Eksploratif: *head* θ di jalur biner tidak membantu",
    "Karena B3m paling kuat, pertanyaan yang wajar: bagaimana kalau *head* θ dipasang di jalur biner itu? Percobaan ini juga eksploratif, tetapi aturan penilaiannya tetap saya bekukan sebelum *training*.",
    `Hasilnya negatif, dan penyebabnya bisa dipisahkan menjadi dua. Pemanduan oleh θ sendiri tetap membantu: PoLiS ${n("NBinThetaGuidePolis")} piksel dan MTA ${n("NBinThetaGuideMta")} derajat, hampir sama dengan di jalur tiga kelas.`,
    `Tetapi melatih *head* θ bersama *mask* menurunkan mutu *mask* itu sendiri. IoU validasi turun dari ${n("NFgIouBthree")}% menjadi ${n("NFgIouBthreet")}%, dan AP turun ${n("NBinThetaTrainApAbs")} poin.`,
    `Kerugian itu lebih besar dari keuntungannya. Totalnya, dibanding B3m: PoLiS ${n("NBinThetaTotalPolis")} piksel, MTA ${n("NBinThetaTotalMta")} derajat, dan AP ${n("NBinThetaTotalAp")} poin, searah di ketiga *seed*. Jadi aturannya tidak terpenuhi, dan B3m tetap yang terbaik.`);

  s = d.slide("Contoh hasil di *tile* asli", "*Ground truth* dan poligon tiap model pada *tile* *test* yang sama.");
  d.mediaStack(s, { image: figure("crowdai_qualitative_id.png"), alt: "Poligon tiap model pada tile test", split: 0.6, items: [
    { head: "B4", body: "Kontur mentah: bergelombang dan boros verteks." },
    { head: "FULL-P0, FULL+Y, B3m", body: "Setelah regularisasi: poligon siku dengan sedikit verteks." },
    { head: "Baris terbawah", body: "Contoh kegagalan: rumah berdempetan masih tergabung." },
  ] });
  say("Contoh hasil di *tile* asli",
    "Supaya tidak hanya angka, ini contoh pada *tile* *test*. Kolom paling kiri adalah *ground truth*, digambar hijau. Empat kolom berikutnya adalah poligon dari B4, FULL-P0, FULL+Y, dan B3m, digambar kuning. Angka kecil di pojok kiri bawah tiap gambar adalah jumlah bangunannya.",
    "Di kolom B4, kontur masih mentah: garisnya bergelombang dan titik sudutnya banyak. Di tiga kolom berikutnya, setelah regularisasi, poligon menjadi siku dengan sedikit titik sudut, dan lebih mirip *ground truth*.",
    "Tiga baris pertama adalah *tile* padat yang hasilnya cukup baik. Baris terbawah sengaja saya pilih sebagai contoh kegagalan: di sisi kanan bawah, beberapa rumah yang berdempetan masih tergabung menjadi satu poligon besar, di semua model. Inilah kasus yang belum terselesaikan.");

  s = d.slide("WHU dan biaya komputasi", "Dataset kedua (satu *seed*) dan kecepatan alur penuh.");
  d.cards(s, [
    { head: "WHU", body: `AP ${n("NWhuAp")} dan C-IoU ${n("NWhuCiouMain")}. Dengan aturan Pix2Poly C-IoU ${n("NWhuCiou")}; dilaporkan di paper: HiSup ${n("NWhuCiouHisup")}, Pix2Poly ${n("NWhuCiouPixtopoly")} (kira-kira).` },
    { head: "Titik lemah", body: `${n("NWhuMergeTouching")}% rumah tergabung di *patch* yang berisi bangunan bersentuhan.` },
    { head: "Biaya", body: `${n("NParamsFull")} juta parameter. ${n("NFpsFull")} *tile* per detik di RTX 5090, dan ${n("NFpsFullY")} dengan detektor.` },
  ]);
  say("WHU dan biaya komputasi",
    `Untuk melihat apakah hasilnya terbawa ke data lain, model yang sama saya latih di dataset WHU, dengan satu *seed*. AP-nya ${n("NWhuAp")} dan C-IoU-nya ${n("NWhuCiouMain")}.`,
    `Kalau dihitung dengan aturan Pix2Poly, C-IoU kita ${n("NWhuCiou")}. Angka yang dilaporkan di paper: HiSup ${n("NWhuCiouHisup")} dan Pix2Poly ${n("NWhuCiouPixtopoly")}. Perbandingan ini hanya kira-kira, karena daftar *tile* uji Pix2Poly tidak bisa direkonstruksi persis. Meski begitu, jaraknya jelas besar.`,
    `Titik lemahnya sama dengan di CrowdAI: di *patch* yang berisi bangunan bersentuhan, ${n("NWhuMergeTouching")}% rumah masih tergabung.`,
    `Dari sisi biaya, modelnya ${n("NParamsFull")} juta parameter. Alur penuh, dari citra sampai poligon, berjalan ${n("NFpsFull")} *tile* per detik di RTX 5090, dan ${n("NFpsFullY")} bila detektor ikut dipakai.`);

  s = d.slide("Keterbatasan", "Hal yang membatasi kesimpulan penelitian ini.");
  d.numbered(s, [
    { head: "Data bersih kecil", body: `Banding HiSup hanya di ${n("NTilesClean")} *tile*, dan bukan sampel acak.` },
    { head: "WHU satu *seed*", body: "Daftar *tile* uji tidak persis sama dengan Pix2Poly." },
    { head: "Efek N1 kecil", body: "Di jalur biner, melatih *head* θ malah menurunkan mutu *mask* (satu konfigurasi, eksploratif)." },
    { head: "Satu detektor", body: "YOLO dilatih sekali dan dipakai untuk ketiga *seed*." },
    { head: "Tanpa lubang", body: "Poligon hanya memuat cincin luar bangunan." },
  ], { labelW: 2.9 });
  say("Keterbatasan",
    "Ada lima hal yang membatasi kesimpulan penelitian ini.",
    `Pertama, data bersih untuk banding dengan HiSup hanya ${n("NTilesClean")} *tile*, dan itu bukan sampel acak dari data validasi.`,
    "Kedua, di WHU hanya ada satu *seed*, dan daftar *tile* ujinya tidak persis sama dengan yang dipakai Pix2Poly.",
    "Ketiga, efek N1 kecil. Di jalur biner, melatih *head* θ malah menurunkan mutu *mask*, walaupun itu baru satu konfigurasi dan sifatnya eksploratif.",
    "Keempat, detektor YOLO hanya dilatih sekali dan dipakai untuk ketiga *seed*, jadi ragamnya antar-*seed* tidak terukur.",
    "Kelima, poligon yang dihasilkan hanya cincin luar bangunan. Bangunan yang punya halaman dalam, atau lubang, tidak terwakili.");

  s = d.slide("Kesimpulan", "Empat hal yang bisa dibawa pulang.");
  d.numbered(s, [
    { head: "Poligon rapi bisa dari *mask* saja", body: "Regularisasi geometris bekerja tanpa anotasi poligon." },
    { head: "*Head* θ membantu sedikit", body: "Konsisten di jalur tiga kelas, makin terasa di *tile* padat; tidak terbawa ke jalur biner." },
    { head: "Prior dan kelas kontak gagal", body: "Dilaporkan sebagai hasil negatif." },
    { head: "*Baseline* sederhana paling kuat", body: "Terlihat karena semua model dinilai dengan protokol yang sama." },
  ], { shape: "square", labelW: 4.7 });
  say("Kesimpulan",
    "Ada empat hal yang bisa dibawa pulang.",
    "Pertama, poligon yang rapi bisa dihasilkan dari *mask* saja. Regularisasi geometris bekerja tanpa anotasi poligon.",
    "Kedua, *head* orientasi membantu sedikit. Efeknya konsisten di jalur tiga kelas dan makin terasa di *tile* padat, tetapi tidak terbawa ke jalur biner.",
    "Ketiga, prior morfologi dan kelas kontak tidak bekerja seperti yang diharapkan, dan keduanya dilaporkan sebagai hasil negatif.",
    "Keempat, *baseline* yang sederhana ternyata paling kuat. Hal ini baru terlihat karena semua model dinilai dengan protokol yang sama, di data yang bebas kebocoran.",
    "Kalau dirangkum dalam satu kalimat: sumbangan utama proyek ini bukan model yang mengalahkan metode terbaik, melainkan evaluasi yang jujur tentang seberapa jauh pendekatan berbasis *mask* bisa melangkah.");

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
  say("Referensi",
    "Ini rujukan utama yang saya pakai. Daftar lengkapnya ada di naskah paper.");

  d.closing();
  say("Penutup",
    "Demikian presentasi saya. Kode, konfigurasi, dan semua angka di slide ini bisa direproduksi dari repositori proyek. Terima kasih atas perhatiannya. Saya persilakan bila ada pertanyaan atau masukan.");
  const target = path.join(HERE, "densepoly.pptx");
  writeScript(path.join(HERE, "naskah.md"), d.pres.slides.length);
  await d.save(target);
  console.log(target);
})().catch((error) => {
  console.error(String(error && error.message ? error.message : error));
  process.exit(1);
});

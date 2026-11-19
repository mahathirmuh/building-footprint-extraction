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
const { ITSDeck, C, ACCENTS, X0, X1 } = require("./its_kit");

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

/**
 * A flow diagram in lanes: every lane is a tinted band with its name on the left and its boxes joined by arrows; `links`
 * are the arrows between lanes, {from: [lane, box], to: [lane, box], label}, drawn straight down when the two boxes
 * overlap and as an elbow through the gap otherwise. o: { bottom, headSize, bodySize } for a diagram above a key message.
 */
function flowLanes(d, s, lanes, links = [], o = {}) {
  const P = d.pres.ShapeType;
  const top = 2.24, bottom = o.bottom || 6.62, gapY = 0.33, labelW = 1.72, arrowW = 0.3, pad = 0.12;
  const laneH = (bottom - top - gapY * (lanes.length - 1)) / lanes.length;
  const boxes = lanes.map((lane, i) => {
    const y = top + i * (laneH + gapY), color = ACCENTS[i % ACCENTS.length];
    s.addShape(P.roundRect, { x: X0, y, w: X1 - X0, h: laneH, rectRadius: 0.08, fill: { color: C.paper }, line: { color: C.line, width: 0.75 } });
    d.text(s, lane.name, { x: X0 + 0.16, y, w: labelW - 0.2, h: laneH }, { size: 13.5, bold: true, color, valign: "middle" });
    const x0 = X0 + labelW, n = lane.steps.length, w = (X1 - pad - x0 - arrowW * (n - 1)) / n;
    return lane.steps.map((step, k) => {
      const x = x0 + k * (w + arrowW);
      d.card(s, x, y + pad, w, laneH - 2 * pad, color, step.head, step.body,
        { headSize: o.headSize || 12.5, bodySize: o.bodySize || 10.5, padX: 0.15, padY: 0.1, gap: 3 });
      if (k < n - 1) s.addShape(P.rightArrow, { x: x + w + 0.045, y: y + laneH / 2 - 0.11, w: arrowW - 0.09, h: 0.22, fill: { color }, line: { color, width: 0 } });
      return { x, w, top: y + pad, bottom: y + laneH - pad };
    });
  });
  const grey = C.muted, line = { color: grey, width: 1.5 };
  for (const link of links) {
    const a = boxes[link.from[0]][link.from[1]], b = boxes[link.to[0]][link.to[1]];
    const left = Math.max(a.x, b.x), right = Math.min(a.x + a.w, b.x + b.w), mid = (a.bottom + b.top) / 2;
    if (right - left > 0.4) { // one arrow straight down
      const x = (left + right) / 2;
      s.addShape(P.downArrow, { x: x - 0.11, y: a.bottom + 0.02, w: 0.22, h: b.top - a.bottom - 0.04, fill: { color: grey }, line: { color: grey, width: 0 } });
      if (link.label) d.text(s, link.label, { x: x - 2.85, y: mid - 0.13, w: 2.65, h: 0.26 }, { size: 10.5, color: grey, align: "right", valign: "middle", italic: true });
    } else { // down, across the gap, down
      const xa = a.x + a.w / 2, xb = b.x + b.w / 2;
      s.addShape(P.line, { x: xa, y: a.bottom, w: 0, h: mid - a.bottom, line });
      s.addShape(P.line, { x: Math.min(xa, xb), y: mid, w: Math.abs(xa - xb), h: 0, line });
      s.addShape(P.downArrow, { x: xb - 0.11, y: mid - 0.01, w: 0.22, h: b.top - mid - 0.01, fill: { color: grey }, line: { color: grey, width: 0 } });
      if (link.label) d.text(s, link.label, { x: Math.min(xa, xb) + 0.3, y: mid - 0.19, w: Math.abs(xa - xb) - 0.6, h: 0.17 }, { size: 10.5, color: grey, align: "center", valign: "middle", italic: true });
    }
  }
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
    { head: "Ketemunya apa?", body: `Dari *mask* saja, regularisasi menaikkan C-IoU (kerapian poligon) dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}. *Head* θ menambah sedikit kerapian. Prior morfologi dan kelas kontak tidak membantu. Di data bersih, AP HiSup ternyata ${n("NCleanApHisup")}, bukan ${n("NLeakyApHisup")} seperti di data bocor.` },
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

  s = d.slide("*State of the art* 2022–2026: ekstraksi poligon bangunan", "Lima tahun terakhir; AP dan C-IoU di val CrowdAI, seperti dilaporkan di paper masing-masing.");
  const reported = (metric, code) => (`NReported${metric}${code}` in numbers ? n(`NReported${metric}${code}`) : "–");
  const sota = (label, idea, code) => [label, idea, reported("Ap", code), reported("Ciou", code)];
  d.table(s, ["Metode", "Gagasan utama", "AP", "C-IoU"], [
    sota("PolyWorld (Zorzi dkk., 2022)", "Deteksi titik sudut, dihubungkan *graph neural network*", "Polyworld"),
    sota("HiSup (Xu dkk., 2023)", "Supervisi hierarkis: titik sudut, garis batas, dan *mask*", "Hisup"),
    sota("PolyR-CNN (Jiao dkk., 2024)", "R-CNN *end-to-end* yang langsung memprediksi poligon", "Polyrcnn"),
    sota("Pix2Poly (Adimoolam dkk., 2025)", "*Transformer* yang memprediksi urutan titik sudut", "Pixtopoly"),
    sota("RoIPoly (Jiao dkk., 2025)", "Tiap titik sudut menjadi *query* *transformer*", "Roipoly"),
    sota("PolyBuild (Zhang dkk., 2026)", "Kontur awal dari kotak deteksi, disempurnakan *transformer*", "Polybuild"),
    ["Penelitian ini (FULL-P0), *test* bersih", "Segmentasi + θ dari *mask* + regularisasi klasik", n("NApFull"), n("NCiouFull")],
  ], { colW: [3.95, 5.55, 1.15, 1.244], align: ["left", "left", "center", "center"],
    callout: "Angka paper diukur di val CrowdAI yang bocor: tidak sebanding langsung dengan baris terakhir. Banding yang adil ada di slide HiSup." });
  say("*State of the art* 2022–2026: ekstraksi poligon bangunan",
    "Sebelum masuk ke gagasan saya, ini peta metode terkini untuk ekstraksi poligon bangunan, dalam lima tahun terakhir.",
    "Semua metode di tabel ini langsung memprediksi titik sudut atau kontur poligon. PolyWorld, tahun 2022, mendeteksi titik sudut lalu menghubungkannya dengan *graph neural network*. HiSup, tahun 2023, mengawasi titik sudut, garis batas, dan *mask* sekaligus. PolyR-CNN, tahun 2024, dan Pix2Poly, tahun 2025, memprediksi poligon secara *end-to-end*. RoIPoly, juga 2025, menjadikan tiap titik sudut sebuah *query* pada *transformer*. Yang terbaru, PolyBuild tahun 2026, membuat kontur awal dari kotak deteksi lalu menyempurnakannya.",
    `Angka tertinggi dilaporkan RoIPoly, yaitu AP ${n("NReportedApRoipoly")}; HiSup, PolyR-CNN, dan Pix2Poly berada di sekitar ${n("NReportedApPolyrcnn")} sampai ${n("NReportedApPixtopoly")}. Menariknya, metode yang lebih baru tidak selalu lebih tinggi: PolyBuild melaporkan AP ${n("NReportedApPolybuild")}. Pendekatan saya berbeda keluarga: segmentasi dulu, lalu dirapikan, seperti *frame field learning* dari tahun 2021 yang menjadi pendahulunya.`,
    "Ada dua catatan penting. Pertama, metode berbasis titik sudut itu dilatih dengan anotasi poligon, sedangkan penelitian ini hanya memakai *mask*. Kedua, semua angka paper tadi diukur di data validasi CrowdAI yang bocor.",
    `Baris terakhir adalah model saya di data *test* yang bersih: AP ${n("NApFull")} dan C-IoU ${n("NCiouFull")}. Karena datanya berbeda, baris itu tidak bisa dibandingkan langsung dengan baris di atasnya. Perbandingan yang adil saya tunjukkan nanti, di slide HiSup.`);

  s = d.slide("Kebaruan penelitian: tiga gagasan yang diuji", "Cara menilai tiap gagasan ditulis dan dibekukan sebelum ada model yang dilatih.");
  d.table(s, ["Kebaruan", "Apa yang baru", "Hasil uji"], [
    ["N1: metode", "*Head* orientasi θ yang labelnya diturunkan dari *mask*, lalu memandu perapian poligon", "Didukung; efeknya kecil"],
    ["N2: evaluasi", "Metrik per tingkat kepadatan di CrowdAI yang sudah bebas duplikat", "Terlaksana; angka *benchmark* yang bocor terbukti terlalu tinggi"],
    ["N3: *preprocessing*", "Prior morfologi (MBI dan peta celah) sebagai kanal input tambahan", "Tidak didukung; dilaporkan sebagai hasil negatif"],
  ], { colW: [2.5, 5.6, 3.794], align: ["left", "left", "left"], rowH: 0.75,
    callout: "Diadopsi, bukan kebaruan: kelas kontak + *watershed*, bobot tepi, U-Net++, MBI, dan deduplikasi *p-hash*." });
  say("Kebaruan penelitian: tiga gagasan yang diuji",
    "Ini kebaruan penelitian saya: tiga gagasan, dan ketiganya saya uji.",
    "N1 adalah kebaruan di sisi metode: jaringan diberi satu keluaran tambahan, disebut *head* orientasi, yang menebak arah hadap bangunan, yaitu sudut θ. Yang baru, label sudutnya dihitung otomatis dari *mask*, lalu sudut itu dipakai untuk merapikan poligon. Hasil ujinya: didukung, tetapi efeknya kecil.",
    "N2 adalah kebaruan di sisi evaluasi: dataset CrowdAI dibersihkan dari duplikat, lalu hasil dilaporkan per tingkat kepadatan bangunan. Dari sini muncul temuan terkuat penelitian ini, yaitu bahwa angka *benchmark* yang bocor terlalu tinggi.",
    "N3 adalah kebaruan di sisi *preprocessing*: dua peta dari morfologi klasik, yaitu MBI dan peta celah, ditambahkan sebagai kanal input, dengan harapan celah antar-rumah lebih mudah terlihat oleh jaringan. Hasil ujinya: tidak didukung, dan saya laporkan sebagai hasil negatif.",
    "Supaya jelas batasnya, ada komponen yang saya adopsi dari penelitian lain dan tidak saya klaim sebagai kebaruan: kelas kontak dengan *watershed*, bobot tepi pada *loss*, arsitektur U-Net++, indeks MBI itu sendiri, dan deduplikasi dengan *perceptual hash*.",
    "Satu hal lagi yang ingin saya tekankan: kriteria berhasil atau gagal untuk tiap gagasan saya tulis dan saya bekukan sebelum ada model yang dilatih. Setiap perbandingan memakai tiga *seed* dan selang kepercayaan *bootstrap*. Karena itulah hasil negatif pun tetap dilaporkan.");

  s = d.slide("Metode yang diusulkan", "U-Net++ dengan *head* orientasi berbasis *mask* dan regularisasi poligon terpandu θ.");
  flowLanes(d, s, [
    { name: "Jaringan", steps: [
      { head: "Input", body: "Citra RGB. Dua kanal prior morfologi (N3) diuji sebagai tambahan, tetapi tidak dipakai di model akhir." },
      { head: "U-Net++ EfficientNet-B4", body: "Satu jaringan segmentasi dengan satu *head* konvolusi berisi lima kanal keluaran." },
      { head: "Tiga kelas + *head* θ (N1)", body: "Latar, bangunan, kontak, dan orientasi (sin 4θ, cos 4θ). Label θ dari *minAreaRect* tiap bangunan di *mask*." },
    ] },
    { name: "Poligon", steps: [
      { head: "Marker + *watershed*", body: "Marker dari selisih peluang bangunan dan kontak; satu *instance* per rumah." },
      { head: "Kontur per rumah", body: "Batas tiap *instance* diambil dengan *marching squares*." },
      { head: "Regularisasi terpandu θ", body: "Bangunan hampir persegi menjadi kotak searah θ; lainnya disederhanakan lalu sisinya diluruskan ke θ." },
      { head: "Keluaran", body: "Satu poligon per rumah, COCO JSON dan GeoJSON." },
    ] },
  ], [{ from: [0, 2], to: [1, 0], label: "peta kelas dan θ" }], { bottom: 5.95, headSize: 13.5, bodySize: 11.5 });
  d.callout(s, "*Loss*: CE berbobot tepi + Dice + λ × *loss* θ. Semua label dari *mask*: tanpa anotasi poligon saat *training*.");
  say("Metode yang diusulkan",
    "Inilah metode yang saya usulkan, dalam dua baris: jaringannya di atas, pembentukan poligonnya di bawah.",
    "Jaringannya U-Net++ dengan *encoder* EfficientNet-B4. Inputnya citra RGB. Dalam rancangan awal ada dua kanal tambahan berisi prior morfologi, yaitu gagasan N3; itu diuji, tetapi karena tidak membantu, model akhirnya hanya memakai RGB.",
    "Yang membedakan jaringan ini adalah keluarannya. Selain tiga kelas, yaitu latar, bangunan, dan kontak, ada *head* orientasi yang memprediksi sudut bangunan θ dalam bentuk sinus dan kosinus 4θ. Inilah gagasan N1. Label θ tidak dianotasi manusia: ia dihitung dari kotak terkecil tiap bangunan di *mask*.",
    "Di baris bawah, peta kelas dipakai membuat *marker*, lalu *watershed* memisahkan rumah. Kontur tiap rumah diambil, lalu dirapikan dengan panduan θ dari jaringan: bangunan yang hampir persegi diganti kotak searah θ, dan bangunan lain disederhanakan lalu sisi-sisinya diluruskan ke arah θ. Hasilnya satu poligon per rumah.",
    "Jaringan dilatih dengan satu *loss* gabungan: *cross-entropy* berbobot tepi, Dice, dan *loss* orientasi. Semua labelnya diturunkan dari *mask*, jadi tidak ada anotasi poligon saat *training*.");

  s = d.slide("Dataset: apa dan dari mana", "Dua dataset publik; keduanya diunduh dari halaman resminya.");
  d.table(s, ["", "CrowdAI Mapping Challenge", "WHU Building (aerial)"], [
    ["Sumber", "AIcrowd, *Mapping Challenge* (aicrowd.com), versi 1.0", "Universitas Wuhan, grup GPCV (gpcv.whu.edu.cn); Ji dkk., 2019"],
    ["Citra", "Satelit RGB, *tile* 300 × 300 piksel", "Foto udara RGB 0,2 m, *tile* 1.024 × 1.024 piksel"],
    ["Jumlah asli", `Train ${n("NTrainImages")}, val ${n("NValImages")} *tile*`, "Train 2.943, val 627, *test* 2.220 *tile*"],
    ["Anotasi", "Poligon per bangunan (COCO)", "*Mask* raster dan COCO per bangunan"],
    ["Setelah disiapkan", `Bebas duplikat: train ${n("NTrainDedup")}, tuning ${n("NTuneTiles")} + ${n("NSelectTiles")}, *test* ${n("NTestTiles")}`, "Dipotong menjadi *patch* 224 × 224 (protokol Pix2Poly)"],
    ["Peran", "Dataset utama: semua klaim dan banding HiSup", "Dataset kedua, satu *seed*"],
  ], { colW: [2.2, 5.0, 4.694], align: ["left", "left", "left"],
    callout: "Untuk *training*, anotasi hanya dipakai sebagai *mask*; poligon aslinya hanya menjadi *ground truth* saat evaluasi." });
  say("Dataset: apa dan dari mana",
    "Penelitian ini memakai dua dataset publik, dan keduanya saya unduh dari halaman resminya.",
    `Yang utama adalah CrowdAI Mapping Challenge, dari platform AIcrowd, versi 1.0. Isinya citra satelit RGB berupa *tile* 300 × 300 piksel: ${n("NTrainImages")} *tile* train dan ${n("NValImages")} *tile* validasi, dengan anotasi poligon per bangunan dalam format COCO. Versi 1.0 saya pilih karena metode pembanding juga memakai versi itu.`,
    `Seperti sudah saya singgung, dataset ini banyak duplikatnya. Setelah deduplikasi, yang dipakai adalah ${n("NTrainDedup")} *tile* train, ${n("NTuneTiles")} dan ${n("NSelectTiles")} *tile* untuk tuning, serta ${n("NTestTiles")} *tile* *test*. Semua klaim dan perbandingan dengan HiSup memakai dataset ini.`,
    "Dataset kedua adalah WHU Building, dari grup GPCV Universitas Wuhan, yang diperkenalkan Ji dan rekan-rekannya tahun 2019. Ini foto udara dengan resolusi 0,2 meter per piksel, dalam *tile* 1.024 × 1.024 piksel: 2.943 train, 627 validasi, dan 2.220 *test*. Mengikuti protokol Pix2Poly, *tile* itu dipotong menjadi *patch* 224 × 224. WHU saya pakai sebagai dataset kedua, dengan satu *seed*, untuk melihat apakah hasilnya terbawa ke data lain.",
    "Satu hal yang perlu jelas: kedua dataset sebenarnya menyediakan anotasi per bangunan, dan di CrowdAI bentuknya poligon. Tetapi untuk *training* saya hanya memakainya sebagai *mask*. Poligon aslinya hanya dipakai sebagai *ground truth* saat evaluasi. Itulah arti kalimat tanpa anotasi poligon di penelitian ini.");

  s = d.slide("Arsitektur: alur pengerjaan dari awal sampai akhir", "Tiga tahap: menyiapkan data, melatih jaringan, lalu membuat poligon dan menilainya.");
  flowLanes(d, s, [
    { name: "1. Data dan label", steps: [
      { head: "CrowdAI dan WHU", body: "Citra RGB dan *mask* per bangunan; tanpa anotasi poligon." },
      { head: "Deduplikasi dan *split*", body: "*p-hash* membuang duplikat; train, tuning, dan *test* dipisah per klaster." },
      { head: "Label dari *mask*", body: "Kelas kontak, orientasi θ, dan bobot tepi, dibuat otomatis." },
    ] },
    { name: "2. Jaringan", steps: [
      { head: "Citra RGB", body: "*Tile* 300 × 300 piksel, di-*pad* ke 320." },
      { head: "U-Net++", body: "*Encoder* EfficientNet-B4 (ImageNet)." },
      { head: "Lima kanal keluaran", body: "Latar, bangunan, kontak, dan (sin 4θ, cos 4θ)." },
      { head: "*Loss* saat *training*", body: "CE berbobot tepi + Dice + λ × *loss* θ." },
    ] },
    { name: "3. Poligon dan evaluasi", steps: [
      { head: "Marker + *watershed*", body: "Satu *instance* per rumah." },
      { head: "Kontur", body: "*Marching squares* per rumah." },
      { head: "Regularisasi", body: "Kotak atau poligon siku searah θ." },
      { head: "Poligon", body: "COCO JSON dan GeoJSON." },
      { head: "Evaluasi", body: "AP, C-IoU, PoLiS, MTA; per kepadatan." },
    ] },
  ], [
    { from: [0, 2], to: [1, 3], label: "label, hanya untuk *training*" },
    { from: [1, 2], to: [2, 0], label: "keluaran jaringan" },
  ]);
  say("Arsitektur: alur pengerjaan dari awal sampai akhir",
    "Ini arsitektur keseluruhannya, dalam tiga tahap. Tiap baris dibaca dari kiri ke kanan.",
    "Baris pertama adalah data dan label. Datanya CrowdAI dan WHU, berupa citra RGB dan *mask* per bangunan, tanpa anotasi poligon. Data itu dideduplikasi dengan *perceptual hash*, lalu dibagi menjadi train, tuning, dan *test* per klaster, supaya tidak ada citra kembar di dua bagian. Dari *mask* kemudian dibuat tiga label secara otomatis: kelas kontak, orientasi θ, dan bobot tepi.",
    "Baris kedua adalah jaringannya. Inputnya satu *tile* 300 × 300 piksel, yang hanya di-*pad* ke 320 dan tidak diubah ukurannya. Jaringannya U-Net++ dengan *encoder* EfficientNet-B4. Keluarannya lima kanal: tiga kelas, yaitu latar, bangunan, dan kontak, ditambah dua kanal orientasi. Saat *training*, keluaran itu dibandingkan dengan label dari baris pertama lewat *loss* gabungan; itulah arti panah turun di sebelah kanan.",
    "Baris ketiga mengubah keluaran jaringan menjadi poligon. Dari peta kelas dibuat *marker*, yaitu inti tiap rumah, lalu *watershed* memekarkannya sehingga satu gumpalan terpecah menjadi satu *instance* per rumah. Kontur tiap rumah diambil, lalu dirapikan searah θ: bangunan yang hampir persegi menjadi kotak, yang lain disederhanakan dan dindingnya diluruskan. Hasilnya satu poligon per rumah, dalam format COCO JSON dan GeoJSON.",
    "Kotak terakhir adalah evaluasi: poligon dinilai dengan AP, C-IoU, PoLiS, dan MTA, dan hasilnya dilaporkan per tingkat kepadatan. Jadi dari awal sampai akhir, tidak ada anotasi poligon yang dipakai.");

  s = d.slide("Pengolahan citra klasik di dalam alur", "Algoritma tanpa pembelajaran yang dipakai sebelum dan sesudah jaringan.");
  d.table(s, ["Tahap", "Algoritma", "Perannya di sini"], [
    ["Label dari *mask*", "Dilasi; *distance transform*; *minAreaRect*", "Kelas kontak; bobot tepi; orientasi θ"],
    ["Marker", "*Thresholding*, *connected component labeling*", "Inti tiap rumah"],
    ["Pemisahan rumah", "*Watershed* berbasis marker", "Satu *instance* per rumah"],
    ["Kontur", "*Marching squares*", "Batas rumah tepat di tepi piksel"],
    ["Regularisasi", "Momen orde dua; Douglas-Peucker (geometri komputasi)", "Kotak dan poligon siku searah θ"],
    ["*Baseline* B0", "MBI, Otsu, *opening*, *h-maxima*, *watershed*", `Pembanding tanpa jaringan: AP ${n("NApBzero")}`],
    ["Prior N3", "*Top-hat* multi-arah (MBI dan peta celah)", "Input tambahan; tidak membantu"],
  ], { colW: [2.6, 5.2, 4.094], align: ["left", "left", "left"],
    callout: "Jaringan hanya memberi peta kelas dan θ; pembentukan poligon seluruhnya dikerjakan algoritma klasik." });
  say("Pengolahan citra klasik di dalam alur",
    "Karena ini mata kuliah Pengolahan Citra Digital, saya tunjukkan di mana saja algoritma klasik dipakai. Yang saya maksud klasik adalah algoritma tetap, tanpa parameter yang dilatih.",
    "Lima baris pertama ada di alur utama. Untuk membuat label: dilasi menghasilkan kelas kontak, *distance transform* menghasilkan bobot tepi, dan *minAreaRect* menghasilkan orientasi. Setelah jaringan: *thresholding* dan *connected component labeling* membuat *marker*, *watershed* memisahkan rumah, *marching squares* mengambil kontur, lalu momen orde dua dan algoritma Douglas-Peucker merapikannya menjadi kotak atau poligon siku.",
    "Jadi jaringan hanya mengerjakan bagian tengah, yaitu dari citra ke peta kelas dan orientasi. Yang mengubah peta itu menjadi satu poligon per rumah seluruhnya algoritma klasik. Untuk tepatnya, *minAreaRect* dan Douglas-Peucker bekerja pada titik dan garis, jadi lebih pas disebut geometri komputasi.",
    `Baris keenam adalah *baseline* B0, yang seluruhnya klasik tanpa jaringan: indeks bangunan MBI, ambang Otsu, *opening*, lalu *watershed*. AP-nya hanya ${n("NApBzero")}, jadi cara klasik saja tidak cukup untuk data ini.`,
    "Baris terakhir adalah gagasan N3: peta dari *top-hat* multi-arah dijadikan input tambahan bagi jaringan. Hasilnya tidak membantu, dan itu saya bahas di slide hasil negatif.");

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

  s = d.slide("Protokol uji yang adil", "Semua model jaringan dilatih dan dinilai dengan setelan yang sama.");
  d.table(s, ["Aspek", "Nilai (sama untuk semua model)"], [
    ["Data", "*Split* bebas duplikat yang sama; *tile* 300 × 300 di-*pad* ke 320, tanpa *resize*; augmentasi yang sama"],
    ["Optimizer", "AdamW, lr 2e-4 (*warmup* 1 *epoch*, lalu kosinus ke 1e-6), *weight decay* 1e-4, AMP fp16"],
    ["Training", "*Batch* 16, 60 *epoch* tanpa *early stopping*, *seed* 0, 1, 2"],
    ["Pemilihan", "*Checkpoint* terbaik di data validasi (bukan *test*); *post-processing* disetel dengan grid yang sama, lalu dibekukan"],
    ["Evaluasi", `${n("NTestTiles")} *tile* *test* yang sama, kode metrik yang sama, selisih berpasangan per *tile*, *bootstrap* 1.000 replikasi`],
    ["Pretrained", "*Encoder* EfficientNet-B4 ImageNet, bobot awal identik di semua model jaringan"],
  ], { colW: [1.6, 5.75], w: 7.35, size: 12, rowH: 0.56, align: ["left", "left"],
    callout: "Setelan identik, diperiksa dari `config.yaml` 23 *run* CrowdAI. Metrik: AP, AP *boundary*, IoU, C-IoU, PoLiS, MTA, *merge rate*." });
  d.card(s, X0 + 7.6, 2.24, X1 - X0 - 7.6, 0.56 * 7, ACCENTS[5], "Yang berbeda (hanya ini)",
    "1. Komponen yang diuji di tiap langkah ablasi: arsitektur, *loss*, keluaran, atau input.\n2. B0 tidak dilatih; B1 dan B2 satu *seed*, model lain tiga *seed*.\n3. B3+θ memakai jaringan B3 yang dibekukan dan *checkpoint* *epoch* terakhir, karena *mask*-nya tidak berubah.",
    { headSize: 15, bodySize: 13, gap: 10 });
  say("Protokol uji yang adil",
    "Supaya perbandingan antarmodel adil, semua model jaringan dilatih dan dinilai dengan setelan yang sama. Tabel di kiri merangkumnya.",
    "Datanya sama: *split* bebas duplikat yang sama, *tile* 300 × 300 yang hanya di-*pad* ke 320 tanpa diubah ukurannya, dan augmentasi yang sama. Optimizer-nya sama: AdamW dengan *learning rate* 2e-4, *warmup* satu *epoch* lalu turun mengikuti kosinus. Lama latihannya sama: *batch* 16 dan 60 *epoch* tanpa *early stopping*, dengan *seed* 0, 1, dan 2.",
    `Pemilihannya juga sama: *checkpoint* terbaik dipilih di data validasi, bukan di data *test*, dan *post-processing* tiap model disetel dengan grid yang sama lalu dibekukan. Evaluasinya memakai ${n("NTestTiles")} *tile* *test* yang sama, kode metrik yang sama, dan selisih berpasangan per *tile*. Semua jaringan berangkat dari bobot ImageNet yang sama.`,
    "Yang berbeda hanya tiga hal, di panel kanan. Pertama, komponen yang memang sedang diuji di tiap langkah ablasi. Kedua, B0 tidak dilatih karena murni klasik, dan B1 serta B2 hanya satu *seed*. Ketiga, B3+θ memakai jaringan B3 yang dibekukan dan *checkpoint* *epoch* terakhir, karena *mask*-nya tidak berubah sehingga tidak ada yang bisa dipilih.",
    "Kesamaan setelan ini saya periksa dari berkas konfigurasi yang tersimpan di tiap *run*, bukan dari ingatan.");

  s = d.slide("Ablasi: satu perubahan per langkah", "CrowdAI *test*. B0–B2 satu *run*; lainnya rata-rata tiga *seed*. MTA dan *merge rate*: makin kecil makin baik.");
  const step = (label, added, code) => [label, added, n(`NAp${code}`), n(`NCiou${code}`), n(`NMta${code}`), n(`NMerge${code}`)];
  d.table(s, ["Model", "Yang ditambahkan", "AP", "C-IoU", "MTA (°)", "*Merge* (%)"], [
    step("B0", "Klasik saja, tanpa jaringan", "Bzero"),
    step("B1", "U-Net EfficientNet-B4, segmentasi biner", "Bone"),
    step("B2", "+ U-Net++", "Btwo"),
    step("B3", "+ bobot tepi pada *loss*", "Bthree"),
    step("B4", "+ kelas kontak dan *watershed*", "Bfour"),
    step("B5", "+ regularisasi, θ dari *minAreaRect*", "Bfive"),
    step("FULL-P0", "+ *head* θ (N1)", "Full"),
    step("FULL-P2", "+ prior MBI dan peta celah (N3)", "FullPtwo"),
  ], { colW: [1.7, 5.1, 1.2, 1.25, 1.3, 1.344], align: ["left", "left", "center", "center", "center", "center"], rowH: 0.4,
    callout: "Kerapian naik paling besar saat regularisasi masuk (B4 → B5). Kelas kontak menurunkan *merge rate*, tetapi AP ikut turun." });
  say("Ablasi: satu perubahan per langkah",
    "Ini tabel ablasinya. Tiap baris menambah satu komponen ke baris di atasnya, jadi pengaruh tiap komponen bisa dibaca sendiri-sendiri.",
    `Baris pertama, B0, adalah cara klasik saja tanpa jaringan: AP-nya hanya ${n("NApBzero")}. Begitu jaringan dipakai, di B1, AP langsung menjadi ${n("NApBone")}. Mengganti U-Net dengan U-Net++, di B2, hampir tidak mengubah apa-apa. Menambah bobot tepi pada *loss*, di B3, menaikkan AP menjadi ${n("NApBthree")}.`,
    `Di B4, kelas kontak dan *watershed* ditambahkan. *Merge rate* turun dari ${n("NMergeBthree")} menjadi ${n("NMergeBfour")} persen, artinya lebih sedikit rumah yang tergabung. Tetapi AP ikut turun menjadi ${n("NApBfour")}. Inilah harga dari jalur tiga kelas.`,
    `Lompatan terbesar ada di B5, saat regularisasi masuk: C-IoU naik dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}, dan galat sudut MTA turun dari ${n("NMtaBfour")} ke ${n("NMtaBfive")} derajat.`,
    `Dua baris terakhir adalah gagasan saya. *Head* θ, di FULL-P0, menambah sedikit kerapian: C-IoU ${n("NCiouFull")} dan MTA ${n("NMtaFull")}. Prior morfologi, di FULL-P2, tidak membawa perbaikan. Dua hal itu saya uji lebih teliti di slide-slide berikutnya.`);

  s = d.slide("Hasil utama di CrowdAI *test*", "Rata-rata tiga *seed*. MTA dan *merge rate*: makin kecil makin baik. Nilai terbaik tiap kolom dicetak tebal.");
  const mains = [
    ["B3: segmentasi biner", "Bthree"],
    ["B3m: B3 + regularisasi (eksploratif)", "Bthreem"],
    ["B3+θ: B3 beku + cabang θ (eksploratif)", "Bthreeh"],
    ["B4: tiga kelas + *watershed*", "Bfour"],
    ["B5: B4 + regularisasi", "Bfive"],
    ["FULL-P0: + *head* θ", "Full"],
    ["FULL+Y: + *seed* detektor", "FullY"],
    ["B3m+Y (eksploratif)", "BthreemY"],
  ];
  const columns = [["NAp", Math.max], ["NCiou", Math.max], ["NMta", Math.min], ["NMerge", Math.min]];  // the best value of each is bold
  const value = (prefix, code) => parseFloat(numbers[`${prefix}${code}`]);
  const row = (label, code) => [label, ...columns.map(([prefix, pick]) =>
    (value(prefix, code) === pick(...mains.map(([, c]) => value(prefix, c))) ? `**${n(prefix + code)}**` : n(prefix + code)))];
  d.table(s, ["Model", "AP", "C-IoU", "MTA (°)", "*Merge rate* (%)"], mains.map(([label, code]) => row(label, code)),
    { colW: [5.3, 1.5, 1.6, 1.6, 1.894], align: "center",
      callout: "Regularisasi yang merapikan poligon. Jalur biner unggul di AP; jalur tiga kelas unggul memisahkan rumah." });
  say("Hasil utama di CrowdAI *test*",
    "Tabel ini merangkum hasil di data *test*, rata-rata tiga *seed*. Sebelum membacanya, sedikit tentang metriknya. AP mengukur ketepatan deteksi tiap bangunan; makin besar makin baik. C-IoU adalah IoU yang dikenai penalti bila jumlah titik sudut poligon jauh dari *ground truth*, jadi ia mengukur kerapian. MTA adalah galat sudut dinding dalam derajat. *Merge rate* adalah persentase rumah yang tergabung dengan tetangganya. Untuk MTA dan *merge rate*, makin kecil makin baik.",
    `Ada tiga hal yang bisa dibaca. Pertama, regularisasi sangat menaikkan kerapian: dari B4 ke B5, C-IoU naik dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}, dan MTA turun dari ${n("NMtaBfour")} ke ${n("NMtaBfive")} derajat.`,
    `Kedua, jalur biner unggul di AP: B3m ${n("NApBthreem")}, sedangkan FULL-P0 ${n("NApFull")}.`,
    `Ketiga, jalur tiga kelas lebih baik memisahkan rumah: *merge rate* FULL-P0 ${n("NMergeFull")}%, B3m ${n("NMergeBthreem")}%. Dengan bantuan detektor, FULL+Y turun lagi ke ${n("NMergeFullY")}%.`,
    "Jadi tidak ada satu model yang menang di semua kolom. Itu saya uraikan satu per satu. Baris B3+θ adalah percobaan eksploratif yang saya jelaskan di slide tersendiri.");

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
    "Bagaimana posisi hasil ini dibanding metode terbaik? Pembandingnya HiSup, metode ekstraksi poligon bangunan dari Xu dan rekan-rekannya tahun 2023, yang dilatih dengan anotasi poligon dan termasuk yang terbaik di CrowdAI. HiSup merilis file prediksinya, jadi bisa saya nilai ulang dengan kode evaluasi yang sama, tanpa menjalankan modelnya.",
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

  s = d.slide("Eksploratif: *head* θ di jalur biner", `Dua cara memasang θ pada B3, dibanding B3m di ${n("NTilesDense")} *tile* padat; aturannya dibekukan sebelum *training*.`);
  d.cards(s, [
    { head: "Dilatih bersama (B3θ): gagal", body: `Mutu *mask* turun: IoU validasi ${n("NFgIouBthreet")}% lawan ${n("NFgIouBthree")}% pada B3. Total: PoLiS ${n("NBinThetaTotalPolis")} piksel, MTA ${n("NBinThetaTotalMta")}°, AP ${n("NBinThetaTotalAp")} poin.` },
    { head: "Cabang θ di atas B3 beku (B3+θ): berhasil", body: `*Mask* tidak berubah; hanya cabang θ yang dilatih. Total: PoLiS ${n("NFrozThetaTotalPolis")} piksel ${n("NFrozThetaTotalPolisCi")}, MTA ${n("NFrozThetaTotalMta")}° ${n("NFrozThetaTotalMtaCi")}, AP ${n("NFrozThetaTotalAp")} poin.` },
    { head: "Artinya", body: `Pemanduan θ juga memperbaiki *baseline* terkuat, asalkan *mask* dilindungi. Efeknya kecil, dan di semua *tile* AP turun ${n("NThetaFrozenApAbs")} poin.` },
  ], { callout: "Aturan beku terpenuhi untuk B3+θ. Keduanya eksploratif, jadi keputusan N1–N3 tidak berubah." });
  say("Eksploratif: *head* θ di jalur biner",
    "Karena B3m paling kuat, pertanyaan yang wajar: bagaimana kalau *head* θ dipasang di jalur biner itu? Saya mencoba dua cara. Keduanya eksploratif, tetapi aturan penilaiannya tetap saya bekukan sebelum *training*.",
    `Cara pertama, B3θ: *head* θ dilatih bersama *mask*. Hasilnya negatif. Pemanduannya sendiri membantu, tetapi melatih θ bersama *mask* menurunkan mutu *mask*: IoU validasi turun dari ${n("NFgIouBthree")}% menjadi ${n("NFgIouBthreet")}%. Totalnya, dibanding B3m, PoLiS ${n("NBinThetaTotalPolis")} piksel, MTA ${n("NBinThetaTotalMta")} derajat, dan AP ${n("NBinThetaTotalAp")} poin: lebih buruk.`,
    `Dari kegagalan itu muncul cara kedua, B3+θ: jaringan B3 dibekukan, lalu hanya sebuah cabang kecil untuk θ yang dilatih di atasnya. Dengan begitu *mask* tidak berubah sama sekali. Hasilnya positif: dibanding B3m, PoLiS ${n("NFrozThetaTotalPolis")} piksel dan MTA ${n("NFrozThetaTotalMta")} derajat, keduanya dengan selang kepercayaan di bawah nol dan searah di ketiga *seed*. AP berubah ${n("NFrozThetaTotalAp")} poin, masih di dalam batas yang dibekukan.`,
    `Jadi pemanduan θ juga memperbaiki *baseline* terkuat, asalkan *mask*-nya dilindungi. Tetapi saya tidak ingin melebih-lebihkan: efeknya kecil, dan kalau dihitung di semua *tile*, AP turun ${n("NThetaFrozenApAbs")} poin. Karena ditambahkan setelah hasil utama ada, temuan ini saya laporkan sebagai eksploratif dan tidak mengubah keputusan N1 sampai N3.`);

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

  s = d.slide("Ringkasan hasil", "Angka dari CrowdAI *test* kecuali disebut lain; rinciannya ada di slide-slide sebelumnya.");
  d.table(s, ["Aspek", "Hasil", "Angka kunci"], [
    ["Poligon dari *mask* saja", "Regularisasi merapikan poligon", `C-IoU ${n("NCiouBfour")} → ${n("NCiouBfive")}`],
    ["*Head* θ (N1)", "Didukung; efeknya kecil", `PoLiS ${n("NNoneTotalPolis")} px, MTA ${n("NNoneTotalMta")}° di *tile* padat`],
    ["Prior morfologi (N3)", "Tidak didukung", `*Merge rate* ${n("NNthreeDeltaMerge")} poin`],
    ["Model terbaik", "B3m: segmentasi biner + regularisasi", `AP ${n("NApBthreem")}, C-IoU ${n("NCiouBthreem")}`],
    ["Cabang θ di atas B3 beku", "Didukung (eksploratif)", `PoLiS ${n("NFrozThetaTotalPolis")} px, MTA ${n("NFrozThetaTotalMta")}° di *tile* padat`],
    ["Pemisahan rumah", "*Seed* detektor mengurangi rumah tergabung", `*Merge rate* ${n("NMergeFull")}% → ${n("NMergeFullY")}%`],
    ["Lawan SOTA, data bersih", "Masih di bawah HiSup", `AP ${n("NCleanApFull")} lawan ${n("NCleanApHisup")}`],
    ["WHU dan kecepatan", "Dataset kedua, satu *seed*", `AP ${n("NWhuAp")}; ${n("NFpsFull")} *tile* per detik`],
  ], { colW: [3.2, 4.6, 4.094], align: ["left", "left", "left"], rowH: 0.4,
    callout: "Yang berhasil: regularisasi dan pemanduan θ. Yang tidak: prior morfologi, kelas kontak, dan mengejar SOTA." });
  say("Ringkasan hasil",
    "Sebelum ke keterbatasan, ini seluruh hasil dalam satu tabel.",
    `Pertama, poligon yang rapi memang bisa dibuat dari *mask* saja: regularisasi menaikkan C-IoU dari ${n("NCiouBfour")} ke ${n("NCiouBfive")}.`,
    `Kedua, tentang tiga gagasan saya. *Head* θ didukung, dengan efek kecil: di *tile* padat PoLiS ${n("NNoneTotalPolis")} piksel dan MTA ${n("NNoneTotalMta")} derajat. Prior morfologi tidak didukung: *merge rate* malah berubah ${n("NNthreeDeltaMerge")} poin.`,
    `Ketiga, model terbaik ternyata yang paling sederhana, B3m, dengan AP ${n("NApBthreem")} dan C-IoU ${n("NCiouBthreem")}. Cabang θ di atas B3 yang dibekukan masih memperbaikinya sedikit di *tile* padat. Untuk memisahkan rumah, benih dari detektor menurunkan *merge rate* dari ${n("NMergeFull")} ke ${n("NMergeFullY")} persen.`,
    `Keempat, dibanding metode terbaik di data yang bersih, model saya masih di bawah HiSup: AP ${n("NCleanApFull")} lawan ${n("NCleanApHisup")}. Di dataset kedua, WHU, AP-nya ${n("NWhuAp")}, dan alur penuh berjalan ${n("NFpsFull")} *tile* per detik.`,
    "Jadi yang berhasil adalah regularisasi dan pemanduan θ. Yang tidak berhasil adalah prior morfologi, kelas kontak, dan upaya mengejar metode terbaik.");

  s = d.slide("Keterbatasan", "Hal yang membatasi kesimpulan penelitian ini.");
  d.numbered(s, [
    { head: "Data bersih kecil", body: `Banding HiSup hanya di ${n("NTilesClean")} *tile*, dan bukan sampel acak.` },
    { head: "WHU satu *seed*", body: "Daftar *tile* uji tidak persis sama dengan Pix2Poly." },
    { head: "Efek N1 kecil", body: "Di jalur biner baru terlihat setelah *mask* dibekukan, dan percobaan itu eksploratif." },
    { head: "Satu detektor", body: "YOLO dilatih sekali dan dipakai untuk ketiga *seed*." },
    { head: "Tanpa lubang", body: "Poligon hanya memuat cincin luar bangunan." },
  ], { labelW: 2.9 });
  say("Keterbatasan",
    "Ada lima hal yang membatasi kesimpulan penelitian ini.",
    `Pertama, data bersih untuk banding dengan HiSup hanya ${n("NTilesClean")} *tile*, dan itu bukan sampel acak dari data validasi.`,
    "Kedua, di WHU hanya ada satu *seed*, dan daftar *tile* ujinya tidak persis sama dengan yang dipakai Pix2Poly.",
    "Ketiga, efek N1 kecil. Di jalur biner, perbaikannya baru terlihat setelah *mask* dibekukan, dan percobaan itu sifatnya eksploratif.",
    "Keempat, detektor YOLO hanya dilatih sekali dan dipakai untuk ketiga *seed*, jadi ragamnya antar-*seed* tidak terukur.",
    "Kelima, poligon yang dihasilkan hanya cincin luar bangunan. Bangunan yang punya halaman dalam, atau lubang, tidak terwakili.");

  s = d.slide("Kesimpulan", "Empat hal yang bisa dibawa pulang.");
  d.numbered(s, [
    { head: "Poligon rapi bisa dari *mask* saja", body: "Regularisasi geometris bekerja tanpa anotasi poligon." },
    { head: "*Head* θ membantu sedikit", body: "Konsisten di jalur tiga kelas; di jalur biner hanya bila *mask* dibekukan (eksploratif)." },
    { head: "Prior dan kelas kontak gagal", body: "Dilaporkan sebagai hasil negatif." },
    { head: "*Baseline* sederhana paling kuat", body: "Terlihat karena semua model dinilai dengan protokol yang sama." },
  ], { shape: "square", labelW: 4.7 });
  say("Kesimpulan",
    "Ada empat hal yang bisa dibawa pulang.",
    "Pertama, poligon yang rapi bisa dihasilkan dari *mask* saja. Regularisasi geometris bekerja tanpa anotasi poligon.",
    "Kedua, *head* orientasi membantu sedikit. Efeknya konsisten di jalur tiga kelas. Di jalur biner, efek itu hanya muncul bila *mask* dibekukan, dan temuan itu masih eksploratif.",
    "Ketiga, prior morfologi dan kelas kontak tidak bekerja seperti yang diharapkan, dan keduanya dilaporkan sebagai hasil negatif.",
    "Keempat, *baseline* yang sederhana ternyata paling kuat. Hal ini baru terlihat karena semua model dinilai dengan protokol yang sama, di data yang bebas kebocoran.",
    "Kalau dirangkum dalam satu kalimat: sumbangan utama proyek ini bukan model yang mengalahkan metode terbaik, melainkan evaluasi yang jujur tentang seberapa jauh pendekatan berbasis *mask* bisa melangkah.");

  s = d.slide("Referensi");
  d.references(s, [
    "Girard, N., dkk. (2021). Polygonal building extraction by frame field learning. *CVPR*.",
    "Xu, B., dkk. (2023). HiSup: Accurate polygonal mapping of buildings in satellite imagery. *ISPRS JPRS*, 198, 284–296.",
    "Adimoolam, Y. K., dkk. (2025). Pix2Poly: A sequence prediction method for polygonal building footprint extraction. *WACV*.",
    "Adimoolam, Y. K., dkk. (2023). Data leakage detection and de-duplication in large scale geospatial image datasets. arXiv:2304.02296.",
    "Zorzi, S., dkk. (2022). PolyWorld: Polygonal building extraction with graph neural networks in satellite images. *CVPR*.",
    "Jiao, W., dkk. (2024). PolyR-CNN: R-CNN for end-to-end polygonal building outline extraction. *ISPRS JPRS*, 218, 33–43.",
    "Jiao, W., dkk. (2025). RoIPoly: Vectorized building outline extraction using vertex and logit embeddings. *ISPRS JPRS*, 224.",
    "Zhang, Y., dkk. (2026). PolyBuild: An end-to-end method for polygonal building contour extraction. arXiv:2606.08920.",
    "Iglovikov, V. I., dkk. (2018). TernausNetV2: Fully convolutional network for instance segmentation. *CVPRW*.",
    "Huang, X., & Zhang, L. (2011). A multidirectional and multiscale morphological index for automatic building extraction. *PE&RS*, 77(7), 721–732.",
    "Zhou, Z., dkk. (2018). UNet++: A nested U-Net architecture for medical image segmentation. *DLMIA*.",
  ], { size: 12 });
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

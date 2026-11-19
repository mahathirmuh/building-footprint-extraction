# Naskah presentasi: Ekstraksi Poligon Bangunan pada Permukiman Padat

Naskah ucapan untuk tiap slide `densepoly.pptx` (32 slide), untuk dosen dan rekan mahasiswa. Lama bicara kira-kira
36 menit pada 125 kata per menit; perkiraan per slide ada di tiap judul.

- Naskah ini dibuat `slides/build.js` bersama deck, jadi angkanya sama dengan yang tampil di slide. Untuk mengubahnya,
  ubah `say(...)` di `build.js` lalu bangun ulang; jangan menyunting file ini langsung.
- Istilah Inggris dicetak miring. Angka di dalam kurung siku adalah selang kepercayaan 95%.
- **Versi singkat, sekitar 14 menit:** hanya slide bertanda (inti), yaitu slide 1, 2, 6, 7, 13, 16, 18, 20, 21, 24, 27, 30, 32.
  Slide lain adalah rincian dan cadangan untuk tanya jawab.

## Slide 1: Sampul

*±0,5 menit · inti*

Selamat pagi, Bapak/Ibu dosen dan teman-teman. Saya Mahathir Muhammad, NRP 6025252008. Saya akan mempresentasikan *final project* mata kuliah Pengolahan Citra Digital, berjudul Ekstraksi Poligon Bangunan pada Permukiman Padat.

Singkatnya: dari foto udara, saya ingin komputer menggambar denah atap setiap rumah sebagai poligon yang rapi, termasuk di kawasan yang rumahnya saling berdempetan.

## Slide 2: Inti presentasi dalam tiga pertanyaan

*±1,5 menit · inti*

Sebelum masuk ke rincian, saya rangkum dulu seluruh presentasi ini dalam tiga pertanyaan.

Pertama, masalahnya apa? Peta membutuhkan satu poligon yang rapi untuk tiap rumah. Di permukiman padat itu sulit, karena atap yang berdempetan terbaca sebagai satu gumpalan. Selain itu, metode terbaik saat ini membutuhkan anotasi poligon, dan data uji CrowdAI yang biasa dipakai ternyata bocor ke data latih.

Kedua, ketemunya apa? Dari *mask* saja, langkah regularisasi sudah membuat poligon jauh lebih rapi: C-IoU, ukuran kerapian poligon yang makin tinggi makin baik, naik dari 62,69 ke 73,87. *Head* orientasi menambah kerapian sedikit lagi. Dua gagasan lain, yaitu prior morfologi dan kelas kontak, tidak membantu.

Lalu soal metode terbaik, HiSup. AP-nya, yang juga makin tinggi makin baik, tercatat 79,43 di data uji yang bocor. Di data yang bersih ternyata hanya 65,60. Modelnya tidak berubah; yang berubah hanya soal ujiannya, yang kali ini belum pernah dilihat model. Jadi angka yang tinggi tadi sebagian berasal dari kebocoran.

Ketiga, kenapa itu penting? Banyak daerah hanya punya data *mask*, jadi poligon yang rapi tetap bisa dibuat tanpa anotasi poligon. Dan angka *benchmark* yang bocor ternyata terlalu optimistis, sehingga metode seharusnya dibandingkan di data yang bersih.

Slide-slide berikutnya menguraikan ketiga jawaban itu satu per satu.

## Slide 3: Mengapa poligon, dan mengapa permukiman padat?

*±1 menit*

Saya mulai dari masalahnya, yang punya tiga sisi. Pertama, peta dan sistem informasi geografis menyimpan bangunan sebagai poligon, yaitu beberapa titik sudut yang dihubungkan dinding lurus. Keluaran segmentasi citra berbeda: bentuknya *mask*, kumpulan piksel yang tepinya bergerigi. Kalau *mask* langsung dijadikan poligon, titik sudutnya bisa ratusan dan dindingnya tidak lurus.

Kedua, di permukiman padat atap rumah saling menempel, sehingga beberapa rumah sering terbaca sebagai satu gumpalan besar. Untuk pemetaan itu masalah, karena satu poligon gabungan menyembunyikan banyak rumah.

Ketiga, metode terbaru seperti HiSup dan Pix2Poly dilatih dengan anotasi poligon. Padahal banyak dataset, terutama data daerah, hanya punya *mask* per bangunan.

Jadi pertanyaan penelitian saya: seberapa baik poligon yang bisa dihasilkan kalau model hanya belajar dari *mask*, dan kalau hasilnya dinilai di data yang bebas kebocoran?

## Slide 4: *State of the art* 2022–2026: ekstraksi poligon bangunan

*±1,5 menit*

Sebelum masuk ke gagasan saya, ini peta metode terkini untuk ekstraksi poligon bangunan, dalam lima tahun terakhir.

Semua metode di tabel ini langsung memprediksi titik sudut atau kontur poligon. PolyWorld, tahun 2022, mendeteksi titik sudut lalu menghubungkannya dengan *graph neural network*. HiSup, tahun 2023, mengawasi titik sudut, garis batas, dan *mask* sekaligus. PolyR-CNN, tahun 2024, dan Pix2Poly, tahun 2025, memprediksi poligon secara *end-to-end*. RoIPoly, juga 2025, menjadikan tiap titik sudut sebuah *query* pada *transformer*. Yang terbaru, PolyBuild tahun 2026, membuat kontur awal dari kotak deteksi lalu menyempurnakannya.

Angka tertinggi dilaporkan RoIPoly, yaitu AP 83,6; HiSup, PolyR-CNN, dan Pix2Poly berada di sekitar 79,2 sampai 79,6. Menariknya, metode yang lebih baru tidak selalu lebih tinggi: PolyBuild melaporkan AP 60,6. Pendekatan saya berbeda keluarga: segmentasi dulu, lalu dirapikan, seperti *frame field learning* dari tahun 2021 yang menjadi pendahulunya.

Ada dua catatan penting. Pertama, metode berbasis titik sudut itu dilatih dengan anotasi poligon, sedangkan penelitian ini hanya memakai *mask*. Kedua, semua angka paper tadi diukur di data validasi CrowdAI yang bocor.

Baris terakhir adalah model saya di data *test* yang bersih: AP 61,94 dan C-IoU 74,55. Karena datanya berbeda, baris itu tidak bisa dibandingkan langsung dengan baris di atasnya. Perbandingan yang adil saya tunjukkan nanti, di slide HiSup.

## Slide 5: Posisi penelitian ini di antara metode lain

*±1,5 menit*

Tabel ini menempatkan penelitian saya di antara metode lain. Tiap kolom adalah satu ciri; Ya berarti karya itu memilikinya.

Kolom pertama: dilatih hanya dari *mask*. Semua metode pembanding membutuhkan anotasi poligon; penelitian ini tidak.

Kolom kedua: orientasi bangunan dipelajari jaringan. Hanya *frame field learning* tahun 2021 yang punya gagasan serupa, lewat medan arah; di penelitian ini orientasi dipelajari dari label yang diturunkan dari *mask*.

Kolom ketiga dan keempat menyangkut evaluasi. Angka CrowdAI di paper-paper itu diukur di data validasi yang bocor, dan tidak dipecah menurut kepadatan bangunan. Penelitian ini menilai di data bebas duplikat dan melaporkan hasil per tingkat kepadatan.

Kolom terakhir sengaja saya masukkan supaya adil, karena di situ penelitian ini yang kurang. Metode terbaru bersifat *end-to-end*: poligon keluar langsung dari jaringan. Metode saya masih membutuhkan *post-processing*, dan AP-nya lebih rendah.

Jadi posisinya begini: belum ada karya yang menggabungkan empat ciri pertama, dan itulah celah yang diisi penelitian ini. Yang ditawarkan bukan akurasi tertinggi, melainkan jalur yang hanya butuh *mask*, dengan evaluasi yang lebih jujur.

## Slide 6: Kebaruan penelitian: tiga gagasan yang diuji

*±1,5 menit · inti*

Ini kebaruan penelitian saya: tiga gagasan, dan ketiganya saya uji.

N1 adalah kebaruan di sisi metode: jaringan diberi satu keluaran tambahan, disebut *head* orientasi, yang menebak arah hadap bangunan, yaitu sudut θ. Yang baru, label sudutnya dihitung otomatis dari *mask*, lalu sudut itu dipakai untuk merapikan poligon. Hasil ujinya: didukung, tetapi efeknya kecil.

N2 adalah kebaruan di sisi evaluasi: dataset CrowdAI dibersihkan dari duplikat, lalu hasil dilaporkan per tingkat kepadatan bangunan. Dari sini muncul temuan terkuat penelitian ini, yaitu bahwa angka *benchmark* yang bocor terlalu tinggi.

N3 adalah kebaruan di sisi *preprocessing*: dua peta dari morfologi klasik, yaitu MBI dan peta celah, ditambahkan sebagai kanal input, dengan harapan celah antar-rumah lebih mudah terlihat oleh jaringan. Hasil ujinya: tidak didukung, dan saya laporkan sebagai hasil negatif.

Supaya jelas batasnya, ada komponen yang saya adopsi dari penelitian lain dan tidak saya klaim sebagai kebaruan: kelas kontak dengan *watershed*, bobot tepi pada *loss*, arsitektur U-Net++, indeks MBI itu sendiri, dan deduplikasi dengan *perceptual hash*.

Satu hal lagi yang ingin saya tekankan: kriteria berhasil atau gagal untuk tiap gagasan saya tulis dan saya bekukan sebelum ada model yang dilatih. Setiap perbandingan memakai tiga *seed* dan selang kepercayaan *bootstrap*. Karena itulah hasil negatif pun tetap dilaporkan.

## Slide 7: Metode yang diusulkan

*±1,5 menit · inti*

Inilah metode yang saya usulkan, dalam dua baris: jaringannya di atas, pembentukan poligonnya di bawah.

Jaringannya U-Net++ dengan *encoder* EfficientNet-B4. Inputnya citra RGB. Dalam rancangan awal ada dua kanal tambahan berisi prior morfologi, yaitu gagasan N3; itu diuji, tetapi karena tidak membantu, model akhirnya hanya memakai RGB.

Yang membedakan jaringan ini adalah keluarannya. Selain tiga kelas, yaitu latar, bangunan, dan kontak, ada *head* orientasi yang memprediksi sudut bangunan θ dalam bentuk sinus dan kosinus 4θ. Inilah gagasan N1. Label θ tidak dianotasi manusia: ia dihitung dari kotak terkecil tiap bangunan di *mask*.

Di baris bawah, peta kelas dipakai membuat *marker*, lalu *watershed* memisahkan rumah. Kontur tiap rumah diambil, lalu dirapikan dengan panduan θ dari jaringan: bangunan yang hampir persegi diganti kotak searah θ, dan bangunan lain disederhanakan lalu sisi-sisinya diluruskan ke arah θ. Hasilnya satu poligon per rumah.

Jaringan dilatih dengan satu *loss* gabungan: *cross-entropy* berbobot tepi, Dice, dan *loss* orientasi. Semua labelnya diturunkan dari *mask*, jadi tidak ada anotasi poligon saat *training*.

## Slide 8: Dataset: apa dan dari mana

*±1,5 menit*

Penelitian ini memakai dua dataset publik, dan keduanya saya unduh dari halaman resminya.

Yang utama adalah CrowdAI Mapping Challenge, dari platform AIcrowd, versi 1.0. Isinya citra satelit RGB berupa *tile* 300 × 300 piksel: 280.741 *tile* train dan 60.317 *tile* validasi, dengan anotasi poligon per bangunan dalam format COCO. Versi 1.0 saya pilih karena metode pembanding juga memakai versi itu.

Seperti sudah saya singgung, dataset ini banyak duplikatnya. Setelah deduplikasi, yang dipakai adalah 23.427 *tile* train, 1.000 dan 1.928 *tile* untuk tuning, serta 2.928 *tile* *test*. Semua klaim dan perbandingan dengan HiSup memakai dataset ini.

Dataset kedua adalah WHU Building, dari grup GPCV Universitas Wuhan, yang diperkenalkan Ji dan rekan-rekannya tahun 2019. Ini foto udara dengan resolusi 0,2 meter per piksel, dalam *tile* 1.024 × 1.024 piksel: 2.943 train, 627 validasi, dan 2.220 *test*. Mengikuti protokol Pix2Poly, *tile* itu dipotong menjadi *patch* 224 × 224. WHU saya pakai sebagai dataset kedua, dengan satu *seed*, untuk melihat apakah hasilnya terbawa ke data lain.

Satu hal yang perlu jelas: kedua dataset sebenarnya menyediakan anotasi per bangunan, dan di CrowdAI bentuknya poligon. Tetapi untuk *training* saya hanya memakainya sebagai *mask*. Poligon aslinya hanya dipakai sebagai *ground truth* saat evaluasi. Itulah arti kalimat tanpa anotasi poligon di penelitian ini.

## Slide 9: Arsitektur: alur pengerjaan dari awal sampai akhir

*±1,5 menit*

Ini arsitektur keseluruhannya, dalam tiga tahap. Tiap baris dibaca dari kiri ke kanan.

Baris pertama adalah data dan label. Datanya CrowdAI dan WHU, berupa citra RGB dan *mask* per bangunan, tanpa anotasi poligon. Data itu dideduplikasi dengan *perceptual hash*, lalu dibagi menjadi train, tuning, dan *test* per klaster, supaya tidak ada citra kembar di dua bagian. Dari *mask* kemudian dibuat tiga label secara otomatis: kelas kontak, orientasi θ, dan bobot tepi.

Baris kedua adalah jaringannya. Inputnya satu *tile* 300 × 300 piksel, yang hanya di-*pad* ke 320 dan tidak diubah ukurannya. Jaringannya U-Net++ dengan *encoder* EfficientNet-B4. Keluarannya lima kanal: tiga kelas, yaitu latar, bangunan, dan kontak, ditambah dua kanal orientasi. Saat *training*, keluaran itu dibandingkan dengan label dari baris pertama lewat *loss* gabungan; itulah arti panah turun di sebelah kanan.

Baris ketiga mengubah keluaran jaringan menjadi poligon. Dari peta kelas dibuat *marker*, yaitu inti tiap rumah, lalu *watershed* memekarkannya sehingga satu gumpalan terpecah menjadi satu *instance* per rumah. Kontur tiap rumah diambil, lalu dirapikan searah θ: bangunan yang hampir persegi menjadi kotak, yang lain disederhanakan dan dindingnya diluruskan. Hasilnya satu poligon per rumah, dalam format COCO JSON dan GeoJSON.

Kotak terakhir adalah evaluasi: poligon dinilai dengan AP, C-IoU, PoLiS, dan MTA, dan hasilnya dilaporkan per tingkat kepadatan. Jadi dari awal sampai akhir, tidak ada anotasi poligon yang dipakai.

## Slide 10: Pengolahan citra klasik di dalam alur

*±1,5 menit*

Karena ini mata kuliah Pengolahan Citra Digital, saya tunjukkan di mana saja algoritma klasik dipakai. Yang saya maksud klasik adalah algoritma tetap, tanpa parameter yang dilatih.

Lima baris pertama ada di alur utama. Untuk membuat label: dilasi menghasilkan kelas kontak, *distance transform* menghasilkan bobot tepi, dan *minAreaRect* menghasilkan orientasi. Setelah jaringan: *thresholding* dan *connected component labeling* membuat *marker*, *watershed* memisahkan rumah, *marching squares* mengambil kontur, lalu momen orde dua dan algoritma Douglas-Peucker merapikannya menjadi kotak atau poligon siku.

Jadi jaringan hanya mengerjakan bagian tengah, yaitu dari citra ke peta kelas dan orientasi. Yang mengubah peta itu menjadi satu poligon per rumah seluruhnya algoritma klasik. Untuk tepatnya, *minAreaRect* dan Douglas-Peucker bekerja pada titik dan garis, jadi lebih pas disebut geometri komputasi.

Baris keenam adalah *baseline* B0, yang seluruhnya klasik tanpa jaringan: indeks bangunan MBI, ambang Otsu, *opening*, lalu *watershed*. AP-nya hanya 0,08, jadi cara klasik saja tidak cukup untuk data ini.

Baris terakhir adalah gagasan N3: peta dari *top-hat* multi-arah dijadikan input tambahan bagi jaringan. Hasilnya tidak membantu, dan itu saya bahas di slide hasil negatif.

## Slide 11: Label diturunkan otomatis dari *mask*

*±1 menit*

Karena tidak ada anotasi poligon, semua label dibuat otomatis dari peta *instance*, yaitu *mask* yang tiap bangunannya punya nomor sendiri.

Label pertama adalah kelas kontak: piksel bangunan yang jaraknya paling jauh dua piksel dari bangunan lain. Tujuannya menandai tempat dua atap bertemu.

Label kedua adalah orientasi. Untuk tiap bangunan dicari kotak terkecil yang membungkusnya, dengan fungsi *minAreaRect* di OpenCV, lalu diambil arah sisinya. Sudut itu dikodekan sebagai sinus dan kosinus dari 4θ. Alasannya, bangunan persegi terlihat sama bila diputar 90 derajat, jadi keempat arah sisinya harus dianggap satu arah yang sama.

Yang ketiga adalah bobot tepi: piksel di dekat batas bangunan diberi bobot *loss* lebih besar, supaya jaringan tidak mengabaikan celah sempit di antara rumah.

Kelas kontak dan bobot tepi saya adopsi dari penelitian sebelumnya. Yang baru di sini adalah label orientasi yang diturunkan dari *mask*.

## Slide 12: Contoh label dan prior pada citra asli

*±1 menit*

Supaya label tadi tidak abstrak, ini contohnya pada dua *tile* asli: satu *tile* padat dan satu *tile* yang bangunannya bersentuhan.

Kolom pertama adalah citranya. Kolom kedua adalah kelas: abu-abu untuk bangunan, dan garis merah tipis untuk kontak, yaitu tempat dua bangunan bertemu. Di baris bawah garis merah itu terlihat di sisi kanan. Dari sini kelihatan betapa sedikitnya piksel kontak.

Kolom ketiga adalah orientasi θ, dengan warna sebagai sudut: bangunan yang searah berwarna sama. Kolom keempat adalah bobot tepi, yang menyala di sekitar batas bangunan.

Dua kolom terakhir adalah hasil pengolahan citra klasik. MBI menonjolkan struktur terang seukuran rumah, dan peta celah menonjolkan garis gelap tipis di antara atap. Keduanya dipakai sebagai input tambahan di gagasan N3.

## Slide 13: Evaluasi tanpa kebocoran data

*±1 menit · inti*

Bagian ini menurut saya yang paling penting. Dataset CrowdAI punya masalah yang sudah dilaporkan peneliti lain: banyak citra validasi yang sama persis dengan citra train, atau hanya diputar dan dicerminkan. Model seolah diuji dengan soal yang sudah pernah dilihatnya, sehingga angkanya terlalu optimistis.

Karena itu saya mendeduplikasi dengan *perceptual hash*. Dari 280.741 *tile* train tersisa 29.283 klaster, dan dari tiap klaster diambil satu wakil.

Klaster itu lalu dibagi: 23.427 *tile* untuk train, 1.000 untuk menyetel *post-processing*, 1.928 untuk memilih model, dan 2.928 *tile* *test* yang disisihkan.

*Tile* *test* dibagi menjadi tiga kelompok kepadatan, yaitu jarang, sedang, dan padat, menurut jumlah bangunannya. Pembagian ini dibekukan dari *ground truth* sebelum ada model yang dievaluasi.

Setiap selisih antarmodel dihitung berpasangan pada *tile* yang sama, dengan tiga *seed* dan *bootstrap* seribu replikasi. Dan semua parameter disetel di data tuning; data *test* tidak pernah dipakai untuk menyetel apa pun.

## Slide 14: Protokol uji yang adil

*±1,5 menit*

Supaya perbandingan antarmodel adil, semua model jaringan dilatih dan dinilai dengan setelan yang sama. Tabel di kiri merangkumnya.

Datanya sama: *split* bebas duplikat yang sama, *tile* 300 × 300 yang hanya di-*pad* ke 320 tanpa diubah ukurannya, dan augmentasi yang sama. Optimizer-nya sama: AdamW dengan *learning rate* 2e-4, *warmup* satu *epoch* lalu turun mengikuti kosinus. Lama latihannya sama: *batch* 16 dan 60 *epoch* tanpa *early stopping*, dengan *seed* 0, 1, dan 2.

Pemilihannya juga sama: *checkpoint* terbaik dipilih di data validasi, bukan di data *test*, dan *post-processing* tiap model disetel dengan grid yang sama lalu dibekukan. Evaluasinya memakai 2.928 *tile* *test* yang sama, kode metrik yang sama, dan selisih berpasangan per *tile*. Semua jaringan berangkat dari bobot ImageNet yang sama.

Yang berbeda hanya tiga hal, di panel kanan. Pertama, komponen yang memang sedang diuji di tiap langkah ablasi. Kedua, B0 tidak dilatih karena murni klasik, dan B1 serta B2 hanya satu *seed*. Ketiga, B3+θ memakai jaringan B3 yang dibekukan dan *checkpoint* *epoch* terakhir, karena *mask*-nya tidak berubah sehingga tidak ada yang bisa dipilih.

Kesamaan setelan ini saya periksa dari berkas konfigurasi yang tersimpan di tiap *run*, bukan dari ingatan.

## Slide 15: Ablasi: satu perubahan per langkah

*±1 menit*

Ini tabel ablasinya. Tiap baris menambah satu komponen ke baris di atasnya, jadi pengaruh tiap komponen bisa dibaca sendiri-sendiri.

Baris pertama, B0, adalah cara klasik saja tanpa jaringan: AP-nya hanya 0,08. Begitu jaringan dipakai, di B1, AP langsung menjadi 63,54. Mengganti U-Net dengan U-Net++, di B2, hampir tidak mengubah apa-apa. Menambah bobot tepi pada *loss*, di B3, menaikkan AP menjadi 65,74.

Di B4, kelas kontak dan *watershed* ditambahkan. *Merge rate* turun dari 2,84 menjadi 2,07 persen, artinya lebih sedikit rumah yang tergabung. Tetapi AP ikut turun menjadi 62,91. Inilah harga dari jalur tiga kelas.

Lompatan terbesar ada di B5, saat regularisasi masuk: C-IoU naik dari 62,69 ke 73,87, dan galat sudut MTA turun dari 47,11 ke 37,30 derajat.

Dua baris terakhir adalah gagasan saya. *Head* θ, di FULL-P0, menambah sedikit kerapian: C-IoU 74,55 dan MTA 36,34. Prior morfologi, di FULL-P2, tidak membawa perbaikan. Dua hal itu saya uji lebih teliti di slide-slide berikutnya.

## Slide 16: Hasil utama di CrowdAI *test*

*±1,5 menit · inti*

Tabel ini merangkum hasil di data *test*, rata-rata tiga *seed*. Sebelum membacanya, sedikit tentang metriknya. AP mengukur ketepatan deteksi tiap bangunan; makin besar makin baik. C-IoU adalah IoU yang dikenai penalti bila jumlah titik sudut poligon jauh dari *ground truth*, jadi ia mengukur kerapian. MTA adalah galat sudut dinding dalam derajat. *Merge rate* adalah persentase rumah yang tergabung dengan tetangganya. Untuk MTA dan *merge rate*, makin kecil makin baik.

Ada tiga hal yang bisa dibaca. Pertama, regularisasi sangat menaikkan kerapian: dari B4 ke B5, C-IoU naik dari 62,69 ke 73,87, dan MTA turun dari 47,11 ke 37,30 derajat.

Kedua, jalur biner unggul di AP: B3m 64,89, sedangkan FULL-P0 61,94.

Ketiga, jalur tiga kelas lebih baik memisahkan rumah: *merge rate* FULL-P0 1,96%, B3m 2,81%. Dengan bantuan detektor, FULL+Y turun lagi ke 1,52%.

Jadi tidak ada satu model yang menang di semua kolom. Itu saya uraikan satu per satu. Baris B3+θ adalah percobaan eksploratif yang saya jelaskan di slide tersendiri.

## Slide 17: Perbandingan model per tingkat kepadatan

*±1 menit*

Gambar ini membandingkan tiga model, yaitu B3, B4, dan FULL-P0, menurut kepadatan *tile*: jarang, sedang, dan padat.

Panel kiri adalah AP. B3, garis biru, berada di atas di semua kepadatan. Panel kanan adalah PoLiS, yang makin kecil makin baik; di sana B3 juga yang terbaik, dan FULL-P0 berada di bawah B4, artinya *head* θ memang mendekatkan batas poligon ke *ground truth*.

Panel tengah adalah *merge rate*, dan di sinilah urutannya terbalik: B3 paling banyak menggabungkan rumah, terutama di *tile* padat. Jalur tiga kelas, B4 dan FULL-P0, lebih baik memisahkan rumah.

Jadi kepadatan tidak mengubah siapa yang unggul di tiap metrik, tetapi memperlebar jarak di *merge rate*: makin padat, makin terasa perlunya mekanisme pemisah rumah.

## Slide 18: N1 didukung: *head* θ merapikan poligon

*±1 menit · inti*

Sekarang gagasan pertama. Yang dibandingkan adalah B5 dan FULL-P0, di 857 *tile* terpadat. Di B5, sudut perapian diambil dari kotak terkecil kontur hasil prediksi. Di FULL-P0, jaringan dilatih dengan *head* θ dan sudutnya dibaca dari *head* itu.

PoLiS adalah jarak rata-rata antara batas poligon prediksi dan batas *ground truth*. Nilainya turun 0,037 piksel. MTA, galat sudut dinding, turun 0,94 derajat. Selang kepercayaan 95% keduanya tidak memuat nol, dan arahnya sama di ketiga *seed*.

AP berubah +0,35 poin, jadi kerapian itu tidak dibayar dengan akurasi deteksi.

Menurut aturan yang sudah dibekukan, N1 dinyatakan didukung. Tetapi saya harus jujur soal besarnya: efeknya kecil, jauh di bawah satu piksel. Jadi kesimpulannya, *head* θ membantu secara konsisten, bukan membantu secara besar.

## Slide 19: Efek N1 membesar di *tile* padat

*±1 menit*

Gambar ini memecah efek tadi menurut kepadatan. Tiap titik adalah selisih antara dua model, dan garis tegaknya selang kepercayaan 95%. Titik di bawah garis putus-putus berarti ada perbaikan.

Di dua panel kiri, untuk *head* θ, perbaikannya membesar dari *tile* jarang ke padat. PoLiS: −0,012, −0,021, lalu −0,037 piksel. MTA: −0,42, −0,73, lalu −0,94 derajat.

Dua titik di sebelah kanan tiap panel membagi *tile* dengan cara lain: yang berisi bangunan bersentuhan dan yang tidak. Selang kelompok bersentuhan lebar karena *tile*-nya hanya 146 dari 2.928.

Kembali ke kepadatan: kecenderungan belum tentu bukti. Ketika selisih padat dikurangi jarang diuji, hanya MTA yang signifikan, yaitu −0,51 derajat. Untuk PoLiS selang kepercayaannya memuat nol. Karena itu klaim bahwa efek N1 paling besar di area padat tidak saya pakai.

Panel kanan adalah prior morfologi, gagasan N3. Di sana titik-titiknya berada di atas nol, artinya rumah yang tergabung tidak berkurang. Itu saya bahas di slide berikutnya.

## Slide 20: Dua hasil negatif, dilaporkan apa adanya

*±1 menit · inti*

Dua bagian yang dirancang untuk memisahkan rumah berdempetan ternyata tidak bekerja seperti yang saya harapkan.

Yang pertama N3. Dengan MBI dan peta celah sebagai input tambahan, *merge rate* berubah +0,15 poin, dengan selang [+0,02; +0,30]. Artinya rumah yang tergabung sedikit bertambah, bukan berkurang, dan arahnya pun tidak sama di ketiga *seed*. Jadi hipotesis bahwa prior celah membantu tidak didukung data.

Yang kedua kelas kontak. Di 6 dari 9 *run* FULL, jaringan tidak pernah memilih kelas kontak sama sekali. Penjelasan yang paling mungkin: piksel kontak sangat langka dibanding piksel bangunan, sehingga jaringan lebih untung mengabaikannya. Rumah-rumah tetap terpisah, tetapi karena ada celah latar di antaranya, bukan karena kelas kontak.

Keduanya saya laporkan apa adanya, karena hasil negatif yang diukur dengan benar tetap berguna bagi yang ingin mencoba gagasan serupa.

## Slide 21: Banding dengan HiSup di data tanpa kebocoran

*±1,5 menit · inti*

Bagaimana posisi hasil ini dibanding metode terbaik? Pembandingnya HiSup, metode ekstraksi poligon bangunan dari Xu dan rekan-rekannya tahun 2023, yang dilatih dengan anotasi poligon dan termasuk yang terbaik di CrowdAI. HiSup merilis file prediksinya, jadi bisa saya nilai ulang dengan kode evaluasi yang sama, tanpa menjalankan modelnya.

Di data validasi penuh, yang bocor, HiSup mendapat AP 79,43. Angka itu cocok dengan yang dilaporkan di papernya, dan itu sekaligus memeriksa bahwa kode evaluasi saya benar.

Tetapi di 212 *tile* validasi yang benar-benar tidak punya kembaran di train, AP HiSup turun menjadi 65,60. Di *tile* yang sama, FULL-P0 mendapat 61,14, tertinggal 4,45 poin. Perlu diingat, HiSup dilatih dengan anotasi poligon, sedangkan model saya tidak.

Ambang "mendekati metode terbaik" yang saya bekukan sejak awal adalah tiga poin AP. Selisihnya lebih besar dari itu, jadi klaim tersebut tidak saya pakai.

Temuan yang menurut saya lebih penting di slide ini adalah baris pertama dan kedua: angka HiSup sendiri turun jauh begitu duplikat disingkirkan. Artinya angka di *benchmark* CrowdAI perlu dibaca dengan hati-hati.

## Slide 22: Eksploratif: *baseline* sederhana dan *seed* detektor

*±1 menit*

Dua percobaan di slide ini saya tambahkan setelah hasil utama ada. Karena itu sifatnya eksploratif dan bukan bagian dari klaim.

Yang pertama B3m: segmentasi biner biasa, ditambah regularisasi yang sama. Ternyata ini konfigurasi yang paling kuat. AP-nya 64,89 dan C-IoU-nya 76,97, sedangkan FULL-P0 61,94 dan 74,55. B3m unggul di semua metrik kecuali satu, yaitu rumah yang tergabung.

Yang kedua, sebuah detektor kecil, YOLOv8n-OBB, yang juga dilatih dari *mask*, dipakai untuk memberi titik benih bagi tiap rumah. Hasilnya *merge rate* turun 0,45 poin di FULL-P0 dan 0,67 poin di B3m, dengan AP yang setara. Ada harganya: rumah yang terpecah bertambah 0,24 dan 0,36 poin.

Jadi gambaran besarnya: jalur biner paling akurat dan paling rapi, jalur tiga kelas paling baik memisahkan rumah berdempetan, dan benih dari detektor membantu keduanya.

## Slide 23: Eksploratif: *head* θ di jalur biner

*±1,5 menit*

Karena B3m paling kuat, pertanyaan yang wajar: bagaimana kalau *head* θ dipasang di jalur biner itu? Saya mencoba dua cara. Keduanya eksploratif, tetapi aturan penilaiannya tetap saya bekukan sebelum *training*.

Cara pertama, B3θ: *head* θ dilatih bersama *mask*. Hasilnya negatif. Pemanduannya sendiri membantu, tetapi melatih θ bersama *mask* menurunkan mutu *mask*: IoU validasi turun dari 89,75% menjadi 88,78%. Totalnya, dibanding B3m, PoLiS +0,084 piksel, MTA +0,30 derajat, dan AP −2,26 poin: lebih buruk.

Dari kegagalan itu muncul cara kedua, B3+θ: jaringan B3 dibekukan, lalu hanya sebuah cabang kecil untuk θ yang dilatih di atasnya. Dengan begitu *mask* tidak berubah sama sekali. Hasilnya positif: dibanding B3m, PoLiS −0,011 piksel dan MTA −0,24 derajat, keduanya dengan selang kepercayaan di bawah nol dan searah di ketiga *seed*. AP berubah −0,12 poin, masih di dalam batas yang dibekukan.

Jadi pemanduan θ juga memperbaiki *baseline* terkuat, asalkan *mask*-nya dilindungi. Tetapi saya tidak ingin melebih-lebihkan: efeknya kecil, dan kalau dihitung di semua *tile*, AP turun 0,34 poin. Karena ditambahkan setelah hasil utama ada, temuan ini saya laporkan sebagai eksploratif dan tidak mengubah keputusan N1 sampai N3.

## Slide 24: Sebelum dan sesudah regularisasi

*±1 menit · inti*

Ini gambar sebelum dan sesudahnya, pada dua *tile* padat dari data *test*. Dari kiri: citra aslinya, kontur sebelum dirapikan, poligon sesudah dirapikan, dan *ground truth*.

Kolom kedua dan ketiga berasal dari jaringan yang sama dan *instance* yang sama, yaitu B4 dan B5. Satu-satunya perbedaan adalah langkah regularisasi. Jadi apa pun yang berubah di antara kedua kolom itu murni akibat perapian poligon.

Sebelum dirapikan, tepi bangunan bergelombang mengikuti piksel. Sesudahnya, sisi-sisinya lurus dan sudutnya siku, lebih mirip *ground truth* di kolom kanan.

Angka di pojok kiri bawah adalah jumlah titik sudut di *tile* itu. Jumlahnya turun mendekati *ground truth*, walaupun belum sama. Inilah yang ditangkap metrik C-IoU.

## Slide 25: Contoh hasil di *tile* asli

*±1 menit*

Supaya tidak hanya angka, ini contoh pada *tile* *test*. Kolom paling kiri adalah *ground truth*, digambar hijau. Empat kolom berikutnya adalah poligon dari B4, FULL-P0, FULL+Y, dan B3m, digambar kuning. Angka kecil di pojok kiri bawah tiap gambar adalah jumlah bangunannya.

Di kolom B4, kontur masih mentah: garisnya bergelombang dan titik sudutnya banyak. Di tiga kolom berikutnya, setelah regularisasi, poligon menjadi siku dengan sedikit titik sudut, dan lebih mirip *ground truth*.

Tiga baris pertama adalah *tile* padat yang hasilnya cukup baik. Baris terbawah sengaja saya pilih sebagai contoh kegagalan: di sisi kanan bawah, beberapa rumah yang berdempetan masih tergabung menjadi satu poligon besar, di semua model. Inilah kasus yang belum terselesaikan.

## Slide 26: WHU dan biaya komputasi

*±1 menit*

Untuk melihat apakah hasilnya terbawa ke data lain, model yang sama saya latih di dataset WHU, dengan satu *seed*. AP-nya 66,97 dan C-IoU-nya 72,07.

Kalau dihitung dengan aturan Pix2Poly, C-IoU kita 66,83. Angka yang dilaporkan di paper: HiSup 79,62 dan Pix2Poly 81,63. Perbandingan ini hanya kira-kira, karena daftar *tile* uji Pix2Poly tidak bisa direkonstruksi persis. Meski begitu, jaraknya jelas besar.

Titik lemahnya sama dengan di CrowdAI: di *patch* yang berisi bangunan bersentuhan, 14,73% rumah masih tergabung.

Dari sisi biaya, modelnya 20,81 juta parameter. Alur penuh, dari citra sampai poligon, berjalan 42,3 *tile* per detik di RTX 5090, dan 36,4 bila detektor ikut dipakai.

## Slide 27: Ringkasan hasil

*±1 menit · inti*

Sebelum ke keterbatasan, ini seluruh hasil dalam satu tabel.

Pertama, poligon yang rapi memang bisa dibuat dari *mask* saja: regularisasi menaikkan C-IoU dari 62,69 ke 73,87.

Kedua, tentang tiga gagasan saya. *Head* θ didukung, dengan efek kecil: di *tile* padat PoLiS −0,037 piksel dan MTA −0,94 derajat. Prior morfologi tidak didukung: *merge rate* malah berubah +0,15 poin.

Ketiga, model terbaik ternyata yang paling sederhana, B3m, dengan AP 64,89 dan C-IoU 76,97. Cabang θ di atas B3 yang dibekukan masih memperbaikinya sedikit di *tile* padat. Untuk memisahkan rumah, benih dari detektor menurunkan *merge rate* dari 1,96 ke 1,52 persen.

Keempat, dibanding metode terbaik di data yang bersih, model saya masih di bawah HiSup: AP 61,14 lawan 65,60. Di dataset kedua, WHU, AP-nya 66,97, dan alur penuh berjalan 42,3 *tile* per detik.

Jadi yang berhasil adalah regularisasi dan pemanduan θ. Yang tidak berhasil adalah prior morfologi, kelas kontak, dan upaya mengejar metode terbaik.

## Slide 28: Keterbatasan

*±1 menit*

Ada lima hal yang membatasi kesimpulan penelitian ini.

Pertama, data bersih untuk banding dengan HiSup hanya 212 *tile*, dan itu bukan sampel acak dari data validasi.

Kedua, di WHU hanya ada satu *seed*, dan daftar *tile* ujinya tidak persis sama dengan yang dipakai Pix2Poly.

Ketiga, efek N1 kecil. Di jalur biner, perbaikannya baru terlihat setelah *mask* dibekukan, dan percobaan itu sifatnya eksploratif.

Keempat, detektor YOLO hanya dilatih sekali dan dipakai untuk ketiga *seed*, jadi ragamnya antar-*seed* tidak terukur.

Kelima, poligon yang dihasilkan hanya cincin luar bangunan. Bangunan yang punya halaman dalam, atau lubang, tidak terwakili.

## Slide 29: Langkah ke depan

*±1 menit*

Dari hasil dan keterbatasan tadi, ada lima langkah lanjutan yang menurut saya paling diperlukan.

Pertama, menggabungkan dua jalur. Jalur biner paling akurat, jalur tiga kelas paling baik memisahkan rumah. Kombinasi yang belum saya coba adalah B3+θ dengan benih dari detektor, yang berpotensi mengambil kelebihan keduanya.

Kedua, menghidupkan kelas kontak. Kelas ini hampir tidak pernah dipilih jaringan karena pikselnya terlalu langka, jadi perlu *loss* atau pembobotan yang dirancang khusus untuk itu.

Ketiga, memperbaiki cabang θ. Saya baru mencoba satu konfigurasi, dan kecepatannya belum diukur.

Keempat, evaluasi bersih yang lebih besar. Perbandingan dengan metode terbaik saat ini hanya memakai 212 *tile*. CrowdAI versi 2.0, yang kebocorannya sudah diperbaiki panitia, WHU dengan tiga *seed*, dan uji lintas-dataset akan membuat kesimpulannya lebih kokoh.

Kelima, data dan bentuk lain: menguji metode ini di citra permukiman padat Indonesia, yang menjadi motivasi awalnya, serta menangani bangunan yang berlubang dan yang tidak siku.

## Slide 30: Kesimpulan

*±1 menit · inti*

Ada empat hal yang bisa dibawa pulang.

Pertama, poligon yang rapi bisa dihasilkan dari *mask* saja. Regularisasi geometris bekerja tanpa anotasi poligon.

Kedua, *head* orientasi membantu sedikit. Efeknya konsisten di jalur tiga kelas. Di jalur biner, efek itu hanya muncul bila *mask* dibekukan, dan temuan itu masih eksploratif.

Ketiga, prior morfologi dan kelas kontak tidak bekerja seperti yang diharapkan, dan keduanya dilaporkan sebagai hasil negatif.

Keempat, *baseline* yang sederhana ternyata paling kuat. Hal ini baru terlihat karena semua model dinilai dengan protokol yang sama, di data yang bebas kebocoran.

Kalau dirangkum dalam satu kalimat: sumbangan utama proyek ini bukan model yang mengalahkan metode terbaik, melainkan evaluasi yang jujur tentang seberapa jauh pendekatan berbasis *mask* bisa melangkah.

## Slide 31: Referensi

*±0,5 menit*

Ini rujukan utama yang saya pakai. Daftar lengkapnya ada di naskah paper.

## Slide 32: Penutup

*±0,5 menit · inti*

Demikian presentasi saya. Kode, konfigurasi, dan semua angka di slide ini bisa direproduksi dari repositori proyek. Terima kasih atas perhatiannya. Saya persilakan bila ada pertanyaan atau masukan.

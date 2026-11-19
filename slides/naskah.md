# Naskah presentasi: Ekstraksi Poligon Bangunan pada Permukiman Padat

Naskah ucapan untuk tiap slide `densepoly.pptx` (22 slide), untuk dosen dan rekan mahasiswa. Lama bicara kira-kira
22 menit pada 125 kata per menit; perkiraan per slide ada di tiap judul.

- Naskah ini dibuat `slides/build.js` bersama deck, jadi angkanya sama dengan yang tampil di slide. Untuk mengubahnya,
  ubah `say(...)` di `build.js` lalu bangun ulang; jangan menyunting file ini langsung.
- Istilah Inggris dicetak miring. Angka di dalam kurung siku adalah selang kepercayaan 95%.
- Bila waktu hanya ±10 menit: lewati slide "Eksploratif" dan "WHU dan biaya komputasi", dan sebut isinya dalam satu
  kalimat di slide Kesimpulan.

## Slide 1: Sampul

*±0,5 menit*

Selamat pagi, Bapak/Ibu dosen dan teman-teman. Saya Mahathir Muhammad, NRP 6025252008. Saya akan mempresentasikan *final project* mata kuliah Pengolahan Citra Digital, berjudul Ekstraksi Poligon Bangunan pada Permukiman Padat.

Singkatnya: dari foto udara, saya ingin komputer menggambar denah atap setiap rumah sebagai poligon yang rapi, termasuk di kawasan yang rumahnya saling berdempetan.

## Slide 2: Inti presentasi dalam tiga pertanyaan

*±1,5 menit*

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

## Slide 4: *State of the art*: ekstraksi poligon bangunan

*±1,5 menit*

Sebelum masuk ke gagasan saya, ini peta metode terkini untuk ekstraksi poligon bangunan. Secara garis besar ada dua keluarga.

Keluarga pertama melakukan segmentasi dulu, lalu merapikan hasilnya. Contohnya FFL, *frame field learning*, yang menambah satu keluaran berupa medan arah untuk meluruskan kontur. AP yang dilaporkan 61,3. Pendekatan saya paling dekat dengan keluarga ini.

Keluarga kedua langsung memprediksi titik sudut poligon. PolyWorld mendeteksi titik sudut lalu menghubungkannya dengan *graph neural network*. HiSup mengawasi titik sudut, garis batas, dan *mask* sekaligus. PolyR-CNN dan Pix2Poly memprediksi poligon secara *end-to-end*. Tiga yang terbaru melaporkan AP sekitar 79,2 sampai 79,6, dan itulah yang sekarang dianggap *state of the art*.

Ada dua catatan penting. Pertama, metode berbasis titik sudut itu dilatih dengan anotasi poligon, sedangkan penelitian ini hanya memakai *mask*. Kedua, semua angka paper tadi diukur di data validasi CrowdAI yang bocor.

Baris terakhir adalah model saya di data *test* yang bersih: AP 61,94 dan C-IoU 74,55. Karena datanya berbeda, baris itu tidak bisa dibandingkan langsung dengan baris di atasnya. Perbandingan yang adil saya tunjukkan nanti, di slide HiSup.

## Slide 5: Tiga gagasan yang diuji, dengan aturan yang dibekukan

*±1 menit*

Saya menguji tiga gagasan. N1 adalah gagasan metode: jaringan diberi satu keluaran tambahan, disebut *head* orientasi, yang menebak arah hadap bangunan, yaitu sudut θ. Label sudutnya dihitung otomatis dari *mask*, lalu sudut itu dipakai untuk merapikan poligon.

N2 bukan metode, melainkan cara evaluasi: dataset CrowdAI dibersihkan dari duplikat, lalu hasil dilaporkan per tingkat kepadatan bangunan.

N3 adalah gagasan *preprocessing*: dua peta dari morfologi klasik, yaitu MBI dan peta celah, ditambahkan sebagai kanal input, dengan harapan celah antar-rumah lebih mudah terlihat oleh jaringan.

Yang ingin saya tekankan: kriteria berhasil atau gagal untuk tiap gagasan saya tulis dan saya bekukan sebelum ada model yang dilatih. Setiap perbandingan memakai tiga *seed* dan selang kepercayaan *bootstrap*. Akibatnya hasil negatif pun tetap dilaporkan, dan itu akan terlihat di slide-slide berikutnya.

## Slide 6: Arsitektur: alur pengerjaan dari awal sampai akhir

*±1,5 menit*

Ini arsitektur keseluruhannya, dalam tiga tahap. Tiap baris dibaca dari kiri ke kanan.

Baris pertama adalah data dan label. Datanya CrowdAI dan WHU, berupa citra RGB dan *mask* per bangunan, tanpa anotasi poligon. Data itu dideduplikasi dengan *perceptual hash*, lalu dibagi menjadi train, tuning, dan *test* per klaster, supaya tidak ada citra kembar di dua bagian. Dari *mask* kemudian dibuat tiga label secara otomatis: kelas kontak, orientasi θ, dan bobot tepi.

Baris kedua adalah jaringannya. Inputnya satu *tile* 300 × 300 piksel, yang hanya di-*pad* ke 320 dan tidak diubah ukurannya. Jaringannya U-Net++ dengan *encoder* EfficientNet-B4. Keluarannya lima kanal: tiga kelas, yaitu latar, bangunan, dan kontak, ditambah dua kanal orientasi. Saat *training*, keluaran itu dibandingkan dengan label dari baris pertama lewat *loss* gabungan; itulah arti panah turun di sebelah kanan.

Baris ketiga mengubah keluaran jaringan menjadi poligon. Dari peta kelas dibuat *marker*, yaitu inti tiap rumah, lalu *watershed* memekarkannya sehingga satu gumpalan terpecah menjadi satu *instance* per rumah. Kontur tiap rumah diambil, lalu dirapikan searah θ: bangunan yang hampir persegi menjadi kotak, yang lain disederhanakan dan dindingnya diluruskan. Hasilnya satu poligon per rumah, dalam format COCO JSON dan GeoJSON.

Kotak terakhir adalah evaluasi: poligon dinilai dengan AP, C-IoU, PoLiS, dan MTA, dan hasilnya dilaporkan per tingkat kepadatan. Jadi dari awal sampai akhir, tidak ada anotasi poligon yang dipakai.

## Slide 7: Pengolahan citra klasik di dalam alur

*±1,5 menit*

Karena ini mata kuliah Pengolahan Citra Digital, saya tunjukkan di mana saja algoritma klasik dipakai. Yang saya maksud klasik adalah algoritma tetap, tanpa parameter yang dilatih.

Lima baris pertama ada di alur utama. Untuk membuat label: dilasi menghasilkan kelas kontak, *distance transform* menghasilkan bobot tepi, dan *minAreaRect* menghasilkan orientasi. Setelah jaringan: *thresholding* dan *connected component labeling* membuat *marker*, *watershed* memisahkan rumah, *marching squares* mengambil kontur, lalu momen orde dua dan algoritma Douglas-Peucker merapikannya menjadi kotak atau poligon siku.

Jadi jaringan hanya mengerjakan bagian tengah, yaitu dari citra ke peta kelas dan orientasi. Yang mengubah peta itu menjadi satu poligon per rumah seluruhnya algoritma klasik. Untuk tepatnya, *minAreaRect* dan Douglas-Peucker bekerja pada titik dan garis, jadi lebih pas disebut geometri komputasi.

Baris keenam adalah *baseline* B0, yang seluruhnya klasik tanpa jaringan: indeks bangunan MBI, ambang Otsu, *opening*, lalu *watershed*. AP-nya hanya 0,08, jadi cara klasik saja tidak cukup untuk data ini.

Baris terakhir adalah gagasan N3: peta dari *top-hat* multi-arah dijadikan input tambahan bagi jaringan. Hasilnya tidak membantu, dan itu saya bahas di slide hasil negatif.

## Slide 8: Label diturunkan otomatis dari *mask*

*±1 menit*

Karena tidak ada anotasi poligon, semua label dibuat otomatis dari peta *instance*, yaitu *mask* yang tiap bangunannya punya nomor sendiri.

Label pertama adalah kelas kontak: piksel bangunan yang jaraknya paling jauh dua piksel dari bangunan lain. Tujuannya menandai tempat dua atap bertemu.

Label kedua adalah orientasi. Untuk tiap bangunan dicari kotak terkecil yang membungkusnya, dengan fungsi *minAreaRect* di OpenCV, lalu diambil arah sisinya. Sudut itu dikodekan sebagai sinus dan kosinus dari 4θ. Alasannya, bangunan persegi terlihat sama bila diputar 90 derajat, jadi keempat arah sisinya harus dianggap satu arah yang sama.

Yang ketiga adalah bobot tepi: piksel di dekat batas bangunan diberi bobot *loss* lebih besar, supaya jaringan tidak mengabaikan celah sempit di antara rumah.

Kelas kontak dan bobot tepi saya adopsi dari penelitian sebelumnya. Yang baru di sini adalah label orientasi yang diturunkan dari *mask*.

## Slide 9: Evaluasi tanpa kebocoran data

*±1 menit*

Bagian ini menurut saya yang paling penting. Dataset CrowdAI punya masalah yang sudah dilaporkan peneliti lain: banyak citra validasi yang sama persis dengan citra train, atau hanya diputar dan dicerminkan. Model seolah diuji dengan soal yang sudah pernah dilihatnya, sehingga angkanya terlalu optimistis.

Karena itu saya mendeduplikasi dengan *perceptual hash*. Dari 280.741 *tile* train tersisa 29.283 klaster, dan dari tiap klaster diambil satu wakil.

Klaster itu lalu dibagi: 23.427 *tile* untuk train, 1.000 untuk menyetel *post-processing*, 1.928 untuk memilih model, dan 2.928 *tile* *test* yang disisihkan.

*Tile* *test* dibagi menjadi tiga kelompok kepadatan, yaitu jarang, sedang, dan padat, menurut jumlah bangunannya. Pembagian ini dibekukan dari *ground truth* sebelum ada model yang dievaluasi.

Setiap selisih antarmodel dihitung berpasangan pada *tile* yang sama, dengan tiga *seed* dan *bootstrap* seribu replikasi. Dan semua parameter disetel di data tuning; data *test* tidak pernah dipakai untuk menyetel apa pun.

## Slide 10: Hasil utama di CrowdAI *test*

*±1 menit*

Tabel ini merangkum hasil di data *test*, rata-rata tiga *seed*. Sebelum membacanya, sedikit tentang metriknya. AP mengukur ketepatan deteksi tiap bangunan; makin besar makin baik. C-IoU adalah IoU yang dikenai penalti bila jumlah titik sudut poligon jauh dari *ground truth*, jadi ia mengukur kerapian. MTA adalah galat sudut dinding dalam derajat. *Merge rate* adalah persentase rumah yang tergabung dengan tetangganya. Untuk MTA dan *merge rate*, makin kecil makin baik.

Ada tiga hal yang bisa dibaca. Pertama, regularisasi sangat menaikkan kerapian: dari B4 ke B5, C-IoU naik dari 62,69 ke 73,87, dan MTA turun dari 47,11 ke 37,30 derajat.

Kedua, jalur biner unggul di AP: B3m 64,89, sedangkan FULL-P0 61,94.

Ketiga, jalur tiga kelas lebih baik memisahkan rumah: *merge rate* FULL-P0 1,96%, B3m 2,81%. Dengan bantuan detektor, FULL+Y turun lagi ke 1,52%.

Jadi tidak ada satu model yang menang di semua kolom. Itu saya uraikan satu per satu.

## Slide 11: N1 didukung: *head* θ merapikan poligon

*±1 menit*

Sekarang gagasan pertama. Yang dibandingkan adalah B5 dan FULL-P0, di 857 *tile* terpadat. Di B5, sudut perapian diambil dari kotak terkecil kontur hasil prediksi. Di FULL-P0, jaringan dilatih dengan *head* θ dan sudutnya dibaca dari *head* itu.

PoLiS adalah jarak rata-rata antara batas poligon prediksi dan batas *ground truth*. Nilainya turun 0,037 piksel. MTA, galat sudut dinding, turun 0,94 derajat. Selang kepercayaan 95% keduanya tidak memuat nol, dan arahnya sama di ketiga *seed*.

AP berubah +0,35 poin, jadi kerapian itu tidak dibayar dengan akurasi deteksi.

Menurut aturan yang sudah dibekukan, N1 dinyatakan didukung. Tetapi saya harus jujur soal besarnya: efeknya kecil, jauh di bawah satu piksel. Jadi kesimpulannya, *head* θ membantu secara konsisten, bukan membantu secara besar.

## Slide 12: Efek N1 membesar di *tile* padat

*±1 menit*

Gambar ini memecah efek tadi menurut kepadatan. Tiap titik adalah selisih antara dua model, dan garis tegaknya selang kepercayaan 95%. Titik di bawah garis putus-putus berarti ada perbaikan.

Di dua panel kiri, untuk *head* θ, perbaikannya membesar dari *tile* jarang ke padat. PoLiS: −0,012, −0,021, lalu −0,037 piksel. MTA: −0,42, −0,73, lalu −0,94 derajat.

Dua titik di sebelah kanan tiap panel membagi *tile* dengan cara lain: yang berisi bangunan bersentuhan dan yang tidak. Selang kelompok bersentuhan lebar karena *tile*-nya hanya 146 dari 2.928.

Kembali ke kepadatan: kecenderungan belum tentu bukti. Ketika selisih padat dikurangi jarang diuji, hanya MTA yang signifikan, yaitu −0,51 derajat. Untuk PoLiS selang kepercayaannya memuat nol. Karena itu klaim bahwa efek N1 paling besar di area padat tidak saya pakai.

Panel kanan adalah prior morfologi, gagasan N3. Di sana titik-titiknya berada di atas nol, artinya rumah yang tergabung tidak berkurang. Itu saya bahas di slide berikutnya.

## Slide 13: Dua hasil negatif, dilaporkan apa adanya

*±1 menit*

Dua bagian yang dirancang untuk memisahkan rumah berdempetan ternyata tidak bekerja seperti yang saya harapkan.

Yang pertama N3. Dengan MBI dan peta celah sebagai input tambahan, *merge rate* berubah +0,15 poin, dengan selang [+0,02; +0,30]. Artinya rumah yang tergabung sedikit bertambah, bukan berkurang, dan arahnya pun tidak sama di ketiga *seed*. Jadi hipotesis bahwa prior celah membantu tidak didukung data.

Yang kedua kelas kontak. Di 6 dari 9 *run* FULL, jaringan tidak pernah memilih kelas kontak sama sekali. Penjelasan yang paling mungkin: piksel kontak sangat langka dibanding piksel bangunan, sehingga jaringan lebih untung mengabaikannya. Rumah-rumah tetap terpisah, tetapi karena ada celah latar di antaranya, bukan karena kelas kontak.

Keduanya saya laporkan apa adanya, karena hasil negatif yang diukur dengan benar tetap berguna bagi yang ingin mencoba gagasan serupa.

## Slide 14: Banding dengan HiSup di data tanpa kebocoran

*±1,5 menit*

Bagaimana posisi hasil ini dibanding metode terbaik? Pembandingnya HiSup, metode ekstraksi poligon bangunan dari Xu dan rekan-rekannya tahun 2023, yang dilatih dengan anotasi poligon dan termasuk yang terbaik di CrowdAI. HiSup merilis file prediksinya, jadi bisa saya nilai ulang dengan kode evaluasi yang sama, tanpa menjalankan modelnya.

Di data validasi penuh, yang bocor, HiSup mendapat AP 79,43. Angka itu cocok dengan yang dilaporkan di papernya, dan itu sekaligus memeriksa bahwa kode evaluasi saya benar.

Tetapi di 212 *tile* validasi yang benar-benar tidak punya kembaran di train, AP HiSup turun menjadi 65,60. Di *tile* yang sama, FULL-P0 mendapat 61,14, tertinggal 4,45 poin. Perlu diingat, HiSup dilatih dengan anotasi poligon, sedangkan model saya tidak.

Ambang "mendekati metode terbaik" yang saya bekukan sejak awal adalah tiga poin AP. Selisihnya lebih besar dari itu, jadi klaim tersebut tidak saya pakai.

Temuan yang menurut saya lebih penting di slide ini adalah baris pertama dan kedua: angka HiSup sendiri turun jauh begitu duplikat disingkirkan. Artinya angka di *benchmark* CrowdAI perlu dibaca dengan hati-hati.

## Slide 15: Eksploratif: *baseline* sederhana dan *seed* detektor

*±1 menit*

Dua percobaan di slide ini saya tambahkan setelah hasil utama ada. Karena itu sifatnya eksploratif dan bukan bagian dari klaim.

Yang pertama B3m: segmentasi biner biasa, ditambah regularisasi yang sama. Ternyata ini konfigurasi yang paling kuat. AP-nya 64,89 dan C-IoU-nya 76,97, sedangkan FULL-P0 61,94 dan 74,55. B3m unggul di semua metrik kecuali satu, yaitu rumah yang tergabung.

Yang kedua, sebuah detektor kecil, YOLOv8n-OBB, yang juga dilatih dari *mask*, dipakai untuk memberi titik benih bagi tiap rumah. Hasilnya *merge rate* turun 0,45 poin di FULL-P0 dan 0,67 poin di B3m, dengan AP yang setara. Ada harganya: rumah yang terpecah bertambah 0,24 dan 0,36 poin.

Jadi gambaran besarnya: jalur biner paling akurat dan paling rapi, jalur tiga kelas paling baik memisahkan rumah berdempetan, dan benih dari detektor membantu keduanya.

## Slide 16: Eksploratif: *head* θ di jalur biner tidak membantu

*±1 menit*

Karena B3m paling kuat, pertanyaan yang wajar: bagaimana kalau *head* θ dipasang di jalur biner itu? Percobaan ini juga eksploratif, tetapi aturan penilaiannya tetap saya bekukan sebelum *training*.

Hasilnya negatif, dan penyebabnya bisa dipisahkan menjadi dua. Pemanduan oleh θ sendiri tetap membantu: PoLiS −0,016 piksel dan MTA −0,18 derajat, hampir sama dengan di jalur tiga kelas.

Tetapi melatih *head* θ bersama *mask* menurunkan mutu *mask* itu sendiri. IoU validasi turun dari 89,75% menjadi 88,78%, dan AP turun 2,25 poin.

Kerugian itu lebih besar dari keuntungannya. Totalnya, dibanding B3m: PoLiS +0,084 piksel, MTA +0,30 derajat, dan AP −2,26 poin, searah di ketiga *seed*. Jadi aturannya tidak terpenuhi, dan B3m tetap yang terbaik.

## Slide 17: Contoh hasil di *tile* asli

*±1 menit*

Supaya tidak hanya angka, ini contoh pada *tile* *test*. Kolom paling kiri adalah *ground truth*, digambar hijau. Empat kolom berikutnya adalah poligon dari B4, FULL-P0, FULL+Y, dan B3m, digambar kuning. Angka kecil di pojok kiri bawah tiap gambar adalah jumlah bangunannya.

Di kolom B4, kontur masih mentah: garisnya bergelombang dan titik sudutnya banyak. Di tiga kolom berikutnya, setelah regularisasi, poligon menjadi siku dengan sedikit titik sudut, dan lebih mirip *ground truth*.

Tiga baris pertama adalah *tile* padat yang hasilnya cukup baik. Baris terbawah sengaja saya pilih sebagai contoh kegagalan: di sisi kanan bawah, beberapa rumah yang berdempetan masih tergabung menjadi satu poligon besar, di semua model. Inilah kasus yang belum terselesaikan.

## Slide 18: WHU dan biaya komputasi

*±1 menit*

Untuk melihat apakah hasilnya terbawa ke data lain, model yang sama saya latih di dataset WHU, dengan satu *seed*. AP-nya 66,97 dan C-IoU-nya 72,07.

Kalau dihitung dengan aturan Pix2Poly, C-IoU kita 66,83. Angka yang dilaporkan di paper: HiSup 79,62 dan Pix2Poly 81,63. Perbandingan ini hanya kira-kira, karena daftar *tile* uji Pix2Poly tidak bisa direkonstruksi persis. Meski begitu, jaraknya jelas besar.

Titik lemahnya sama dengan di CrowdAI: di *patch* yang berisi bangunan bersentuhan, 14,73% rumah masih tergabung.

Dari sisi biaya, modelnya 20,81 juta parameter. Alur penuh, dari citra sampai poligon, berjalan 42,3 *tile* per detik di RTX 5090, dan 36,4 bila detektor ikut dipakai.

## Slide 19: Keterbatasan

*±1 menit*

Ada lima hal yang membatasi kesimpulan penelitian ini.

Pertama, data bersih untuk banding dengan HiSup hanya 212 *tile*, dan itu bukan sampel acak dari data validasi.

Kedua, di WHU hanya ada satu *seed*, dan daftar *tile* ujinya tidak persis sama dengan yang dipakai Pix2Poly.

Ketiga, efek N1 kecil. Di jalur biner, melatih *head* θ malah menurunkan mutu *mask*, walaupun itu baru satu konfigurasi dan sifatnya eksploratif.

Keempat, detektor YOLO hanya dilatih sekali dan dipakai untuk ketiga *seed*, jadi ragamnya antar-*seed* tidak terukur.

Kelima, poligon yang dihasilkan hanya cincin luar bangunan. Bangunan yang punya halaman dalam, atau lubang, tidak terwakili.

## Slide 20: Kesimpulan

*±1 menit*

Ada empat hal yang bisa dibawa pulang.

Pertama, poligon yang rapi bisa dihasilkan dari *mask* saja. Regularisasi geometris bekerja tanpa anotasi poligon.

Kedua, *head* orientasi membantu sedikit. Efeknya konsisten di jalur tiga kelas dan makin terasa di *tile* padat, tetapi tidak terbawa ke jalur biner.

Ketiga, prior morfologi dan kelas kontak tidak bekerja seperti yang diharapkan, dan keduanya dilaporkan sebagai hasil negatif.

Keempat, *baseline* yang sederhana ternyata paling kuat. Hal ini baru terlihat karena semua model dinilai dengan protokol yang sama, di data yang bebas kebocoran.

Kalau dirangkum dalam satu kalimat: sumbangan utama proyek ini bukan model yang mengalahkan metode terbaik, melainkan evaluasi yang jujur tentang seberapa jauh pendekatan berbasis *mask* bisa melangkah.

## Slide 21: Referensi

*±0,5 menit*

Ini rujukan utama yang saya pakai. Daftar lengkapnya ada di naskah paper.

## Slide 22: Penutup

*±0,5 menit*

Demikian presentasi saya. Kode, konfigurasi, dan semua angka di slide ini bisa direproduksi dari repositori proyek. Terima kasih atas perhatiannya. Saya persilakan bila ada pertanyaan atau masukan.

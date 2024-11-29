# Klasifikasi dokumen berita bahasa indonesia


##Cara Train model
- Download dataset
- kemudian lakukan preprocess pada dataset (pastikan outputnya berupa file txt dengan isi "__label__kategori text"
- masukan dataset yang sudah melalui tahap preprocess kedalam folder Train model
- Download model cc.id.300.bin (file model word embeddings yang telah dilatih pada korpus teks dalam bahasa Indonesia)
- Masukan cc.id.300.bin kedalam folder Train Model
- buka file Train.py (sesuaikan nama dataset .txt dengan yang ditrain
- run file dengan python3 Train.py atau python.py
- setelah program tersebut selesai maka akan menghasilkan file model bernama cnn_classifier.pth

##Cara Testing model
- pindahkan file cc.id.300.bi dan cnn_classifier.pth kedalam folder pengujian
- buka file pengujian.py
- ubah kode inputtesting.csv jika ingin menguji menggunakan data uji judul dan pilih inputtestingisi.csv jika ingin menggunakan dataset isi berita
- run file dengan python3 Train.py atau python.py
- setelah program tersebut selesai maka akan menghasilkan output terminal yang terdiri dari text, label dan label prediksi serta menampilkan hasil akurasi dari model tersebut
  

# Klasifikasi dokumen berita bahasa indonesia


## Cara Train model
- Download dataset (convert file menjadi format txt dengan isi sebagai berikut "__label__kategori text" tanpa tanda petik)
- kemudian lakukan preprocessing pada dataset (bisa menggunakan file preprocessing.py pada folder preprocessing)
- masukan dataset yang sudah melalui tahap preprocess kedalam folder Train model
- Download model cc.id.300.bin pada link berikut https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz (file model word embeddings yang telah dilatih pada korpus teks dalam bahasa Indonesia)
- Masukan cc.id.300.bin kedalam folder Train Model
- buka file Train.py (sesuaikan nama dataset .txt dengan yang ditrain
- run file dengan python3 Train.py atau python.py
- setelah program tersebut selesai maka akan menghasilkan file model bernama cnn_classifier.pth

## Cara Testing model
- pindahkan file cc.id.300.bi dan cnn_classifier.pth kedalam folder pengujian
- buka file pengujian.py
- ubah kode inputtesting.csv jika ingin menguji menggunakan data uji judul dan pilih inputtestingisi.csv jika ingin menggunakan dataset isi berita
- run file dengan python3 Train.py atau python.py
- setelah program tersebut selesai maka akan menghasilkan output terminal yang terdiri dari text, label dan label prediksi serta menampilkan hasil akurasi dari model tersebut
  

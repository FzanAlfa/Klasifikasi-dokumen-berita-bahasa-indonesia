import pandas as pd  # Untuk pengolahan data menggunakan DataFrame
import nltk  # Untuk pengolahan teks seperti tokenisasi dan stopword
from nltk.corpus import stopwords  # Untuk daftar stopword
from nltk.tokenize import word_tokenize  # Untuk memecah teks menjadi token
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Untuk stemming teks bahasa Indonesia


nltk.download('punkt')  # Unduh data tokenizer untuk NLTK
nltk.download('stopwords')  # Unduh daftar stopword

# Fungsi preprocessing teks
def preprocess_text(text, index=None, total=None):
    # Menampilkan progres teks yang sedang diproses
    if index is not None and total is not None:
        print(f"Preprocessing text {index + 1}/{total}: {text[:50]}...")  # Menampilkan bagian awal teks
    
    # 1. Case Folding
    text = text.lower()  # Mengubah semua teks menjadi huruf kecil
    
    # 2. Tokenization
    tokens = word_tokenize(text)  # Memecah teks menjadi kata-kata atau token
    
    # 3. Stopword Elimination
    stop_words = set(stopwords.words('indonesian'))  # Ambil daftar stopword bahasa Indonesia
    tokens = [word for word in tokens if word not in stop_words]  # Hilangkan kata-kata yang ada dalam stopword
    
    # 4. Stemming
    stemmer_factory = StemmerFactory()  # Membuat objek factory untuk stemming
    stemmer = stemmer_factory.create_stemmer()  # Membuat stemmer
    tokens = [stemmer.stem(word) for word in tokens]  # Stem setiap kata dalam token
    
    # Gabungkan kembali token menjadi satu string
    return ' '.join(tokens)  # Kembalikan hasil preprocessing sebagai string

# Baca data teks
file_path = 'isiberita.txt'  # Path ke file teks yang berisi data ####################################################

# Buat list untuk menyimpan data
data = []

# Baca file teks
with open(file_path, 'r') as f:  # Buka file teks dalam mode baca
    for i, line in enumerate(f, start=1):  # Iterasi setiap baris dalam file
        label, text = line.split(' ', 1)  # Pisahkan label (sebelum spasi pertama) dan teks (setelahnya)
        data.append({'label': label.strip(), 'text': text.strip()})  # Simpan sebagai dictionary

# Konversi ke DataFrame
df = pd.DataFrame(data)  # Konversi list dictionary ke DataFrame

# Terapkan preprocessing pada kolom 'text' dengan progress
print("\nStart preprocessing texts...")  # Tampilkan pesan mulai preprocessing
############################ TAHAP PREPROCESSING ###################################
total_rows = len(df)  # Hitung total baris dalam DataFrame
for i, row in df.iterrows():  # Iterasi setiap baris dalam DataFrame
    df.at[i, 'clean_text'] = preprocess_text(row['text'], index=i, total=total_rows)  # Preprocessing teks
############################ TAHAP PREPROCESSING ###################################
print("Preprocessing complete!\n")  # Tampilkan pesan selesai

# Simpan hasil preprocessing ke file baru dalam format txt
output_file = 'outputjudulberita.txt'  # Path file output
with open(output_file, 'w') as f:  # Buka file output dalam mode tulis
    for _, row in df.iterrows():  # Iterasi setiap baris dalam DataFrame
        f.write(f"{row['label']} {row['clean_text']}\n")  # Tulis label dan teks yang sudah diproses

print(f"Processed data saved to '{output_file}'")  # Tampilkan pesan lokasi file hasil

import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
nltk.download('punkt')


# Title
st.title('Aplikasi Klasifikasi Kategori Berita')

# Load dataset
# Pastikan dataset Anda memiliki kolom 'Artikel' untuk konten berita dan 'category' untuk kategori berita
news_dataset = pd.read_csv('data-uas.csv')  # Ganti 'your_news_dataset.csv' dengan jalur file aktual Anda

# Gantilah NaN dengan string kosong
news_dataset['Artikel'].fillna('', inplace=True)
news_dataset['Kategori'].fillna('', inplace=True)  # Sesuaikan dengan nama kolom yang berisi label kategori

# Tampilkan Dataset
st.write('## Dataset Berita')
st.dataframe(data=news_dataset)

# Hapus baris dengan nilai NaN dari dataset
news_dataset.dropna(subset=['Artikel', 'Kategori'], inplace=True)

# Tambahkan Klasifikasi
algorithm = st.sidebar.selectbox(
    'Pilih Algoritma Klasifikasi',
    ('Multinomial Naive Bayes',)
)

# Tambahkan Parameter
def tambah_parameter(algorithm):
    params = dict()
    if algorithm == 'Multinomial Naive Bayes':
        alpha = st.sidebar.slider('Alpha', 0.1, 1.0, step=0.1)
        params['alpha'] = alpha
    return params

parameters = tambah_parameter(algorithm)

# Pilih Klasifikasi
def pilih_klasifikasi(algorithm, parameters):
    classifier = None
    if algorithm == 'Multinomial Naive Bayes':
        classifier = MultinomialNB(alpha=parameters['alpha'])
    return classifier

# Inisialisasi model dan vektorisasi di sini
def custom_tokenizer(text):
    # Tokenisasi kustom sesuai kebutuhan Anda
    words = word_tokenize(text)
    words = [re.sub(r'[^A-Za-z]', '', word) for word in words]  # Hapus karakter selain huruf
    return words

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, tokenizer=custom_tokenizer)
X = vectorizer.fit_transform(news_dataset['Artikel']).toarray()
y = news_dataset['Kategori']
clf = pilih_klasifikasi(algorithm, parameters)

# Proses Klasifikasi
if clf:
    # Input Text Area
    berita_input = st.text_area("Masukkan teks berita:")

    if st.button("Klasifikasi"):
        if berita_input:
            # Pastikan model dilatih sebelum melakukan prediksi
            clf.fit(X, y)

            # Vektorisasi TF-IDF
            berita_input_vectorized = vectorizer.transform([berita_input]).toarray()
            
            # Prediksi
            prediction = clf.predict(berita_input_vectorized)[0]

            # Tampilkan hasil prediksi
            st.subheader("Hasil Prediksi Kategori:")
            st.write(prediction)

            # Menampilkan metrik klasifikasi jika model sudah dilatih
            # Prediksi pada data pengujian
            y_pred = clf.predict(X)

            # Metrik klasifikasi
            st.write("## Metrik Klasifikasi pada Data Pengujian")
            st.write("Akurasi:", accuracy_score(y, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y, y_pred))
            st.write("Classification Report:")
            st.write(classification_report(y, y_pred))

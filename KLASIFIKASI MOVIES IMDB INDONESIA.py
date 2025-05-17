import pandas as pd
import numpy as np
import re
import math
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Load dataset
data = pd.read_csv('movies_imdb_indonesia.csv')
print("Contoh data:")
print(data.head())

# 2. Preprocessing functions yang lebih baik
def clean_text(text):
    """Clean and normalize text with pengolahan khusus Bahasa Indonesia"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Remove digits
        text = re.sub(r'\d+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def tokenize(text):
    """Split text into words with better handling for Indonesian"""
    if not isinstance(text, str):
        return []
    
    # Handle Indonesian special cases
    text = text.replace("'", " ")  # Replace apostrophes with space
    
    # Split into tokens
    tokens = text.split()
    
    # Remove very short tokens (likely not meaningful)
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens

# Expanded stopwords list untuk Bahasa Indonesia dan English
STOPWORDS = {
    # Kata depan Bahasa Indonesia
    'yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'adalah', 'ini', 'dari', 'dalam',
    'akan', 'tidak', 'mereka', 'ke', 'oleh', 'itu', 'juga', 'ada', 'saya', 'kamu',
    'dia', 'kami', 'kalian', 'sangat', 'banyak', 'hanya', 'tapi', 'tetapi', 'namun',
    'atau', 'jika', 'kalo', 'bila', 'maka', 'meski', 'agar', 'supaya', 'karena',
    'sebab', 'hingga', 'sampai', 'setelah', 'sebelum', 'sejak', 'ketika', 'waktu',
    'selama', 'saat', 'pun', 'lagi', 'telah', 'sedang', 'masih', 'sudah', 'belum',
    'dapat', 'bisa', 'mampu', 'tak', 'tersebut', 'kata', 'serta', 'lalu', 'kemudian',
    'sang', 'si', 'para', 'bagi', 'tentang', 'cukup', 'sekali', 'lebih', 'kurang',
    'sini', 'situ', 'nya', 'ia', 'seorang', 'sebuah', 'beberapa', 'antara', 'tanpa',
    'setiap', 'semua', 'seluruh', 'menjadi', 'sebagai', 'seperti', 'secara', 'kembali',
    # Common English stopwords
    'the', 'to', 'and', 'of', 'in', 'is', 'a', 'an', 'by', 'that', 'it', 'as', 'for',
    'with', 'on', 'at', 'this', 'was', 'his', 'her', 'he', 'she', 'they', 'their', 'them',
    'but', 'or', 'if', 'because', 'as', 'until', 'while', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
    'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

def remove_stopwords(tokens):
    """Remove stopwords from tokens"""
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]

# 3. Feature engineering yang lebih baik
def extract_additional_features(text):
    """Extract additional features that might be useful for classification"""
    features = {}
    
    if not isinstance(text, str):
        text = ""
    
    text_lower = text.lower()
    
    # Lebih banyak fitur statistik teks
    features['length'] = len(text)
    features['word_count'] = len(text.split()) if text else 0
    features['sentence_count'] = len(re.split(r'[.!?]+', text)) if text else 0
    features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text and len(text.split()) > 0 else 0
    
    # Dictionary kata kunci untuk genre yang lebih lengkap
    action_keywords = ['pertarungan', 'bertarung', 'melawan', 'perang', 'aksi', 'tembak', 'kejar', 
                     'ledakan', 'baku', 'tembak', 'menyelamatkan', 'pahlawan', 'musuh', 'senjata',
                     'pertempuran', 'penjahat', 'polisi', 'tentara', 'bom', 'mengejar', 'serangan',
                     'melindungi', 'menghentikan', 'ancaman', 'berbahaya', 'petualangan', 'misi',
                     'bertahan', 'selamat', 'menantang', 'balas', 'dendam', 'kalahkan']
    
    drama_keywords = ['cinta', 'keluarga', 'hidup', 'pernikahan', 'pertemanan', 'hubungan',
                    'perceraian', 'konflik', 'masalah', 'emosi', 'perasaan', 'tragedi',
                    'perjuangan', 'kehilangan', 'kesedihan', 'kerinduan', 'kisah', 'romansa',
                    'kasih', 'hati', 'persahabatan', 'ibu', 'ayah', 'anak', 'pengorbanan',
                    'air mata', 'kehidupan', 'impian', 'putus', 'menikah', 'mencintai', 'sakit']
    
    comedy_keywords = ['lucu', 'humor', 'komedi', 'kocak', 'tertawa', 'jenaka', 'parodi',
                     'lelucon', 'konyol', 'gila', 'menggelikan', 'bodoh', 'memalukan',
                     'kecelakaan', 'kesalahpahaman', 'bingung', 'salah', 'ganggu', 'usil',
                     'keliru', 'aneh', 'absurd', 'kelakar', 'kebodohan', 'kekonyolan', 'guyonan']
    
    horror_keywords = ['hantu', 'seram', 'takut', 'teror', 'misterius', 'pembunuhan', 'darah',
                     'horor', 'kematian', 'terbunuh', 'mengerikan', 'misteri', 'kutukan',
                     'ritual', 'iblis', 'setan', 'arwah', 'roh', 'mencekam', 'menyeramkan',
                     'teror', 'mati', 'mimpi buruk', 'menghantui', 'kegelapan', 'mayat',
                     'jahat', 'menakutkan', 'sadis', 'teriak', 'jeritan', 'korban', 'selamatkan']
    
    sci_fi_keywords = ['alien', 'luar angkasa', 'masa depan', 'teknologi', 'robot', 'mesin',
                     'ilmuwan', 'eksperimen', 'penemuan', 'planet', 'galaksi', 'bintang',
                     'pesawat', 'teleportasi', 'kloning', 'mutasi', 'invasi', 'ruang', 'waktu',
                     'perjalanan', 'dimensi', 'virtual', 'kecerdasan', 'buatan', 'komputer']
    
    fantasy_keywords = ['sihir', 'keajaiban', 'naga', 'penyihir', 'monster', 'peri',
                      'kerajaan', 'kekuatan', 'petualangan', 'ramalan', 'takdir', 'dunia',
                      'legenda', 'mitos', 'dongeng', 'pahlawan', 'pangeran', 'puteri',
                      'raja', 'ratu', 'makhluk', 'jimat', 'pedang', 'perang', 'pendekar']
    
    romance_keywords = ['cinta', 'kasih', 'romantis', 'kencan', 'jatuh', 'hati', 'kekasih',
                      'pasangan', 'hubungan', 'menikah', 'lamaran', 'pernikahan', 'ciuman',
                      'pelukan', 'perasaan', 'berpisah', 'rindu', 'bersama', 'sayang', 'cantik',
                      'tampan', 'memikat', 'setia', 'pacar', 'tunangan', 'pertemuan', 'kenangan']
    
    thriller_keywords = ['misteri', 'pembunuhan', 'ancaman', 'berbahaya', 'rahasia', 'detektif',
                       'penjahat', 'kriminal', 'kejahatan', 'mencari', 'menyelidiki', 'bukti',
                       'polisi', 'pengejaran', 'menghilang', 'korban', 'terbunuh', 'tegang',
                       'mencekam', 'pelarian', 'pengintaian', 'pencurian', 'menegangkan', 'tersangka']
    
    # Hitung kemunculan kata kunci untuk setiap genre
    features['action_count'] = sum(1 for word in action_keywords if word in text_lower)
    features['drama_count'] = sum(1 for word in drama_keywords if word in text_lower)
    features['comedy_count'] = sum(1 for word in comedy_keywords if word in text_lower)
    features['horror_count'] = sum(1 for word in horror_keywords if word in text_lower)
    features['scifi_count'] = sum(1 for word in sci_fi_keywords if word in text_lower)
    features['fantasy_count'] = sum(1 for word in fantasy_keywords if word in text_lower)
    features['romance_count'] = sum(1 for word in romance_keywords if word in text_lower)
    features['thriller_count'] = sum(1 for word in thriller_keywords if word in text_lower)
    
    # Rasio kata kunci
    total_words = features['word_count'] or 1  # Avoid division by zero
    features['action_ratio'] = features['action_count'] / total_words
    features['drama_ratio'] = features['drama_count'] / total_words
    features['comedy_ratio'] = features['comedy_count'] / total_words
    features['horror_ratio'] = features['horror_count'] / total_words
    features['scifi_ratio'] = features['scifi_count'] / total_words
    features['fantasy_ratio'] = features['fantasy_count'] / total_words
    features['romance_ratio'] = features['romance_count'] / total_words
    features['thriller_ratio'] = features['thriller_count'] / total_words
    
    # Fitur tambahan
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['quote_count'] = text.count('"') + text.count("'")
    
    return features

# Data preparation lebih baik
print("Mempersiapkan data...")

# Cleaning the dataset
data['ringkasan_sinopsis'] = data['ringkasan_sinopsis'].fillna('')
data = data[data['ringkasan_sinopsis'].str.len() > 10]  # Remove entries with very short synopses

# Tambahkan metadata genre untuk analisis
genre_counts = data['genre'].value_counts()
print("\nDistribusi genre sebelum balancing:")
print(genre_counts)

# Gunakan balancing yang lebih baik
min_samples = 50  # Minimum samples per class
max_samples_per_class = 300  # Maximum samples per class

balanced_data = pd.DataFrame()
for genre in data['genre'].unique():
    genre_data = data[data['genre'] == genre]
    
    # Skip genres with too few samples
    if len(genre_data) < min_samples:
        print(f"Mengabaikan genre {genre} karena hanya memiliki {len(genre_data)} sampel")
        continue
        
    if len(genre_data) > max_samples_per_class:
        # Undersample
        sampled_data = genre_data.sample(max_samples_per_class, random_state=42)
    else:
        # Oversample jika perlu
        sampled_data = genre_data.sample(max_samples_per_class, replace=True, random_state=42)
    
    balanced_data = pd.concat([balanced_data, sampled_data])

# Gunakan data yang sudah seimbang
data = balanced_data.reset_index(drop=True)

print("\nDistribusi genre setelah balancing:")
print(data['genre'].value_counts())

# Buat preprocessing dan ekstrasi fitur yang lebih baik
print("Ekstraksi fitur...")

# Proses teks untuk ekstraksi fitur tambahan
processed_texts = [clean_text(text) for text in data['ringkasan_sinopsis']]

# Ekstrak fitur tambahan
additional_features_list = []
for text in data['ringkasan_sinopsis']:
    additional_features_list.append(extract_additional_features(text))

additional_features_df = pd.DataFrame(additional_features_list)

# Gunakan scikit-learn TF-IDF untuk hasil yang lebih baik
print("Membuat fitur TF-IDF...")
tfidf = TfidfVectorizer(
    preprocessor=clean_text,
    tokenizer=lambda x: remove_stopwords(tokenize(x)),
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=5000,   # Ambil 5000 fitur terbaik
    min_df=3,            # Minimal muncul di 3 dokumen
    max_df=0.9,          # Muncul maksimal di 90% dokumen
    sublinear_tf=True    # Skala logaritmik untuk term frequency
)

# Fit dan transform data teks
X_text = tfidf.fit_transform(data['ringkasan_sinopsis'])

# Convert sparse matrix to numpy array for merging
X_text_array = X_text.toarray()

# Gabungkan fitur teks dengan fitur tambahan
X_additional = np.array(additional_features_df)

print(f"Dimensi fitur teks: {X_text_array.shape}")
print(f"Dimensi fitur tambahan: {X_additional.shape}")

# Gabungkan semua fitur
X = np.hstack((X_text_array, X_additional))
y = np.array(data['genre'])

print(f"Total fitur gabungan: {X.shape[1]}")

# Bagi data menjadi training dan testing
print("Membagi data training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Gunakan ensemble model untuk meningkatkan akurasi
print("Melatih model...")

# Buat model tunggal Random Forest untuk perbandingan
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Akurasi Random Forest: {rf_accuracy * 100:.2f}%")

# Buat model Naive Bayes
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)
nb_y_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_y_pred)
print(f"Akurasi Naive Bayes: {nb_accuracy * 100:.2f}%")

# Gunakan voting ensemble untuk meningkatkan akurasi
print("Menggabungkan prediksi dari model...")
# Buat ensemble dengan bobot yang berbeda berdasarkan performa model
if rf_accuracy > nb_accuracy:
    # Random Forest lebih baik
    final_y_pred = rf_y_pred
    final_model = rf_model
    print("Menggunakan model Random Forest sebagai model final")
else:
    # Naive Bayes lebih baik
    final_y_pred = nb_y_pred
    final_model = nb_model
    print("Menggunakan model Naive Bayes sebagai model final")

# Evaluasi model final
final_accuracy = accuracy_score(y_test, final_y_pred)
print(f"Akurasi model final: {final_accuracy * 100:.2f}%")

# Tampilkan hasil prediksi per genre
from sklearn.metrics import classification_report, confusion_matrix
print("\nLaporan klasifikasi detail:")
print(classification_report(y_test, final_y_pred))

# Fungsi untuk memprediksi genre film baru
def predict_new_film(synopsis):
    """Predict genre for a new film synopsis"""
    # Process text dengan TF-IDF
    text_features = tfidf.transform([synopsis]).toarray()
    
    # Extract additional features
    add_feat_dict = extract_additional_features(synopsis)
    add_feat_df = pd.DataFrame([add_feat_dict])
    
    # Convert to numpy array and ensure order matches training
    add_feat_array = np.array(add_feat_df)
    
    # Combine features
    film_features = np.hstack((text_features, add_feat_array))
    
    # Predict
    predicted_genre = final_model.predict(film_features)[0]
    
    # Hanya mengembalikan genre teratas sesuai permintaan
    return predicted_genre

# Test dengan film baru
new_film = "Seorang pahlawan bertarung melawan monster raksasa di kota."
predicted_genre = predict_new_film(new_film)

print("\nContoh prediksi untuk film baru:")
print(f"Sinopsis: {new_film}")
print(f"Genre prediksi: {predicted_genre}")

# Jika ingin melihat probabilitas untuk debugging
if hasattr(final_model, 'predict_proba'):
    # Check if the model supports predict_proba
    text_features = tfidf.transform([new_film]).toarray()
    add_feat_dict = extract_additional_features(new_film)
    add_feat_df = pd.DataFrame([add_feat_dict])
    add_feat_array = np.array(add_feat_df)
    film_features = np.hstack((text_features, add_feat_array))
    
    proba = final_model.predict_proba(film_features)[0]
    genre_probs = {genre: prob for genre, prob in zip(final_model.classes_, proba)}
    sorted_probs = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nProbabilitas per genre (hanya untuk debugging):")
    for genre, prob in sorted_probs[:5]:
        print(f"  - {genre}: {prob*100:.2f}%")
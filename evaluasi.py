import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Path ke file CSV ===
csv_path = "file_audio_10detik.csv"  # Pastikan file ini ada di folder yang sama dengan evaluasi.py

# === Cek apakah file ada ===
if not os.path.exists(csv_path):
    print("❌ File CSV tidak ditemukan. Pastikan path sudah benar.")
    exit()

# === Load data ===
print("📥 Membaca data dari CSV...")
df = pd.read_csv(csv_path)

# === Ekstrak label genre dari path file ===
df['genre'] = df['file'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else 'unknown')

# === Pisahkan fitur dan label ===
X = df.drop(columns=['file', 'segment', 'genre'])
y = df['genre']

# === Split data ===
print("✂️ Membagi data menjadi training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Latih model ===
print("🧠 Melatih model Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Prediksi dan evaluasi ===
print("📊 Mengevaluasi model...")
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# === Simpan classification report ===
report_path = "classification_report.csv"
report_df.to_csv(report_path)
print(f"✅ Classification report disimpan sebagai {report_path}")
print(report_df)

# === Confusion matrix ===
print("🖼️ Membuat confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# === Simpan dan tampilkan grafik ===
conf_path = "confusion_matrix_final.png"
plt.tight_layout()
plt.savefig(conf_path)
plt.show()
print(f"✅ Grafik confusion matrix disimpan sebagai {conf_path}")

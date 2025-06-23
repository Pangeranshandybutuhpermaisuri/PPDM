import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Folder tempat file .wav berada
folder_path = "data/genres_original"
output_path = "PPDM/TUGAS_KELOMPOK_2/file_audio_10detik.csv"

# Buat folder output jika belum ada
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data = []

# Jelajahi semua file .wav dalam folder dan subfolder
for root, _, files in os.walk(folder_path):
    for filename in files:
        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(root, filename)
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            if duration < 30:
                print(f"⚠️ {filename} dilewati (durasi {duration:.2f} detik < 30 detik)")
                continue

            print(f"✅ Memproses {filename} ({duration:.2f} detik)")

            segment_duration = 3  # 3 detik
            samples_per_segment = int(segment_duration * sr)
            num_segments = 10

            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                if end > len(y):
                    break

                segment = y[start:end]

                # === Fitur Domain Waktu ===
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                rms = np.mean(librosa.feature.rms(y=segment))

                # === Fitur Domain Frekuensi ===
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                mfcc_mean = [np.mean(m) for m in mfcc]

                fitur = {
                    "file": os.path.relpath(file_path, folder_path),  # simpan nama file relatif
                    "segment": i + 1,
                    "zcr": zcr,
                    "rms": rms
                }

                for j, val in enumerate(mfcc_mean):
                    fitur[f"mfcc_{j+1}"] = val

                data.append(fitur)
                print(f"   → Segment {i+1} berhasil ditambahkan")

        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")

# Simpan ke CSV jika ada data
if data:
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Selesai! Hasil disimpan di '{output_path}'")
else:
    print("\n⚠️ Tidak ada data yang berhasil diproses. CSV tidak dibuat.")

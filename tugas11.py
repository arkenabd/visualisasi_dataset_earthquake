# --- Bagian A: Import Library yang Diperlukan ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE # Pastikan Anda sudah menginstal imbalanced-learn: pip install imbalanced-learn
import warnings

# Mengabaikan warning untuk tampilan yang lebih bersih
warnings.filterwarnings('ignore')

print("Library berhasil diimpor.")

# --- Bagian B: Pemuatan Data ---
try:
    df = pd.read_csv('earthquake_1995-2023.csv')
    print("Dataset berhasil dimuat.")
    print(f"Jumlah baris dan kolom: {df.shape}")
    print("\n5 baris pertama dataset:")
    print(df.head())
    print("\nInformasi tipe data kolom:")
    print(df.info())
except FileNotFoundError:
    print("Error: File 'earthquake_1995-2023.csv' tidak ditemukan. Pastikan file berada di direktori yang sama atau sesuaikan path.")
    exit() # Keluar jika file tidak ditemukan

# --- Bagian C: Preprocessing Data ---

print("\n--- Memulai Preprocessing Data ---")

# 1. Penanganan Missing Values
print("\nJumlah Missing Values per kolom sebelum penanganan:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Kolom yang mungkin kosong dan strategi penanganannya
# 'alert', 'cdi', 'mmi', 'dmin', 'gap', 'nst'
# Untuk nilai numerik, kita bisa mengisi dengan median karena lebih tahan terhadap outlier
numeric_cols_to_fill_median = ['cdi', 'mmi', 'dmin', 'gap', 'nst']
for col in numeric_cols_to_fill_median:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Untuk kolom kategorikal seperti 'alert', 'location', 'continent', 'country'
# Bisa diisi dengan 'Unknown' atau modus.
# Jika 'alert' sangat penting dan banyak missing, bisa juga dihapus barisnya.
# Untuk tujuan pemodelan, kita bisa mengisi dengan 'Unknown' untuk mempertahankan baris.
categorical_cols_to_fill_unknown = ['alert', 'location', 'continent', 'country']
for col in categorical_cols_to_fill_unknown:
    if col in df.columns:
        df[col].fillna('Unknown', inplace=True)

# Kolom 'title' tidak digunakan dalam pemodelan/visualisasi inti, bisa diabaikan missingnya
# atau dihapus jika tidak ada gunanya.

print("\nJumlah Missing Values per kolom setelah penanganan:")
print(df.isnull().sum()[df.isnull().sum() > 0]) # Cek lagi, seharusnya sebagian besar 0

# 2. Transformasi Data
# Mengubah 'date_time' ke format datetime
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

# Menghapus baris jika 'date_time' gagal dikonversi (NaN)
df.dropna(subset=['date_time'], inplace=True)

# Ekstraksi fitur berbasis waktu
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour

print("\nKolom waktu (year, month, day, hour) berhasil diekstrak.")

# --- Bagian D: Visualisasi Data Awal (Eksplorasi) ---
print("\n--- Memulai Visualisasi Data Awal ---")
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Grafik Batang: Distribusi Gempa Berdasarkan Benua
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='continent', order=df['continent'].value_counts().index, palette='viridis')
plt.title('Distribusi Gempa Berdasarkan Benua')
plt.xlabel('Benua')
plt.ylabel('Jumlah Gempa')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
print("Interpretasi: Grafik ini menunjukkan frekuensi gempa bumi di setiap benua. Benua [X] dan [Y] tampak memiliki jumlah gempa tertinggi, yang bisa jadi berhubungan dengan lokasi lempeng tektonik.")

# 2. Diagram Pie: Proporsi Potensi Tsunami
plt.figure(figsize=(8, 8))
df['tsunami'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'],
                                      labels=['Tidak Tsunami (0)', 'Tsunami (1)'])
plt.title('Proporsi Gempa yang Memicu Tsunami')
plt.ylabel('') # Hapus label y agar tidak tumpang tindih
plt.tight_layout()
plt.show()
print("Interpretasi: Diagram ini menunjukkan proporsi gempa yang memicu tsunami dan yang tidak. Mayoritas gempa ([X]%) tidak memicu tsunami, menggarisbawahi bahwa kejadian tsunami relatif jarang meskipun ada banyak gempa.")

# 3. Grafik Garis: Tren Magnitude Rata-rata Gempa per Tahun
# Filter data untuk tahun yang wajar
df_filtered_years = df[(df['year'] >= 2001) & (df['year'] <= 2023)]
avg_magnitude_per_year = df_filtered_years.groupby('year')['magnitude'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_magnitude_per_year, x='year', y='magnitude', marker='o', color='purple')
plt.title('Tren Magnitude Rata-rata Gempa per Tahun (2001-2023)')
plt.xlabel('Tahun')
plt.ylabel('Magnitude Rata-rata')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Interpretasi: Grafik ini menggambarkan tren magnitude rata-rata gempa dari tahun ke tahun. Terlihat [pola, misal: relatif stabil dengan fluktuasi kecil / ada peningkatan/penurunan di tahun tertentu].")

# 4. Scatter Plot: Hubungan antara Magnitude dan Significance (Sig)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='magnitude', y='sig', alpha=0.6, hue='tsunami', palette='coolwarm', s=100)
plt.title('Hubungan antara Magnitude dan Significance Event')
plt.xlabel('Magnitude')
plt.ylabel('Significance Event (sig)')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Interpretasi: Scatter plot ini menunjukkan korelasi positif yang kuat antara magnitude dan signifikansi. Gempa dengan magnitude lebih tinggi cenderung memiliki nilai sig yang lebih tinggi. Perhatikan juga bagaimana titik-titik 'tsunami' cenderung berada di area magnitude dan sig yang lebih tinggi.")

# 5. Heatmap: Matriks Korelasi Antar Variabel Numerik
numeric_cols = ['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']
# Hanya ambil kolom yang ada di DataFrame
numeric_cols_existing = [col for col in numeric_cols if col in df.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols_existing].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriks Korelasi Antar Variabel Numerik')
plt.tight_layout()
plt.show()
print("Interpretasi: Heatmap ini menunjukkan tingkat korelasi antar pasangan variabel numerik. Korelasi positif kuat terlihat antara `magnitude`, `cdi`, `mmi`, dan `sig`, yang wajar karena semuanya terkait dengan 'kekuatan' gempa.")

# 6. Boxplot: Distribusi Magnitude Berdasarkan Alert Level
plt.figure(figsize=(10, 6))
# Urutkan alert level secara logis jika memungkinkan
alert_order = ['green', 'yellow', 'orange', 'red', 'Unknown'] # Pastikan 'Unknown' ada jika diisi demikian
sns.boxplot(data=df, x='alert', y='magnitude', order=[level for level in alert_order if level in df['alert'].unique()], palette='viridis')
plt.title('Distribusi Magnitude Berdasarkan Alert Level')
plt.xlabel('Alert Level')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()
print("Interpretasi: Boxplot ini memperjelas bahwa gempa dengan level `alert` yang lebih tinggi (orange, red) memiliki distribusi magnitude yang lebih tinggi dan rentang yang lebih luas, konsisten dengan sistem peringatan bahaya.")

# --- Bagian E: Pemodelan Klasifikasi (Prediksi Tsunami) ---
print("\n--- Memulai Pemodelan Klasifikasi ---")

# 1. Pemilihan Fitur dan Variabel Target
# Variabel target
y = df['tsunami']

# Fitur yang akan digunakan (numerical dan categorical)
# Fitur yang relevan secara geofisika dan dari dataset
features = [
    'magnitude', 'depth', 'latitude', 'longitude', 'sig', 'cdi', 'mmi',
    'nst', 'dmin', 'gap', 'magType', 'net', 'continent', 'country'
]
X = df[features]

# Pisahkan fitur numerik dan kategorikal
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nFitur Numerik: {numerical_features}")
print(f"Fitur Kategorikal: {categorical_features}")

# 2. Pipeline untuk Preprocessing Fitur
# Menggunakan ColumnTransformer untuk preprocessing yang berbeda pada jenis fitur yang berbeda
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any, though not expected here)
)

# 3. Pembagian Data Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Stratify=y penting untuk memastikan proporsi kelas 'tsunami' sama di train dan test set.

print(f"\nUkuran data training (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Ukuran data testing (X_test, y_test): {X_test.shape}, {y_test.shape}")
print(f"Proporsi kelas tsunami di y_train: {y_train.value_counts(normalize=True)}")
print(f"Proporsi kelas tsunami di y_test: {y_test.value_counts(normalize=True)}")


# 4. Penanganan Imbalance Data (SMOTE)
print("\nMenangani Imbalance Data menggunakan SMOTE...")
# Menerapkan preprocessing pada data training sebelum SMOTE
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Konversi sparse matrix dari OneHotEncoder menjadi dense array jika perlu untuk SMOTE
if hasattr(X_train_preprocessed, "toarray"):
    X_train_preprocessed = X_train_preprocessed.toarray()

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

print(f"Ukuran data training setelah SMOTE (X_train_resampled, y_train_resampled): {X_train_resampled.shape}, {y_train_resampled.shape}")
print(f"Proporsi kelas tsunami di y_train_resampled: {y_train_resampled.value_counts(normalize=True)}")

# 5. Membangun dan Melatih Model Random Forest
# Menggunakan Pipeline untuk menggabungkan preprocessing dan model
model = Pipeline(steps=[
    # Preprocessor sudah diterapkan secara manual di atas untuk SMOTE
    # ('preprocessor', preprocessor), # Tidak perlu di pipeline jika SMOTE diterapkan di luar
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')) # class_weight bisa membantu juga
])

# Melatih model dengan data yang sudah di-resample
print("\nMelatih model Random Forest...")
model.fit(X_train_resampled, y_train_resampled)
print("Model Random Forest berhasil dilatih.")

# 6. Prediksi pada Data Testing
# Terapkan preprocessing pada data testing
X_test_preprocessed = preprocessor.transform(X_test)
if hasattr(X_test_preprocessed, "toarray"):
    X_test_preprocessed = X_test_preprocessed.toarray()

y_pred = model.predict(X_test_preprocessed)
y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1] # Probabilitas kelas positif (tsunami=1)

# --- Bagian F: Evaluasi Model dan Visualisasi Hasil Pemodelan ---
print("\n--- Evaluasi Model Klasifikasi ---")

# 1. Metrik Evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1) # Presisi untuk kelas tsunami (1)
recall = recall_score(y_test, y_pred, pos_label=1)     # Recall untuk kelas tsunami (1)
f1 = f1_score(y_test, y_pred, pos_label=1)           # F1-Score untuk kelas tsunami (1)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Tsunami): {precision:.4f}")
print(f"Recall (Tsunami): {recall:.4f}")
print(f"F1-Score (Tsunami): {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Tsunami', 'Tsunami'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
print("Interpretasi Confusion Matrix: Tunjukkan True Negatives (TN), False Positives (FP), False Negatives (FN), dan True Positives (TP). Fokus pada FN (tsunami yang tidak terdeteksi) dan FP (peringatan palsu).")
print(f"True Negatives (TN): {cm[0,0]} (Benar memprediksi tidak tsunami)")
print(f"False Positives (FP): {cm[0,1]} (Salah memprediksi tsunami, padahal tidak - Peringatan Palsu)")
print(f"False Negatives (FN): {cm[1,0]} (Salah memprediksi tidak tsunami, padahal tsunami - Gagal Deteksi)")
print(f"True Positives (TP): {cm[1,1]} (Benar memprediksi tsunami)")


# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Interpretasi ROC Curve: Kurva yang lebih mendekati sudut kiri atas menunjukkan kinerja model yang lebih baik dalam membedakan kelas. Nilai AUC yang tinggi ([X]) mengindikasikan kemampuan diskriminatif model yang sangat baik.")

# 4. Visualisasi Pentingnya Fitur (Feature Importance)
# Dapatkan nama fitur setelah OneHotEncoder
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

# Dapatkan feature importances dari classifier
if hasattr(model.named_steps['classifier'], 'feature_importances_'):
    importances = model.named_steps['classifier'].feature_importances_
    # Buat Series untuk memudahkan sorting
    feature_importances = pd.Series(importances, index=feature_names)

    # Urutkan dan ambil top N fitur
    top_n = 15 # Sesuaikan jumlah fitur yang ingin ditampilkan
    top_features = feature_importances.nlargest(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette='coolwarm')
    plt.title(f'Top {top_n} Feature Importance (Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    print("Interpretasi Feature Importance: Grafik ini menunjukkan fitur-fitur yang paling berkontribusi pada prediksi model. Terlihat bahwa [sebutkan fitur-fitur teratas, contoh: `magnitude`, `latitude`, `longitude`, `depth`] memiliki dampak paling besar dalam menentukan potensi tsunami.")
else:
    print("\nModel tidak memiliki atribut 'feature_importances_'.")


print("\n--- Analisis selesai ---")
print("Pastikan untuk mengisi interpretasi di setiap bagian grafik dan hasil pemodelan pada laporan Anda.")
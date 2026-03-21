# 🏦 Home Credit Default Risk — Credit Scoring Prediction

> Memaksimalkan potensi data untuk memprediksi kelayakan kredit secara akurat, memastikan pelanggan yang layak tidak ditolak, dan menyediakan struktur kredit yang mendukung keberhasilan pelunasan.

---

## 📌 Latar Belakang

Home Credit melayani segmen masyarakat yang memiliki keterbatasan akses terhadap layanan perbankan tradisional. Tantangan utamanya adalah menilai kelayakan kredit secara akurat tanpa riwayat kredit yang memadai.

Proyek ini bertujuan membangun model machine learning yang mampu:
- **Memprediksi kemungkinan default** (gagal bayar) seorang pemohon kredit
- **Meminimalkan penolakan** terhadap pemohon yang sebenarnya layak
- **Mengoptimalkan struktur kredit** agar sesuai dengan kemampuan bayar nasabah

---

## 🎯 Tujuan Proyek

| Tujuan | Deskripsi |
|--------|-----------|
| Prediksi Akurat | Membangun model klasifikasi biner untuk memprediksi status default (TARGET: 0 = tidak default, 1 = default) |
| Inklusivitas Kredit | Mengurangi false negative agar pemohon layak tidak salah ditolak |
| Evaluasi Mendalam | Menganalisis performa model di berbagai threshold keputusan |

---

## 📂 Dataset

Dataset berasal dari kompetisi Kaggle **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)**.

| File | Deskripsi |
|------|-----------|
| `application_train.csv` | Data training dengan label TARGET |
| `application_test.csv` | Data testing untuk prediksi akhir |

**Target Variable:**
- `0` → Tidak default (pembayaran lancar)
- `1` → Default (gagal bayar)

> ⚠️ Dataset sangat imbalanced: mayoritas data adalah kelas 0 (tidak default).

---

## 🔧 Teknologi & Library

```
Python 3.x
├── pandas, numpy          → Manipulasi data
├── matplotlib, seaborn    → Visualisasi
├── scikit-learn           → Preprocessing & Logistic Regression
├── xgboost                → XGBoost Classifier
└── imbalanced-learn       → SMOTEENN (class balancing)
```

---

## 🔄 Alur Pipeline

```
Raw Data
   ↓
1. Exploratory Data Analysis (EDA)
   - Analisis missing values
   - Distribusi target variable
   - Analisis fitur kategorik vs target
   - Analisis Income vs Credit
   ↓
2. Preprocessing
   - Drop kolom dengan missing values > 30%
   - Konversi tipe data (kategorikal)
   - Transformasi kolom DAYS → YEARS
   - Label Encoding (kategorikal)
   - Median Imputation (numerik)
   - Outlier Removal (IQR, threshold = 1.5)
   - RobustScaler (scaling)
   ↓
3. Class Balancing
   - SMOTEENN (sampling_strategy = 0.5)
   ↓
4. Pelatihan Model
   - XGBoost Classifier
   - Logistic Regression
   ↓
5. Evaluasi & Threshold Tuning
   - Threshold: 0.1 s/d 0.9
   - Metrics: Precision, Recall, F1, Specificity, AUC-ROC
   ↓
6. Prediksi Data Test
```

---

## ⚙️ Detail Preprocessing

### Penanganan Missing Values
- Kolom dengan **> 30% missing values** dihapus
- Kolom numerik diimputasi menggunakan **median**

### Feature Engineering
Kolom bertipe `DAYS_*` dikonversi menjadi satuan tahun (`_YEARS`) untuk interpretasi yang lebih intuitif:
- `DAYS_BIRTH` → `DAYS_BIRTH_YEARS`
- `DAYS_EMPLOYED` → `DAYS_EMPLOYED_YEARS`
- `DAYS_REGISTRATION` → `DAYS_REGISTRATION_YEARS`
- `DAYS_ID_PUBLISH` → `DAYS_ID_PUBLISH_YEARS`
- `DAYS_LAST_PHONE_CHANGE` → `DAYS_LAST_PHONE_CHANGE_YEARS`

### Outlier Handling
Menggunakan metode **IQR (Interquartile Range)** dengan threshold 1.5:
```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```
Data di luar batas tersebut dihapus dari training set.

### Class Balancing
**SMOTEENN** (kombinasi SMOTE oversampling + Edited Nearest Neighbors undersampling) digunakan untuk menangani ketidakseimbangan kelas dengan `sampling_strategy = 0.5`.

---

## 🤖 Model

### 1. XGBoost Classifier

```python
XGBClassifier(
    n_estimators      = 1000,
    max_depth         = 6,
    min_child_weight  = 5,
    subsample         = 0.85,
    scale_pos_weight  = 20,
    random_state      = 42,
    eval_metric       = "logloss",
    tree_method       = 'auto'
)
```

### 2. Logistic Regression

```python
LogisticRegression(
    class_weight = 'balanced',
    max_iter     = 1000,
    random_state = 42,
    penalty      = 'l2',
    C            = 1.0,
    solver       = 'liblinear'
)
```

---

## 📊 Hasil Evaluasi

### XGBoost — Performa per Threshold

| Threshold | Precision | Recall | F1 Score | Specificity | Accuracy |
|-----------|-----------|--------|----------|-------------|----------|
| 0.1 | 0.132 | 0.687 | 0.221 | 0.603 | 0.61 |
| 0.2 | 0.146 | 0.600 | 0.234 | 0.692 | 0.68 |
| 0.3 | 0.158 | 0.540 | 0.244 | 0.748 | 0.73 |
| 0.4 | 0.168 | 0.484 | 0.250 | 0.789 | 0.77 |
| 0.5 | 0.180 | 0.437 | 0.255 | 0.825 | 0.79 |
| 0.6 | 0.191 | 0.388 | 0.256 | 0.856 | 0.82 |
| **0.7** ✅ | **0.207** | **0.336** | **0.256** | **0.887** | **0.84** |
| 0.8 | 0.228 | 0.275 | 0.250 | 0.918 | 0.87 |
| 0.9 | 0.259 | 0.194 | 0.222 | 0.951 | 0.89 |

> ✅ **Best Threshold XGBoost: 0.70** (F1 tertinggi = 0.256)

---

### Logistic Regression — Performa per Threshold

| Threshold | Precision | Recall | F1 Score | Specificity | Accuracy |
|-----------|-----------|--------|----------|-------------|----------|
| 0.1 | 0.106 | 0.785 | 0.187 | 0.418 | 0.45 |
| 0.2 | 0.121 | 0.698 | 0.206 | 0.555 | 0.57 |
| 0.3 | 0.135 | 0.613 | 0.221 | 0.657 | 0.65 |
| 0.4 | 0.150 | 0.529 | 0.233 | 0.737 | 0.72 |
| **0.5** ✅ | **0.167** | **0.449** | **0.243** | **0.803** | **0.77** |
| 0.6 | 0.183 | 0.360 | 0.242 | 0.859 | 0.82 |
| 0.7 | 0.205 | 0.275 | 0.235 | 0.907 | 0.86 |
| 0.8 | 0.237 | 0.178 | 0.204 | 0.949 | 0.89 |
| 0.9 | 0.273 | 0.073 | 0.115 | 0.983 | 0.91 |

> ✅ **Best Threshold Logistic Regression: 0.50** (F1 tertinggi = 0.243)

---

### Perbandingan Model

| Metrik | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| Best Threshold | 0.70 | 0.50 |
| Precision | 0.207 | 0.167 |
| Recall | 0.336 | 0.449 |
| **F1 Score** | **0.256** | **0.243** |
| Specificity | 0.887 | 0.803 |

> 🏆 **XGBoost** menghasilkan F1 Score lebih tinggi dengan presisi lebih baik. **Logistic Regression** memiliki recall lebih tinggi, yang berarti lebih sensitif dalam mendeteksi potensi default.

---

## 💡 Insight & Kesimpulan

1. **Imbalanced Class** adalah tantangan utama — hanya ~8% data adalah kelas default (1). SMOTEENN membantu, namun F1 pada kelas minoritas tetap rendah (~0.25).

2. **Trade-off Threshold**: Threshold rendah meningkatkan recall (lebih banyak default terdeteksi) namun menurunkan presisi (lebih banyak false alarm). Threshold tinggi sebaliknya.

3. **XGBoost vs Logistic Regression**: XGBoost lebih unggul secara keseluruhan — lebih baik dalam precision dan F1, sementara Logistic Regression lebih agresif dalam mendeteksi kasus default (recall lebih tinggi).

4. **Konteks Bisnis**: Dari perspektif Home Credit, **recall yang tinggi lebih diprioritaskan** agar tidak melewatkan nasabah yang berpotensi gagal bayar, meskipun harus menerima lebih banyak false positive.

---

## 📁 Struktur File

```
├── Sintaks.ipynb           # Notebook utama (XGBoost + Logistic Regression)
├── application_train.csv   # Dataset training
├── application_test.csv    # Dataset testing
└── README.md               # Dokumentasi proyek
```

---

## 👤 Author

Proyek ini dikerjakan sebagai bagian dari Program **Rakamin × Home Credit Indonesia (Project-Based Internship)**.

---

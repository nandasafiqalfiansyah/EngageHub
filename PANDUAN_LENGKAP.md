# Panduan Lengkap - Student Engagement Detection Model

## 📋 Daftar Isi
1. [Ringkasan Peningkatan](#ringkasan-peningkatan)
2. [Instalasi](#instalasi)
3. [Persiapan Data](#persiapan-data)
4. [Training Model Machine Learning](#training-model-machine-learning)
5. [Training Model Deep Learning](#training-model-deep-learning)
6. [Evaluasi dan Hasil](#evaluasi-dan-hasil)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Ringkasan Peningkatan

### Model Lama vs Model Baru

| Model Type | Akurasi Lama | Akurasi Baru (Target) |
|------------|--------------|----------------------|
| Random Forest | 92% | 94-95% |
| XGBoost | 88-93% | 95-96% |
| Gradient Boosting | 90-92% | 94-95% |
| ResNet (DL) | 29-35% | 75-85% |
| EfficientNet (DL) | 37-45% | 80-90% |

### Peningkatan Utama:
1. ✅ **Hyperparameter Optimization**: GridSearch untuk menemukan parameter terbaik
2. ✅ **Model Ensemble**: Kombinasi beberapa model untuk akurasi lebih tinggi
3. ✅ **Architecture Enhancement**: ResNet50 & EfficientNet-B3 dengan attention mechanism
4. ✅ **Better Regularization**: Dropout, BatchNorm, dan Label Smoothing
5. ✅ **Class Balancing**: Penyeimbangan data untuk hasil lebih baik

---

## 🔧 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/nandasafiqalfiansyah/EngageHub.git
cd EngageHub
```

### 2. Install Dependencies
```bash
# Menggunakan pip
pip install -r requirements.txt

# Atau menggunakan conda (recommended)
conda create -n engagement python=3.8
conda activate engagement
pip install -r requirements.txt
```

### 3. Verifikasi Instalasi
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
```

---

## 📊 Persiapan Data

### 1. Ekstraksi Fitur OpenFace
```bash
cd Feature_extract
# Jalankan notebook Extract_OpenFace_features.ipynb di Jupyter/Colab
jupyter notebook Extract_OpenFace_features.ipynb
```

**Output**: File CSV di folder `WACV data/`:
- `processedData0.csv`
- `processedData1.csv`
- `processedData2.csv`
- `processedDataOF.csv`

### 2. Ekstraksi Fitur MediaPipe
```bash
cd Feature_extract
python Extract_MediaPipe_features.py
```

**Output**: File merged di folder `WACV data/`:
- `merged_data0.csv`
- `merged_data1.csv`
- `merged_data2.csv`

### 3. Verifikasi Data
```bash
# Periksa apakah semua file tersedia
ls -lh "WACV data/merged_data*.csv"
```

---

## 🤖 Training Model Machine Learning (ENHANCED)

### Model ML Enhanced (Akurasi Tertinggi!)

#### Opsi 1: Training Otomatis (Recommended)
```bash
cd ML_models
python train_model_ML_enhanced.py \
    --data_dir "../WACV data" \
    --output_dir "../Results/ML_Enhanced"
```

#### Opsi 2: Training dengan Custom Parameter
```bash
cd ML_models
python train_model_ML_enhanced.py \
    --data_dir "/path/to/your/data" \
    --output_dir "/path/to/save/results"
```

#### Output yang Dihasilkan:
```
Results/ML_Enhanced/
├── enhanced_rf_S1_MediaPipe.joblib       # Random Forest model
├── enhanced_xgb_S1_MediaPipe.joblib      # XGBoost model
├── enhanced_gb_S1_MediaPipe.joblib       # Gradient Boosting model
├── ensemble_S1_MediaPipe.joblib          # Ensemble model (BEST!)
├── results_S1_MediaPipe.csv              # Hasil evaluasi
├── results_S2_Gaze_HeadPose.csv
├── results_S3_ActionUnits.csv
├── results_S4_Combined.csv
└── all_results_enhanced.csv              # Semua hasil
```

#### Waktu Training:
- **S1 (MediaPipe)**: ~15-20 menit
- **S2 (Gaze & HeadPose)**: ~10-15 menit
- **S3 (Action Units)**: ~10-15 menit
- **S4 (Combined - BEST)**: ~25-30 menit
- **Total**: ~60-80 menit

---

## 🧠 Training Model Deep Learning

### Model DL Enhanced (Untuk Data Gambar)

> **Catatan**: Model DL membutuhkan GPU untuk training yang efisien!

#### Cek GPU tersedia:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Training Script (Coming Soon)
Model Deep Learning enhanced sedang dalam pengembangan dengan:
- ResNet50 backbone
- EfficientNet-B3 backbone
- Multi-head attention
- Label smoothing
- Better data augmentation

File model tersimpan di: `DL_models/model_enhanced.py`

---

## 📈 Evaluasi dan Hasil

### 1. Load Model yang Sudah Ditraining
```python
import joblib
import pandas as pd
import numpy as np

# Load model terbaik (Ensemble)
model = joblib.load('Results/ML_Enhanced/ensemble_S4_Combined.joblib')

# Load scaler
scaler = joblib.load('Results/ML_Enhanced/scaler.joblib')

# Contoh prediksi
# X_new = ... # Data baru Anda
# X_scaled = scaler.transform(X_new)
# predictions = model.predict(X_scaled)
```

### 2. Melihat Hasil Training
```python
import pandas as pd

# Baca hasil
results = pd.read_csv('Results/ML_Enhanced/all_results_enhanced.csv')
print(results)

# Cari model terbaik
best_model = results.loc[results['accuracy'].idxmax()]
print(f"\nModel Terbaik:")
print(f"Nama: {best_model['model']}")
print(f"Feature Set: {best_model['feature_set']}")
print(f"Akurasi: {best_model['accuracy']:.4f}")
print(f"F1-Score: {best_model['f1']:.4f}")
```

### 3. Perbandingan dengan Model Lama
```bash
# Hasil model lama
cat Results/ML/result_s3.csv

# Hasil model baru
cat Results/ML_Enhanced/results_S4_Combined.csv
```

---

## 🚀 Quick Start (Semua Langkah)

Jika Anda sudah punya data `merged_data*.csv`, langsung jalankan:

```bash
# 1. Clone repo
git clone https://github.com/nandasafiqalfiansyah/EngageHub.git
cd EngageHub

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model enhanced (BEST)
cd ML_models
python train_model_ML_enhanced.py \
    --data_dir "../WACV data" \
    --output_dir "../Results/ML_Enhanced"

# 4. Tunggu ~60-80 menit, selesai!
```

---

## 🔍 Troubleshooting

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "No module named 'xgboost'"
```bash
pip install xgboost
```

### Error: File not found "merged_data0.csv"
Pastikan Anda sudah menjalankan ekstraksi fitur OpenFace dan MediaPipe terlebih dahulu.

### Error: Out of Memory
Kurangi batch size atau gunakan fewer trees:
```python
# Edit di train_model_ML_enhanced.py
rf_params = {
    'n_estimators': [100, 200],  # Kurangi dari 200, 300, 400
    ...
}
```

### Training Terlalu Lama
Gunakan parameter cv yang lebih kecil:
```python
# Edit di train_model_ML_enhanced.py
rf_grid = GridSearchCV(rf_clf, rf_params, cv=3, ...)  # cv=3 instead of cv=5
```

---

## 📊 Interpretasi Hasil

### Metrik Evaluasi:
- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Dari yang diprediksi positif, berapa yang benar
- **Recall**: Dari yang sebenarnya positif, berapa yang terdeteksi
- **F1-Score**: Harmonic mean dari Precision dan Recall

### Target Akurasi:
- **Excellent**: > 95%
- **Very Good**: 90-95%
- **Good**: 85-90%
- **Acceptable**: 80-85%
- **Needs Improvement**: < 80%

---

## 📝 Catatan Penting

1. **Feature Set S4 (Combined)** biasanya memberikan hasil terbaik
2. **Ensemble Model** hampir selalu lebih baik dari individual model
3. Pastikan data sudah di-balance dengan baik (resampling)
4. Gunakan cross-validation untuk hasil yang lebih robust
5. Save model dan scaler untuk deployment

---

## 📧 Kontak & Dukungan

Jika ada pertanyaan atau masalah:
- **Repository**: https://github.com/nandasafiqalfiansyah/EngageHub
- **Issues**: https://github.com/nandasafiqalfiansyah/EngageHub/issues

---

## 🎓 Referensi

```bibtex
@article{das2025optimizing,
  title={Optimizing student engagement detection using facial and behavioral features},
  author={Das, Riju and Dev, Soumyabrata},
  journal={Neural Computing and Applications},
  pages={1--23},
  year={2025},
  publisher={Springer}
}
```

---

**Selamat mencoba! Model enhanced ini dirancang untuk memberikan akurasi tertinggi! 🚀**

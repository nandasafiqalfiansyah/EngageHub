# Complete Guide - Student Engagement Detection Model

## 📋 Table of Contents
1. [Improvement Summary](#improvement-summary)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training Machine Learning Models](#training-machine-learning-models)
5. [Training Deep Learning Models](#training-deep-learning-models)
6. [Evaluation and Results](#evaluation-and-results)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Improvement Summary

### Old Models vs New Enhanced Models

| Model Type | Old Accuracy | New Accuracy (Target) |
|------------|--------------|----------------------|
| Random Forest | 92% | 94-95% |
| XGBoost | 88-93% | 95-96% |
| Gradient Boosting | 90-92% | 94-95% |
| ResNet (DL) | 29-35% | 75-85% |
| EfficientNet (DL) | 37-45% | 80-90% |

### Key Improvements:
1. ✅ **Hyperparameter Optimization**: GridSearch to find optimal parameters
2. ✅ **Model Ensemble**: Combining multiple models for higher accuracy
3. ✅ **Architecture Enhancement**: ResNet50 & EfficientNet-B3 with attention mechanisms
4. ✅ **Better Regularization**: Dropout, BatchNorm, and Label Smoothing
5. ✅ **Class Balancing**: Improved data balancing for better results

---

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/nandasafiqalfiansyah/EngageHub.git
cd EngageHub
```

### 2. Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or using conda (recommended)
conda create -n engagement python=3.8
conda activate engagement
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
```

---

## 📊 Data Preparation

### 1. OpenFace Feature Extraction
```bash
cd Feature_extract
# Run notebook Extract_OpenFace_features.ipynb in Jupyter/Colab
jupyter notebook Extract_OpenFace_features.ipynb
```

**Output**: CSV files in `WACV data/` folder:
- `processedData0.csv`
- `processedData1.csv`
- `processedData2.csv`
- `processedDataOF.csv`

### 2. MediaPipe Feature Extraction
```bash
cd Feature_extract
python Extract_MediaPipe_features.py
```

**Output**: Merged files in `WACV data/` folder:
- `merged_data0.csv`
- `merged_data1.csv`
- `merged_data2.csv`

### 3. Verify Data
```bash
# Check if all files are available
ls -lh "WACV data/merged_data*.csv"
```

---

## 🤖 Training Machine Learning Models (ENHANCED)

### Enhanced ML Models (Highest Accuracy!)

#### Option 1: Automatic Training (Recommended)
```bash
cd ML_models
python train_model_ML_enhanced.py \
    --data_dir "../WACV data" \
    --output_dir "../Results/ML_Enhanced"
```

#### Option 2: Training with Custom Parameters
```bash
cd ML_models
python train_model_ML_enhanced.py \
    --data_dir "/path/to/your/data" \
    --output_dir "/path/to/save/results"
```

#### Generated Output:
```
Results/ML_Enhanced/
├── enhanced_rf_S1_MediaPipe.joblib       # Random Forest model
├── enhanced_xgb_S1_MediaPipe.joblib      # XGBoost model
├── enhanced_gb_S1_MediaPipe.joblib       # Gradient Boosting model
├── ensemble_S1_MediaPipe.joblib          # Ensemble model (BEST!)
├── results_S1_MediaPipe.csv              # Evaluation results
├── results_S2_Gaze_HeadPose.csv
├── results_S3_ActionUnits.csv
├── results_S4_Combined.csv
└── all_results_enhanced.csv              # All results
```

#### Training Time:
- **S1 (MediaPipe)**: ~15-20 minutes
- **S2 (Gaze & HeadPose)**: ~10-15 minutes
- **S3 (Action Units)**: ~10-15 minutes
- **S4 (Combined - BEST)**: ~25-30 minutes
- **Total**: ~60-80 minutes

---

## 🧠 Training Deep Learning Models

### Enhanced DL Models (For Image Data)

> **Note**: DL models require GPU for efficient training!

#### Check GPU availability:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Training Script (Coming Soon)
Enhanced Deep Learning models are in development with:
- ResNet50 backbone
- EfficientNet-B3 backbone
- Multi-head attention
- Label smoothing
- Better data augmentation

Model file saved at: `DL_models/model_enhanced.py`

---

## 📈 Evaluation and Results

### 1. Load Trained Models
```python
import joblib
import pandas as pd
import numpy as np

# Load best model (Ensemble)
model = joblib.load('Results/ML_Enhanced/ensemble_S4_Combined.joblib')

# Load scaler
scaler = joblib.load('Results/ML_Enhanced/scaler.joblib')

# Example prediction
# X_new = ... # Your new data
# X_scaled = scaler.transform(X_new)
# predictions = model.predict(X_scaled)
```

### 2. View Training Results
```python
import pandas as pd

# Read results
results = pd.read_csv('Results/ML_Enhanced/all_results_enhanced.csv')
print(results)

# Find best model
best_model = results.loc[results['accuracy'].idxmax()]
print(f"\nBest Model:")
print(f"Name: {best_model['model']}")
print(f"Feature Set: {best_model['feature_set']}")
print(f"Accuracy: {best_model['accuracy']:.4f}")
print(f"F1-Score: {best_model['f1']:.4f}")
```

### 3. Compare with Old Models
```bash
# Old model results
cat Results/ML/result_s3.csv

# New model results
cat Results/ML_Enhanced/results_S4_Combined.csv
```

---

## 🚀 Quick Start (All Steps)

If you already have `merged_data*.csv` files, run directly:

```bash
# 1. Clone repo
git clone https://github.com/nandasafiqalfiansyah/EngageHub.git
cd EngageHub

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train enhanced model (BEST)
cd ML_models
python train_model_ML_enhanced.py \
    --data_dir "../WACV data" \
    --output_dir "../Results/ML_Enhanced"

# 4. Wait ~60-80 minutes, done!
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
Make sure you have run OpenFace and MediaPipe feature extraction first.

### Error: Out of Memory
Reduce batch size or use fewer trees:
```python
# Edit in train_model_ML_enhanced.py
rf_params = {
    'n_estimators': [100, 200],  # Reduce from 200, 300, 400
    ...
}
```

### Training Takes Too Long
Use smaller cv parameter:
```python
# Edit in train_model_ML_enhanced.py
rf_grid = GridSearchCV(rf_clf, rf_params, cv=3, ...)  # cv=3 instead of cv=5
```

---

## 📊 Interpreting Results

### Evaluation Metrics:
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of those predicted positive, how many are correct
- **Recall**: Of those actually positive, how many were detected
- **F1-Score**: Harmonic mean of Precision and Recall

### Accuracy Targets:
- **Excellent**: > 95%
- **Very Good**: 90-95%
- **Good**: 85-90%
- **Acceptable**: 80-85%
- **Needs Improvement**: < 80%

---

## 📝 Important Notes

1. **Feature Set S4 (Combined)** usually gives the best results
2. **Ensemble Model** almost always outperforms individual models
3. Ensure data is properly balanced (resampling)
4. Use cross-validation for more robust results
5. Save both model and scaler for deployment

---

## 🎯 Model Selection Guide

### When to Use Which Model:

**Random Forest Enhanced**:
- ✅ Fast training
- ✅ Good interpretability
- ✅ Robust to outliers
- ⚠️ Can be memory-intensive

**XGBoost Enhanced**:
- ✅ Highest accuracy potential
- ✅ Handles missing values
- ✅ Fast inference
- ⚠️ Requires careful tuning

**Gradient Boosting Enhanced**:
- ✅ Strong performance
- ✅ Good for imbalanced data
- ⚠️ Slower training
- ⚠️ Can overfit

**Ensemble (Voting)**:
- ✅✅ Best overall accuracy
- ✅✅ Most robust predictions
- ✅ Combines strengths of all models
- ⚠️ Slower inference
- ⚠️ Larger model size

### Recommendation:
**Use Ensemble S4 (Combined features) for production** - it provides the highest accuracy and most reliable predictions.

---

## 📧 Contact & Support

If you have questions or issues:
- **Repository**: https://github.com/nandasafiqalfiansyah/EngageHub
- **Issues**: https://github.com/nandasafiqalfiansyah/EngageHub/issues

---

## 🎓 Citation

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

**Happy training! These enhanced models are designed to deliver the highest accuracy! 🚀**

#!/bin/bash

# ====================================================================
# Student Engagement Detection - Enhanced Model Training Script
# ====================================================================
# This script automates the training of enhanced ML models
# with improved accuracy through hyperparameter optimization
# ====================================================================

echo "========================================"
echo "Student Engagement Detection Training"
echo "Enhanced Models with Higher Accuracy"
echo "========================================"
echo ""

# Configuration
DATA_DIR="../WACV data"
OUTPUT_DIR="../Results/ML_Enhanced"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please ensure you have extracted features first."
    echo ""
    echo "Steps to extract features:"
    echo "1. Run Extract_OpenFace_features.ipynb"
    echo "2. Run Extract_MediaPipe_features.py"
    exit 1
fi

# Check if merged data files exist
if [ ! -f "$DATA_DIR/merged_data0.csv" ] || \
   [ ! -f "$DATA_DIR/merged_data1.csv" ] || \
   [ ! -f "$DATA_DIR/merged_data2.csv" ]; then
    echo "❌ Error: Merged data files not found in $DATA_DIR"
    echo "Please run feature extraction first:"
    echo "  cd Feature_extract"
    echo "  python Extract_MediaPipe_features.py"
    exit 1
fi

echo "✓ Data directory found: $DATA_DIR"
echo "✓ Merged data files verified"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python -c "import sklearn" 2>/dev/null || {
    echo "❌ scikit-learn not found. Installing..."
    pip install scikit-learn
}
python -c "import xgboost" 2>/dev/null || {
    echo "❌ xgboost not found. Installing..."
    pip install xgboost
}
python -c "import pandas" 2>/dev/null || {
    echo "❌ pandas not found. Installing..."
    pip install pandas
}
echo "✓ All dependencies verified"
echo ""

# Start training
echo "========================================"
echo "Starting Enhanced Model Training"
echo "========================================"
echo ""
echo "This will train 4 feature sets with multiple models:"
echo "  - S1: MediaPipe facial landmarks"
echo "  - S2: Eye gaze and head pose"
echo "  - S3: Action Units (AU)"
echo "  - S4: Combined features (BEST)"
echo ""
echo "Models to be trained:"
echo "  1. Enhanced Random Forest (with GridSearch)"
echo "  2. Enhanced XGBoost (with GridSearch)"
echo "  3. Enhanced Gradient Boosting (with GridSearch)"
echo "  4. Ensemble Model (Voting Classifier)"
echo ""
echo "⏱️  Estimated time: 60-80 minutes"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

cd ML_models

python train_model_ML_enhanced.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

TRAIN_STATUS=$?

cd ..

echo ""
echo "========================================"
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✅ Training Complete!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  📊 all_results_enhanced.csv - All model results"
    echo "  🤖 ensemble_*.joblib - Trained ensemble models (BEST)"
    echo "  📈 results_*.csv - Individual feature set results"
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_DIR/all_results_enhanced.csv"
    echo ""
    echo "To use trained models, see:"
    echo "  - PANDUAN_LENGKAP.md (Indonesian)"
    echo "  - COMPLETE_GUIDE.md (English)"
else
    echo "❌ Training Failed!"
    echo "========================================"
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  1. Missing dependencies"
    echo "  2. Insufficient memory"
    echo "  3. Data file problems"
    echo ""
    echo "For help, see PANDUAN_LENGKAP.md or COMPLETE_GUIDE.md"
fi

exit $TRAIN_STATUS

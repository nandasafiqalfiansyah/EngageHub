"""
Engagement Prediction Script
Use trained models to predict student engagement on new data
"""

import argparse
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_model_and_scaler(model_path, scaler_path):
    """Load trained model and scaler"""
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict(model, scaler, data_path, feature_set='S4'):
    """
    Make predictions on new data
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        data_path: Path to CSV file with features
        feature_set: Which feature set to use (S1, S2, S3, S4)
    """
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Data shape: {df.shape}")
    
    # Extract features based on feature set
    if feature_set == 'S1':
        # MediaPipe facial landmarks
        if 'x0' not in df.columns or 'y467' not in df.columns:
            raise ValueError("MediaPipe landmark columns not found. Expected 'x0' through 'y467'.")
        X = df.loc[:, 'x0':'y467'].values
        print("Using S1: MediaPipe facial landmarks (468 landmarks)")
    elif feature_set == 'S2':
        # Gaze and head pose
        gaze_cols = [col for col in df.columns if 'gaze' in col.lower()]
        pose_cols = [col for col in df.columns if 'pose' in col.lower()]
        if not gaze_cols and not pose_cols:
            raise ValueError("No gaze or pose columns found. Expected columns containing 'gaze' or 'pose'.")
        X = df[gaze_cols + pose_cols].values
        print(f"Using S2: Gaze and Head Pose ({len(gaze_cols + pose_cols)} features)")
    elif feature_set == 'S3':
        # Action Units
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        if not au_cols:
            raise ValueError("No Action Unit columns found. Expected columns like 'AU01_r', 'AU02_r', etc.")
        X = df[au_cols].values
        print(f"Using S3: Action Units ({len(au_cols)} features)")
    elif feature_set == 'S4':
        # Combined features
        if 'x0' not in df.columns or 'AU45_c' not in df.columns:
            raise ValueError("Combined feature columns not found. Expected 'x0' through 'AU45_c'.")
        X = df.loc[:, "x0":"AU45_c"].values
        print(f"Using S4: Combined features ({X.shape[1]} features)")
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    
    # Scale features
    print("\nScaling features...")
    X_scaled = scaler.transform(X)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Prediction': predictions,
        'Prob_Class_0': probabilities[:, 0],
        'Prob_Class_1': probabilities[:, 1],
        'Prob_Class_2': probabilities[:, 2],
        'Confidence': np.max(probabilities, axis=1)
    })
    
    # Add engagement labels
    engagement_labels = {
        0: 'Disengaged',
        1: 'Neutral',
        2: 'Engaged'
    }
    results['Engagement_Level'] = results['Prediction'].map(engagement_labels)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Predict student engagement using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict using ensemble model on combined features
  python predict_engagement.py \\
    --model Results/ML_Enhanced/ensemble_S4_Combined.joblib \\
    --scaler Results/ML_Enhanced/scaler.joblib \\
    --data test_data.csv \\
    --feature_set S4 \\
    --output predictions.csv
  
  # Predict using XGBoost on Action Units only
  python predict_engagement.py \\
    --model Results/ML_Enhanced/enhanced_xgb_S3_ActionUnits.joblib \\
    --scaler Results/ML_Enhanced/scaler.joblib \\
    --data test_data.csv \\
    --feature_set S3 \\
    --output predictions.csv
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.joblib file)')
    parser.add_argument('--scaler', type=str, required=True,
                        help='Path to fitted scaler (.joblib file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV file with features to predict')
    parser.add_argument('--feature_set', type=str, default='S4',
                        choices=['S1', 'S2', 'S3', 'S4'],
                        help='Feature set to use (default: S4)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save predictions (default: predictions.csv)')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.scaler):
        print(f"❌ Error: Scaler file not found: {args.scaler}")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        return
    
    print("="*60)
    print("Student Engagement Prediction")
    print("="*60)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    
    # Make predictions
    results = predict(model, scaler, args.data, args.feature_set)
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"\n✅ Predictions saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    print(f"\nTotal samples: {len(results)}")
    print("\nEngagement Distribution:")
    print(results['Engagement_Level'].value_counts().to_string())
    
    print("\nAverage Confidence:")
    for level in ['Disengaged', 'Neutral', 'Engaged']:
        subset = results[results['Engagement_Level'] == level]
        if len(subset) > 0:
            avg_conf = subset['Confidence'].mean()
            print(f"  {level}: {avg_conf:.3f}")
    
    print("\nFirst 10 predictions:")
    print(results[['Engagement_Level', 'Confidence']].head(10).to_string(index=False))
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()

"""
Enhanced ML Model Training with Hyperparameter Optimization
This script provides improved accuracy through:
1. Better hyperparameter tuning
2. Feature engineering
3. Cross-validation
4. Ensemble methods
"""

from ML_classification import ML_classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import joblib
import argparse

def scalingDF(df, scaler_type='minmax'):
    """Scale dataframe using MinMax or Standard scaler"""
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df_s = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_s, scaler

def load_and_prepare_data(data_path):
    """Load and prepare data with class balancing"""
    print("Loading data...")
    df0 = pd.read_csv(os.path.join(data_path, "merged_data0.csv"))
    df1 = pd.read_csv(os.path.join(data_path, "merged_data1.csv"))
    df2 = pd.read_csv(os.path.join(data_path, "merged_data2.csv"))
    
    # Filter by confidence
    df00 = df0.loc[df0['confidence'] >= 0.7]
    df11 = df1.loc[df1['confidence'] >= 0.7]
    df22 = df2.loc[df2['confidence'] >= 0.7]
    
    # Resampling for class imbalance
    n = max(len(df00), len(df11), len(df22))  # Using max for better training
    df0_ds = resample(df00, replace=True, n_samples=n, random_state=42)
    df1_ds = resample(df11, replace=True, n_samples=n, random_state=42)
    df2_ds = resample(df22, replace=True, n_samples=n, random_state=42)
    
    df = pd.concat([df0_ds, df1_ds, df2_ds])
    df = df.sample(frac=1, random_state=42)
    
    df_x = df.loc[:, "x0":"AU45_c"]
    df_y = df.loc[:, "Label_y"]
    
    df_scaled, scaler = scalingDF(df_x)
    df_final = pd.concat([df_scaled, df_y], axis=1)
    
    return df_final, scaler

def train_enhanced_model(training_data, feature_set_name, output_dir):
    """Train enhanced models with optimized hyperparameters"""
    print(f"\n{'='*60}")
    print(f"Training Enhanced Models for {feature_set_name}")
    print(f"{'='*60}")
    
    X_train = training_data['X_train']
    Y_train = training_data['Y_train'].ravel()
    X_test = training_data['X_test']
    Y_test = training_data['Y_test'].ravel()
    
    results = []
    
    # 1. Enhanced Random Forest with GridSearch
    print("\n1. Training Enhanced Random Forest...")
    rf_params = {
        'n_estimators': [200, 300, 400],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=1)
    rf_grid = GridSearchCV(rf_clf, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, Y_train)
    
    best_rf = rf_grid.best_estimator_
    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(Y_test, rf_pred)
    
    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    results.append({
        'model': 'Enhanced Random Forest',
        'accuracy': rf_acc,
        'precision': precision_score(Y_test, rf_pred, average='weighted'),
        'recall': recall_score(Y_test, rf_pred, average='weighted'),
        'f1': f1_score(Y_test, rf_pred, average='weighted')
    })
    
    # Save model
    joblib.dump(best_rf, os.path.join(output_dir, f"enhanced_rf_{feature_set_name}.joblib"))
    
    # 2. Enhanced XGBoost
    print("\n2. Training Enhanced XGBoost...")
    xgb_params = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [200, 300, 500],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb_clf = XGBClassifier(random_state=42, objective='multi:softmax')
    xgb_grid = GridSearchCV(xgb_clf, xgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, Y_train)
    
    best_xgb = xgb_grid.best_estimator_
    xgb_pred = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(Y_test, xgb_pred)
    
    print(f"Best XGB params: {xgb_grid.best_params_}")
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    
    results.append({
        'model': 'Enhanced XGBoost',
        'accuracy': xgb_acc,
        'precision': precision_score(Y_test, xgb_pred, average='weighted'),
        'recall': recall_score(Y_test, xgb_pred, average='weighted'),
        'f1': f1_score(Y_test, xgb_pred, average='weighted')
    })
    
    joblib.dump(best_xgb, os.path.join(output_dir, f"enhanced_xgb_{feature_set_name}.joblib"))
    
    # 3. Enhanced Gradient Boosting
    print("\n3. Training Enhanced Gradient Boosting...")
    gb_params = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [5, 7, 9],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9]
    }
    
    gb_clf = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb_clf, gb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, Y_train)
    
    best_gb = gb_grid.best_estimator_
    gb_pred = best_gb.predict(X_test)
    gb_acc = accuracy_score(Y_test, gb_pred)
    
    print(f"Best GB params: {gb_grid.best_params_}")
    print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
    
    results.append({
        'model': 'Enhanced Gradient Boosting',
        'accuracy': gb_acc,
        'precision': precision_score(Y_test, gb_pred, average='weighted'),
        'recall': recall_score(Y_test, gb_pred, average='weighted'),
        'f1': f1_score(Y_test, gb_pred, average='weighted')
    })
    
    joblib.dump(best_gb, os.path.join(output_dir, f"enhanced_gb_{feature_set_name}.joblib"))
    
    # 4. Ensemble Model (Voting Classifier)
    print("\n4. Training Ensemble Model...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('xgb', best_xgb),
            ('gb', best_gb)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, Y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(Y_test, ensemble_pred)
    
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    
    results.append({
        'model': 'Ensemble (Voting)',
        'accuracy': ensemble_acc,
        'precision': precision_score(Y_test, ensemble_pred, average='weighted'),
        'recall': recall_score(Y_test, ensemble_pred, average='weighted'),
        'f1': f1_score(Y_test, ensemble_pred, average='weighted')
    })
    
    joblib.dump(ensemble, os.path.join(output_dir, f"ensemble_{feature_set_name}.joblib"))
    
    # Print classification report for best model
    best_model_idx = np.argmax([r['accuracy'] for r in results])
    best_model_name = results[best_model_idx]['model']
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"{'='*60}")
    
    if best_model_name == 'Ensemble (Voting)':
        print(classification_report(Y_test, ensemble_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Train enhanced ML models for student engagement')
    parser.add_argument('--data_dir', type=str, default='../WACV data',
                        help='Directory containing the merged data files')
    parser.add_argument('--output_dir', type=str, default='../Results/ML_Enhanced',
                        help='Directory to save trained models and results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    df, scaler = load_and_prepare_data(args.data_dir)
    
    # Save scaler for later use
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    
    # Create ML classification object
    obj = ML_classification()
    
    # Generate training data for the four feature sets
    training_data = obj.feature_sets(df)
    
    feature_names = ['S1_MediaPipe', 'S2_Gaze_HeadPose', 'S3_ActionUnits', 'S4_Combined']
    
    all_results = []
    
    # Train enhanced models for each feature set
    for i, feature_name in enumerate(feature_names):
        print(f"\n\n{'#'*80}")
        print(f"# Processing Feature Set {i}: {feature_name}")
        print(f"{'#'*80}\n")
        
        results_df = train_enhanced_model(training_data[i], feature_name, args.output_dir)
        results_df['feature_set'] = feature_name
        all_results.append(results_df)
        
        # Save individual results
        results_df.to_csv(os.path.join(args.output_dir, f"results_{feature_name}.csv"), index=False)
        
        print(f"\nResults for {feature_name}:")
        print(results_df.to_string(index=False))
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(os.path.join(args.output_dir, "all_results_enhanced.csv"), index=False)
    
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll models and results saved to: {args.output_dir}")
    print("\nBest Model per Feature Set:")
    
    for feature_name in feature_names:
        subset = combined_results[combined_results['feature_set'] == feature_name]
        best_idx = subset['accuracy'].idxmax()
        best_model = subset.loc[best_idx]
        print(f"\n{feature_name}:")
        print(f"  Model: {best_model['model']}")
        print(f"  Accuracy: {best_model['accuracy']:.4f}")
        print(f"  F1-Score: {best_model['f1']:.4f}")

if __name__ == "__main__":
    main()

"""
Test script to verify installation and dependencies
Run this before training to ensure everything is set up correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("="*60)
    print("Testing Package Imports")
    print("="*60)
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'joblib': 'joblib',
    }
    
    failed = []
    for package_name, import_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name:20s} {version}")
        except ImportError:
            print(f"✗ {package_name:20s} NOT FOUND")
            failed.append(package_name)
    
    # Test optional packages
    print("\nOptional Packages (for Deep Learning):")
    optional = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'tensorflow': 'tensorflow',
    }
    
    for package_name, import_name in optional.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name:20s} {version}")
        except ImportError:
            print(f"○ {package_name:20s} not installed (optional)")
    
    if failed:
        print(f"\n❌ Missing required packages: {', '.join(failed)}")
        print("\nInstall missing packages with:")
        print(f"  pip install {' '.join(failed)}")
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def test_data_files():
    """Test if data files exist"""
    print("\n" + "="*60)
    print("Testing Data Files")
    print("="*60)
    
    import os
    
    data_dir = "WACV data"
    required_files = [
        "merged_data0.csv",
        "merged_data1.csv",
        "merged_data2.csv",
    ]
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("\nYou need to extract features first:")
        print("  1. Run Extract_OpenFace_features.ipynb")
        print("  2. Run Extract_MediaPipe_features.py")
        return False
    
    missing = []
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            print(f"✓ {file:25s} ({size:.2f} MB)")
        else:
            print(f"✗ {file:25s} NOT FOUND")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing data files: {', '.join(missing)}")
        print("\nExtract features first:")
        print("  cd Feature_extract")
        print("  python Extract_MediaPipe_features.py")
        return False
    
    print("\n✅ All data files are present!")
    return True

def test_model_loading():
    """Test if ML model classes can be instantiated"""
    print("\n" + "="*60)
    print("Testing Model Classes")
    print("="*60)
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        
        print("✓ RandomForestClassifier")
        print("✓ GradientBoostingClassifier")
        print("✓ XGBClassifier")
        
        # Try to instantiate
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        xgb = XGBClassifier(n_estimators=10, random_state=42, objective='multi:softmax')
        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        
        print("\n✅ All model classes can be instantiated!")
        return True
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        return False

def test_enhanced_model():
    """Test if enhanced model script exists and is valid Python"""
    print("\n" + "="*60)
    print("Testing Enhanced Model Script")
    print("="*60)
    
    import os
    
    script_path = "ML_models/train_model_ML_enhanced.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Enhanced model script not found: {script_path}")
        return False
    
    print(f"✓ Script exists: {script_path}")
    
    # Try to compile the script
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print("✓ Script is valid Python")
        print("\n✅ Enhanced model script is ready!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in script: {e}")
        return False

def main():
    print("\n" + "#"*60)
    print("# Student Engagement Detection - Installation Test")
    print("#"*60 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Package Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Model Classes", test_model_loading()))
    results.append(("Enhanced Script", test_enhanced_model()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run the enhanced training:")
        print("  cd ML_models")
        print("  python train_model_ML_enhanced.py \\")
        print('    --data_dir "../WACV data" \\')
        print('    --output_dir "../Results/ML_Enhanced"')
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        print("\nFor help, see:")
        print("  - PANDUAN_LENGKAP.md (Indonesian)")
        print("  - COMPLETE_GUIDE.md (English)")
        sys.exit(1)

if __name__ == "__main__":
    main()

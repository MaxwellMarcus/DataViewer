#!/usr/bin/env python3
"""
Setup verification script for the enhanced data visualization tool
"""

import sys
import os
import numpy as np
import pandas as pd

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    required_modules = {
        'dearpygui': 'dearpygui',
        'numpy': 'numpy', 
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'umap': 'umap-learn',
        'hdbscan': 'hdbscan'
    }
    
    optional_modules = {
        'sentence_transformers': 'sentence-transformers',
        'transformers': 'transformers', 
        'torch': 'torch',
        'datasets': 'datasets',
        'openai': 'openai'
    }
    
    all_good = True
    
    # Check required modules
    for module, package in required_modules.items():
        try:
            __import__(module)
            print(f"  ✅ {module} ({package})")
        except ImportError:
            print(f"  ❌ {module} ({package}) - REQUIRED")
            all_good = False
    
    # Check optional modules
    for module, package in optional_modules.items():
        try:
            __import__(module)
            print(f"  ✅ {module} ({package}) - optional")
        except ImportError:
            print(f"  ⚠️  {module} ({package}) - optional (enhances functionality)")
    
    return all_good

def check_files():
    """Check if all required files exist"""
    print("\n🔍 Checking files...")
    
    required_files = [
        'app_dearpygui.py',
        'enhanced_file_selector.py', 
        'sample_models.py',
        'preprocess.py',
        'load_dataset.py',
        'data_manager.py',
        'tab_manager.py'
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_good = False
    
    return all_good

def check_enhanced_selector():
    """Check if enhanced file selector can be instantiated"""
    print("\n🔍 Checking enhanced file selector...")
    
    try:
        from enhanced_file_selector import EnhancedFileSelector
        selector = EnhancedFileSelector()
        print("  ✅ EnhancedFileSelector can be instantiated")
        return True
    except Exception as e:
        print(f"  ❌ EnhancedFileSelector error: {e}")
        return False

def check_models():
    """Check available models"""
    print("\n🔍 Checking available models...")
    
    try:
        from sample_models import get_available_models
        models = get_available_models()
        
        if models:
            print(f"  ✅ {len(models)} model types available:")
            for model_type, info in models.items():
                print(f"    • {model_type}: {len(info['models'])} models")
        else:
            print("  ⚠️  No models available (will use fallback)")
        
        return True
    except Exception as e:
        print(f"  ❌ Model check error: {e}")
        return False

def check_generated_files():
    """Check if test files exist"""
    print("\n🔍 Checking generated test files...")
    
    test_files = ['generated_embeddings.npy', 'generated_metadata.csv']
    
    for file in test_files:
        if os.path.exists(file):
            if file.endswith('.npy'):
                try:
                    data = np.load(file)
                    print(f"  ✅ {file} - shape: {data.shape}")
                except:
                    print(f"  ⚠️  {file} - exists but corrupted")
            elif file.endswith('.csv'):
                try:
                    data = pd.read_csv(file)
                    print(f"  ✅ {file} - shape: {data.shape}")
                except:
                    print(f"  ⚠️  {file} - exists but corrupted")
        else:
            print(f"  ❓ {file} - not found (run test_enhanced_selector.py to generate)")

def main():
    """Main check function"""
    print("🚀 Enhanced Data Visualization Tool - Setup Check")
    print("=" * 50)
    
    checks = [
        check_imports(),
        check_files(), 
        check_enhanced_selector(),
        check_models()
    ]
    
    check_generated_files()
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 All checks passed! The enhanced file selector is ready to use.")
        print("\nTo run the application:")
        print("  python app_dearpygui.py")
        print("\nTo test the enhanced selector:")
        print("  python test_enhanced_selector.py")
    else:
        print("❌ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements_enhanced.txt")
        
    print("\n📚 For more information, see ENHANCED_FEATURES.md")

if __name__ == "__main__":
    main() 
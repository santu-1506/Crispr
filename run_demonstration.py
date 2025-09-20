# run_demonstration.py - Simple runner for the complete demonstration
import os
import sys

print("=" * 100)
print("CRISPR-Cas9 Off-Target Prediction - PROFESSOR DEMONSTRATION")
print("=" * 100)

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "datasets/c.txt",
        "data_process.py", 
        "utils.py",
        "detailed_model.py",
        "demonstrate_model.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    """Main demonstration runner"""
    print(f"\n🔍 [System Check] Verifying requirements...")
    
    if not check_requirements():
        print("❌ Please ensure all required files are present before running the demonstration.")
        return
    
    print(f"\n🚀 [Starting Demonstration] Running complete model demonstration...")
    print(f"    This will show every step of the CRISPR off-target prediction model")
    print(f"    Perfect for professor presentation!")
    
    try:
        # Import and run the demonstration
        from demonstrate_model import main_demonstration
        main_demonstration()
        
    except Exception as e:
        print(f"❌ [Error] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 [Troubleshooting] Make sure you have:")
        print(f"    1. TensorFlow installed: pip install tensorflow")
        print(f"    2. scikit-learn installed: pip install scikit-learn")
        print(f"    3. All dataset files in the datasets/ folder")
        print(f"    4. Sufficient memory for model training")

if __name__ == "__main__":
    main()

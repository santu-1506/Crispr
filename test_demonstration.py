# test_demonstration.py - Simple test to verify everything works
import numpy as np
import tensorflow as tf
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE

print("=" * 80)
print("CRISPR Model - Quick Test")
print("=" * 80)

def test_data_loading():
    """Test data loading"""
    print("\n📊 [Testing Data Loading]")
    try:
        X_tokens, X_onehot, y = load_dataset("datasets/c.txt")
        print(f"✅ Data loaded successfully!")
        print(f"   Samples: {len(y)}")
        print(f"   Token shape: {X_tokens.shape}")
        print(f"   One-hot shape: {X_onehot.shape}")
        print(f"   Labels: {np.unique(y, return_counts=True)}")
        return X_tokens, X_onehot, y
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None, None

def test_model_building():
    """Test model building"""
    print("\n🏗️  [Testing Model Building]")
    try:
        from detailed_model import build_crispr_bert_model
        model = build_crispr_bert_model(VOCAB_SIZE, MAX_LEN, small_debug=True)
        print(f"✅ Model built successfully!")
        print(f"   Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"   Output shape: {model.output.shape}")
        return model
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        return None

def test_single_prediction(model, X_tokens, X_onehot, y):
    """Test single prediction"""
    print("\n🎯 [Testing Single Prediction]")
    try:
        # Take first sample
        token_sample = X_tokens[0:1]
        onehot_sample = X_onehot[0:1]
        label_sample = y[0]
        
        print(f"   Sample token shape: {token_sample.shape}")
        print(f"   Sample onehot shape: {onehot_sample.shape}")
        print(f"   True label: {label_sample}")
        
        # Make prediction
        prediction = model.predict([token_sample, onehot_sample], verbose=0)
        print(f"✅ Prediction successful!")
        print(f"   Prediction probabilities: {prediction[0]}")
        print(f"   Predicted class: {np.argmax(prediction[0])}")
        print(f"   Confidence: {np.max(prediction[0]):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def main():
    """Main test function"""
    print("\n🚀 [Starting Tests]")
    
    # Test 1: Data loading
    X_tokens, X_onehot, y = test_data_loading()
    if X_tokens is None:
        return
    
    # Test 2: Model building
    model = test_model_building()
    if model is None:
        return
    
    # Test 3: Single prediction
    success = test_single_prediction(model, X_tokens, X_onehot, y)
    
    if success:
        print(f"\n🎉 [All Tests Passed!] Ready for professor demonstration!")
        print(f"\nTo run the full demonstration:")
        print(f"   python demonstrate_model.py")
        print(f"   python run_demonstration.py")
    else:
        print(f"\n❌ [Tests Failed] Please check the errors above.")

if __name__ == "__main__":
    main()

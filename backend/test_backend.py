#!/usr/bin/env python3
"""
Test script for SignBridge backend API.
Generates random landmark data and sends to the prediction endpoint.
"""

import requests
import numpy as np
import json

BACKEND_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("[TEST] Health check...")
    response = requests.get(f"{BACKEND_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("[TEST] Model info...")
    response = requests.get(f"{BACKEND_URL}/model-info")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Model: {data['model_name']} v{data['version']}")
    print(f"Classes: {data['num_classes']}")
    print(f"Supported: {', '.join(sorted(data['supported_classes']))}")
    print(f"Test Accuracy: {data['test_accuracy']:.2%}\n")
    return response.status_code == 200

def test_predict():
    """Test prediction endpoint with random landmarks"""
    print("[TEST] Single prediction...")
    
    # Generate random landmarks (63 features)
    landmarks = np.random.randn(63).tolist()
    
    payload = {"landmarks": landmarks}
    response = requests.post(f"{BACKEND_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Predicted: {data['predicted_class']} (index {data['predicted_index']})")
        print(f"Confidence: {data['confidence']:.2%}")
        print(f"Top 5 predictions:")
        sorted_preds = sorted(data['all_predictions'].items(), key=lambda x: x[1], reverse=True)[:5]
        for label, conf in sorted_preds:
            print(f"  {label}: {conf:.4f}")
    print()
    return response.status_code == 200

def test_batch_predict():
    """Test batch prediction"""
    print("[TEST] Batch prediction (3 samples)...")
    
    batch = [
        {"landmarks": np.random.randn(63).tolist()},
        {"landmarks": np.random.randn(63).tolist()},
        {"landmarks": np.random.randn(63).tolist()},
    ]
    
    response = requests.post(f"{BACKEND_URL}/batch-predict", json=batch)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Predictions: {len(data['predictions'])} samples")
        for i, pred in enumerate(data['predictions']):
            print(f"  Sample {i+1}: {pred['predicted_class']} (conf: {pred['confidence']:.2%})")
    print()
    return response.status_code == 200

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("SignBridge Backend Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_predict),
        ("Batch Prediction", test_batch_predict),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} FAILED: {e}\n")
            results[name] = False
    
    print("="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    print()
    
    all_passed = all(results.values())
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    
    return all_passed

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to backend at {BACKEND_URL}")
        print("Make sure the backend server is running: python app.py")

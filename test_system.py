#!/usr/bin/env python3
"""
Test script for Delhi House Price Prediction System

This script runs basic tests to ensure the system works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from delhi_house_prediction import DelhiHousePricePredictor

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    predictor = DelhiHousePricePredictor()
    
    try:
        df = predictor.load_data()
        assert df is not None, "Data loading failed"
        assert len(df) > 0, "Empty dataset"
        assert 'Price_lakhs' in df.columns, "Target column missing"
        print("✅ Data loading test passed")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_data_cleaning():
    """Test data cleaning functionality"""
    print("Testing data cleaning...")
    predictor = DelhiHousePricePredictor()
    
    try:
        predictor.load_data()
        df_clean = predictor.clean_data()
        assert df_clean is not None, "Data cleaning failed"
        assert len(df_clean) > 0, "All data removed during cleaning"
        print("✅ Data cleaning test passed")
        return True
    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("Testing feature engineering...")
    predictor = DelhiHousePricePredictor()
    
    try:
        predictor.load_data()
        predictor.clean_data()
        X, y = predictor.feature_engineering()
        assert X is not None and y is not None, "Feature engineering failed"
        assert len(X) == len(y), "Feature and target length mismatch"
        assert X.shape[1] > 0, "No features generated"
        print("✅ Feature engineering test passed")
        return True
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("Testing model training...")
    predictor = DelhiHousePricePredictor()
    
    try:
        predictor.load_data()
        predictor.clean_data()
        X, y = predictor.feature_engineering()
        train_pred, test_pred = predictor.train_model(X, y)
        assert predictor.model is not None, "Model not trained"
        assert len(train_pred) > 0 and len(test_pred) > 0, "No predictions generated"
        print("✅ Model training test passed")
        return True
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("Testing prediction...")
    predictor = DelhiHousePricePredictor()
    
    try:
        # Run minimal pipeline to train model
        predictor.load_data()
        predictor.clean_data()
        X, y = predictor.feature_engineering()
        predictor.train_model(X, y)
        
        # Test prediction
        test_house = {
            'Area_sqft': 1200,
            'Bedrooms': 3,
            'Bathrooms': 2,
            'Age_years': 5,
            'Parking_spaces': 1,
            'Metro_nearby': 1,
            'Floor': 5,
            'Total_floors': 15,
            'Area_per_bedroom': 400,
            'Bathroom_bedroom_ratio': 0.67,
            'Is_highrise': 1,
            'Locality_encoded': 10,
            'Furnished_status_encoded': 1,
            'Floor_position_encoded': 1
        }
        
        predicted_price = predictor.predict_price(test_house)
        assert isinstance(predicted_price, (int, float)), "Invalid prediction type"
        assert predicted_price > 0, "Negative price prediction"
        print(f"✅ Prediction test passed (predicted: ₹{predicted_price:.2f} Lakhs)")
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_dataset_integrity():
    """Test dataset integrity"""
    print("Testing dataset integrity...")
    
    try:
        df = pd.read_csv('delhi_house_data.csv')
        
        # Check required columns
        required_columns = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Age_years', 
                          'Locality', 'Parking_spaces', 'Furnished_status',
                          'Metro_nearby', 'Floor', 'Total_floors', 'Price_lakhs']
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check data types and ranges
        assert df['Area_sqft'].min() > 0, "Invalid area values"
        assert df['Bedrooms'].min() >= 1, "Invalid bedroom count"
        assert df['Bathrooms'].min() >= 1, "Invalid bathroom count"
        assert df['Price_lakhs'].min() > 0, "Invalid price values"
        
        print("✅ Dataset integrity test passed")
        return True
    except Exception as e:
        print(f"❌ Dataset integrity test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏠 DELHI HOUSE PRICE PREDICTION - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_dataset_integrity,
        test_data_loading,
        test_data_cleaning,
        test_feature_engineering,
        test_model_training,
        test_prediction
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
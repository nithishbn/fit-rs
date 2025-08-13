#!/usr/bin/env python3
"""Simple test for JSON config functionality."""

import json
import requests

def test_json_config():
    """Test the server with new JSON config format."""
    print("Testing JSON config format...")
    
    # Prepare test data (CSV)
    csv_data = """concentration,response
1e-9,0.1
1e-8,0.2
1e-7,0.5
1e-6,0.8
1e-5,0.9"""
    
    # JSON config format (matches CLI configuration)
    config = {
        "input": {
            "file": "dummy.csv",
            "output_dir": "test_output"
        },
        "mcmc": {
            "samples": 500,
            "burnin": 200,
            "chains": 2,
            "sigma": 0.05,
            "backend": "mh"
        },
        "priors": {
            "emin": {
                "type": "normal",
                "mean": 0.0,
                "std": 0.5,
                "description": "Lower asymptote (baseline response)"
            },
            "emax": {
                "type": "normal",
                "mean": 1.0,
                "std": 0.5,
                "description": "Upper asymptote (maximum response)"
            },
            "ec50": {
                "type": "normal",
                "mean": -7.0,
                "std": 1.0,
                "description": "EC50 on log10 scale"
            },
            "hillslope": {
                "type": "normal",
                "mean": 1.0,
                "std": 0.5,
                "description": "Hill slope (steepness of curve)"
            }
        },
        "plotting": {
            "bounds": {
                "x_min": None,
                "x_max": None,
                "y_min": None,
                "y_max": None
            },
            "verbose": False
        },
        "metadata": {
            "description": "Test configuration for API",
            "version": "1.0"
        }
    }
    
    files = {
        'file': ('test_data.csv', csv_data, 'text/csv'),
        'config': (None, json.dumps(config), 'application/json')
    }
    
    response = requests.post('http://localhost:3000/fit_curve', files=files)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ JSON config format test passed")
            print(f"   EC50 estimate: {result['parameters']['ec50_log10']['mean']:.3f}")
            print(f"   Model quality: {result['diagnostics']['model_quality']}")
            print(f"   Convergence: {result['diagnostics']['rhat_estimates']['convergence_status']}")
            return True
        else:
            print(f"❌ JSON config failed: {result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"❌ JSON config HTTP error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_legacy_format():
    """Test the server with legacy parameter format."""
    print("Testing legacy parameter format...")
    
    # Prepare test data (CSV)
    csv_data = """concentration,response
1e-9,0.1
1e-8,0.2
1e-7,0.5
1e-6,0.8
1e-5,0.9"""
    
    # Legacy parameter format
    params = {
        "emin_mean": 0.0,
        "emin_std": 0.5,
        "emax_mean": 1.0,
        "emax_std": 0.5,
        "ec50_mean": -7.0,
        "ec50_std": 1.0,
        "hillslope_mean": 1.0,
        "hillslope_std": 0.5,
        "samples": 500,
        "burnin": 200,
        "chains": 2,
        "sigma": 0.05,
        "backend": "mh"
    }
    
    files = {
        'file': ('test_data.csv', csv_data, 'text/csv'),
        'parameters': (None, json.dumps(params), 'application/json')
    }
    
    response = requests.post('http://localhost:3000/fit_curve', files=files)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ Legacy format test passed")
            print(f"   EC50 estimate: {result['parameters']['ec50_log10']['mean']:.3f}")
            return True
        else:
            print(f"❌ Legacy format failed: {result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"❌ Legacy format HTTP error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

if __name__ == "__main__":
    print("Testing both formats...")
    print("=" * 40)
    
    legacy_passed = test_legacy_format()
    print()
    json_passed = test_json_config()
    
    print()
    print(f"Results: Legacy={legacy_passed}, JSON Config={json_passed}")
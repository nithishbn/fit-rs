#!/usr/bin/env python3
"""Test script for the updated EC50 fitting server with JSON config support."""

import json
import requests
import time
import subprocess
import signal
import os
from pathlib import Path

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
        "samples": 1000,
        "burnin": 500,
        "chains": 2,
        "sigma": 0.05,
        "backend": "mh"
    }
    
    files = {
        'file': ('test_data.csv', csv_data, 'text/csv'),
        'parameters': (None, json.dumps(params), 'application/json')
    }
    
    response = requests.post('http://localhost:3000/fit_curve', files=files)
    
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
        return False

def test_json_config_format():
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
            "file": "dummy.csv",  # This will be ignored since we upload the file
            "output_dir": "test_output"
        },
        "mcmc": {
            "samples": 1000,
            "burnin": 500,
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
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ JSON config format test passed")
            print(f"   EC50 estimate: {result['parameters']['ec50_log10']['mean']:.3f}")
            print(f"   Model quality: {result['diagnostics']['model_quality']}")
            return True
        else:
            print(f"❌ JSON config failed: {result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"❌ JSON config HTTP error: {response.status_code}")
        return False

def test_stan_backend():
    """Test the server with Stan backend using JSON config."""
    print("Testing Stan backend with JSON config...")
    
    # Prepare test data (CSV)
    csv_data = """concentration,response
1e-9,0.1
1e-8,0.2
1e-7,0.5
1e-6,0.8
1e-5,0.9"""
    
    # JSON config with Stan backend
    config = {
        "input": {
            "file": "dummy.csv",
            "output_dir": "test_output"
        },
        "mcmc": {
            "samples": 500,  # Smaller for faster testing
            "burnin": 200,
            "chains": 2,
            "sigma": 0.05,
            "backend": "stan"
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
                "type": "uniform",
                "mean": -7.0,  # Ignored for uniform
                "std": 1.0,    # Ignored for uniform
                "min": -10.0,
                "max": -4.0,
                "description": "EC50 on log10 scale"
            },
            "hillslope": {
                "type": "lognormal",
                "mean": 0.0,   # Log-mean
                "std": 0.5,    # Log-std
                "description": "Hill slope (steepness of curve)"
            }
        },
        "plotting": {
            "bounds": {
                "x_min": -10.0,
                "x_max": -4.0,
                "y_min": 0.0,
                "y_max": 1.0
            },
            "verbose": True
        },
        "metadata": {
            "description": "Stan backend test configuration",
            "version": "1.0"
        }
    }
    
    files = {
        'file': ('test_data.csv', csv_data, 'text/csv'),
        'config': (None, json.dumps(config), 'application/json')
    }
    
    response = requests.post('http://localhost:3000/fit_curve', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print("✅ Stan backend test passed")
            print(f"   EC50 estimate: {result['parameters']['ec50_log10']['mean']:.3f}")
            print(f"   Convergence: {result['diagnostics']['rhat_estimates']['convergence_status']}")
            return True
        else:
            print(f"❌ Stan backend failed: {result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"❌ Stan backend HTTP error: {response.status_code}")
        return False

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    
    response = requests.get('http://localhost:3000/health')
    
    if response.status_code == 200:
        result = response.json()
        if result.get('status') == 'healthy':
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {result}")
            return False
    else:
        print(f"❌ Health check HTTP error: {response.status_code}")
        return False

def main():
    print("🚀 Starting EC50 Fitting Server Test Suite")
    print("==========================================")
    
    # Start the server
    print("Starting server...")
    server_process = subprocess.Popen(
        ["cargo", "run", "--bin", "fit-server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint first
        if not test_health_endpoint():
            print("❌ Server not responding properly")
            return False
        
        print()
        
        # Run all tests
        tests = [
            test_legacy_format,
            test_json_config_format,
            test_stan_backend
        ]
        
        passed = 0
        for test in tests:
            if test():
                passed += 1
            print()
        
        print(f"Test Results: {passed}/{len(tests)} passed")
        
        if passed == len(tests):
            print("🎉 All tests passed! Server is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the output above.")
            return False
            
    finally:
        # Clean up: stop the server
        print("Stopping server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
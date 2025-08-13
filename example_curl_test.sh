#!/bin/bash

# Example curl script to test the EC50 fitting API with JSON configuration
# This demonstrates how to use the new JSON config format with the web server

set -e

echo "🧪 EC50 Fitting API Test with JSON Configuration"
echo "================================================"

# Create test data CSV file
cat > test_data.csv << 'EOF'
concentration,response
1e-10,0.05
1e-9,0.08
1e-8,0.15
1e-7,0.45
1e-6,0.85
1e-5,0.95
1e-4,0.98
EOF

echo "📄 Created test_data.csv with sample dose-response data"

# Create JSON configuration file (same format as CLI)
cat > config.json << 'EOF'
{
  "input": {
    "file": "data/test_data.csv",
    "output_dir": "api_output"
  },
  "mcmc": {
    "samples": 1000,
    "burnin": 500,
    "chains": 3,
    "sigma": 0.05,
    "backend": "mh"
  },
  "priors": {
    "emin": {
      "type": "normal",
      "mean": 0.0,
      "std": 0.2,
      "description": "Lower asymptote (baseline response)"
    },
    "emax": {
      "type": "normal",
      "mean": 1.0,
      "std": 0.2,
      "description": "Upper asymptote (maximum response)"
    },
    "ec50": {
      "type": "normal",
      "mean": -7.0,
      "std": 1.5,
      "description": "EC50 on log10 scale"
    },
    "hillslope": {
      "type": "lognormal",
      "mean": 0.0,
      "std": 0.5,
      "description": "Hill slope (steepness of curve)"
    }
  },
  "plotting": {
    "bounds": {
      "x_min": -11.0,
      "x_max": -3.0,
      "y_min": 0.0,
      "y_max": 1.0
    },
    "verbose": true
  },
  "metadata": {
    "description": "Example API configuration with lognormal hill slope",
    "version": "1.0",
    "created": "2025-01-01T00:00:00Z"
  }
}
EOF

echo "⚙️  Created config.json with JSON configuration"

# Test with JSON configuration
echo "🚀 Sending request to API with JSON config..."
echo

response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
  -X POST \
  -F "file=@test_data.csv" \
  -F "config=@config.json" \
  http://localhost:3000/fit_curve)

# Extract HTTP status and response body
http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
response_body=$(echo "$response" | sed '/HTTP_STATUS:/d')

echo "📊 Response Status: $http_status"
echo

if [ "$http_status" = "200" ]; then
    echo "✅ Request successful!"
    echo
    
    # Pretty print the JSON response
    echo "📈 Results:"
    echo "$response_body" | python3 -m json.tool | head -50
    
    # Extract key results
    echo
    echo "🎯 Key Results:"
    ec50_mean=$(echo "$response_body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['parameters']['ec50_log10']['mean']:.3f}\")" 2>/dev/null || echo "N/A")
    ec50_linear=$(echo "$response_body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['parameters']['ec50_linear']['mean']:.2e}\")" 2>/dev/null || echo "N/A")
    convergence=$(echo "$response_body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['diagnostics']['rhat_estimates']['convergence_status'])" 2>/dev/null || echo "N/A")
    model_quality=$(echo "$response_body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['diagnostics']['model_quality'])" 2>/dev/null || echo "N/A")
    
    echo "   EC50 (log10): $ec50_mean"
    echo "   EC50 (linear): $ec50_linear"
    echo "   Convergence: $convergence"
    echo "   Model Quality: $model_quality"
else
    echo "❌ Request failed with status $http_status"
    echo "Response: $response_body"
fi

echo
echo "🧹 Cleaning up..."
rm -f test_data.csv config.json

echo "✨ Test complete!"

# Additional examples for different configurations
echo
echo "📝 Additional curl examples:"
echo
echo "# Example 1: Using uniform priors"
echo 'curl -X POST \'
echo '  -F "file=@your_data.csv" \'
echo '  -F "config={
    \"input\": {\"file\": \"data.csv\", \"output_dir\": \"output\"},
    \"mcmc\": {\"samples\": 2000, \"burnin\": 1000, \"chains\": 4, \"sigma\": 0.05, \"backend\": \"mh\"},
    \"priors\": {
      \"emin\": {\"type\": \"uniform\", \"mean\": 0, \"std\": 0, \"min\": -0.5, \"max\": 0.5},
      \"emax\": {\"type\": \"uniform\", \"mean\": 0, \"std\": 0, \"min\": 0.5, \"max\": 1.5},
      \"ec50\": {\"type\": \"normal\", \"mean\": -6.0, \"std\": 2.0},
      \"hillslope\": {\"type\": \"normal\", \"mean\": 1.0, \"std\": 1.0}
    },
    \"plotting\": {\"bounds\": {}, \"verbose\": false}
  }" \'
echo '  http://localhost:3000/fit_curve'

echo
echo "# Example 2: Using Stan backend"
echo 'curl -X POST \'
echo '  -F "file=@your_data.csv" \'
echo '  -F "config={
    \"input\": {\"file\": \"data.csv\", \"output_dir\": \"output\"},
    \"mcmc\": {\"samples\": 1000, \"burnin\": 500, \"chains\": 4, \"sigma\": 0.05, \"backend\": \"stan\"},
    \"priors\": {
      \"emin\": {\"type\": \"normal\", \"mean\": 0.0, \"std\": 0.5},
      \"emax\": {\"type\": \"normal\", \"mean\": 100.0, \"std\": 20.0},
      \"ec50\": {\"type\": \"normal\", \"mean\": -6.0, \"std\": 2.0},
      \"hillslope\": {\"type\": \"lognormal\", \"mean\": 0.0, \"std\": 0.5}
    },
    \"plotting\": {\"bounds\": {}, \"verbose\": true}
  }" \'
echo '  http://localhost:3000/fit_curve'

echo
echo "# Example 3: Legacy format (backward compatibility)"
echo 'curl -X POST \'
echo '  -F "file=@your_data.csv" \'
echo '  -F "parameters={
    \"emin_mean\": 0.0, \"emin_std\": 0.5,
    \"emax_mean\": 1.0, \"emax_std\": 0.5,
    \"ec50_mean\": -7.0, \"ec50_std\": 1.0,
    \"hillslope_mean\": 1.0, \"hillslope_std\": 0.5,
    \"samples\": 1000, \"burnin\": 500, \"chains\": 2,
    \"sigma\": 0.05, \"backend\": \"mh\"
  }" \'
echo '  http://localhost:3000/fit_curve'
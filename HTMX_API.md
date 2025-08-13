# EC50 Curve Fitting - HTMX Server API Documentation

This document describes the HTTP API for the `htmx-server` binary, which provides a web interface for Bayesian EC50 curve fitting.

## 🚀 Server Setup

### Starting the Server

```bash
# Build and run the server
cargo build --bin htmx-server
cargo run --bin htmx-server

# Or run the built binary directly
./target/debug/htmx-server
```

**Server URL**: `http://localhost:3001`

## 📡 API Endpoints

### 1. Main Interface

#### `GET /`
Returns the main HTMX web interface.

**Response**: HTML page with file upload forms and plot containers.

---

### 2. File Upload Endpoints

#### `POST /upload-files` (Recommended)
Upload both CSV data and JSON configuration files in a single request.

**Content-Type**: `multipart/form-data`

**Form Fields**:
- `files`: Multiple files (CSV + JSON)
  - **CSV File**: Data file with `concentration` and `response` columns
  - **JSON File**: Configuration file with priors and MCMC settings

**Response**: HTML with plot updates and status messages

**HTMX Targets**:
- `#upload-status`: Upload status messages
- `#main-plot-container`: Data plot (out-of-band swap)
- `#parameter-form-container`: Parameter form (if both files uploaded)

**Example**:
```bash
curl -X POST http://localhost:3001/upload-files \
  -F "files=@data.csv" \
  -F "files=@parameters.json"
```

#### `POST /upload-data` (Legacy)
Upload only CSV data file.

**Content-Type**: `multipart/form-data`

**Form Fields**:
- `data_file`: CSV file with concentration and response data

#### `POST /upload-config` (Legacy)  
Upload only JSON configuration file.

**Content-Type**: `multipart/form-data`

**Form Fields**:
- `config_file`: JSON configuration file

---

### 3. Parameter Management

#### `GET /parameter-form`
Get the parameter editing form (loaded automatically after config upload).

**Response**: HTML form for editing priors and MCMC settings

#### `POST /update-parameters`
Update parameters and run curve fitting.

**Content-Type**: `application/x-www-form-urlencoded`

**Form Fields**:
- **Prior Parameters**:
  - `emin_mean`, `emin_std`: Emin prior (mean, std dev)
  - `emax_mean`, `emax_std`: Emax prior (mean, std dev)  
  - `ec50_mean`, `ec50_std`: EC50 prior (mean, std dev)
  - `hillslope_mean`, `hillslope_std`: Hill slope prior (mean, std dev)

- **MCMC Settings**:
  - `samples`: Number of MCMC samples (100-10000)
  - `burnin`: Burn-in samples (50-5000)
  - `chains`: Number of chains (1-8)
  - `sigma`: Noise parameter (0.0001-1.0)
  - `backend`: MCMC backend (`"mh"` or `"stan"`)

**Response**: HTML with fitted curve plot and results

**HTMX Target**: `#main-plot-container`

**Example**:
```bash
curl -X POST http://localhost:3001/update-parameters \
  -d "emin_mean=0.98&emin_std=0.1&emax_mean=1.31&emax_std=0.1" \
  -d "ec50_mean=4.0&ec50_std=1.0&hillslope_mean=1.0&hillslope_std=1.0" \
  -d "samples=4000&burnin=1000&chains=4&sigma=0.05&backend=mh"
```

---

### 4. Results and Downloads

#### `GET /download-csv`
Download fitting results as CSV.

**Response**: CSV file with parameter estimates and confidence intervals

**Headers**:
- `Content-Type: text/csv`
- `Content-Disposition: attachment; filename="ec50_results_YYYYMMDD_HHMMSS.csv"`

**CSV Format**:
```csv
Parameter,Mean,Std_Dev,CI_Lower,CI_Upper
Emin,0.98,0.05,0.88,1.08
Emax,1.31,0.07,1.17,1.45
EC50_log10,4.12,0.15,3.82,4.42
EC50_linear,13193.7,4521.2,6606.9,26302.7
Hillslope,1.05,0.22,0.61,1.49
Acceptance_Rate_Percent,65.2,,,
```

---

## 📄 Data Formats

### CSV Data File Format

Required columns:
- `concentration`: Numeric values (positive, any unit)
- `response`: Numeric response values

**Example**:
```csv
concentration,response
0.1,0.98
1.0,0.95
10.0,0.85
100.0,0.45
1000.0,0.15
10000.0,0.05
```

### JSON Configuration Format

```json
{
  "input": {
    "file": "data.csv",
    "output_dir": "ec50_output"
  },
  "mcmc": {
    "samples": 4000,
    "burnin": 1000,
    "chains": 4,
    "sigma": 0.05,
    "backend": "mh"
  },
  "priors": {
    "emin": {"type": "normal", "mean": 0.98, "std": 0.1},
    "emax": {"type": "normal", "mean": 1.31, "std": 0.1},
    "ec50": {"type": "normal", "mean": 4.0, "std": 1.0},
    "hillslope": {"type": "normal", "mean": 1.0, "std": 1.0}
  }
}
```

---

## 🔄 Typical Workflow

### 1. Upload Files
```bash
# Upload both data and config
curl -X POST http://localhost:3001/upload-files \
  -F "files=@mydata.csv" \
  -F "files=@config.json"
```

### 2. Run Fitting (Optional Parameter Adjustment)
```bash
# Use default parameters or adjust via form
curl -X POST http://localhost:3001/update-parameters \
  -d "emin_mean=1.0&emin_std=0.1" \
  # ... other parameters ...
  -d "samples=2000&backend=mh"
```

### 3. Download Results
```bash
# Download CSV results
curl -X GET http://localhost:3001/download-csv \
  -o "ec50_results.csv"
```

---

## 🎯 HTMX Integration

The server is designed for HTMX frontend integration:

- **Out-of-band swaps**: Updates multiple DOM elements simultaneously
- **Progress indicators**: Loading spinners during MCMC computation
- **Form validation**: Client-side and server-side validation
- **Real-time updates**: Dynamic plot and results updates

### HTMX Attributes Used:
- `hx-post`: Form submissions
- `hx-get`: Dynamic content loading
- `hx-target`: DOM update targets
- `hx-swap-oob`: Out-of-band element updates
- `hx-indicator`: Loading state management

---

## 🐛 Error Handling

### Common HTTP Status Codes:
- **200**: Success
- **400**: Bad request (invalid file format, missing fields)
- **404**: Resource not found
- **500**: Internal server error (MCMC fitting failure)

### Error Response Format:
```html
<div class="alert alert-danger">❌ Error message here</div>
```

### Common Errors:
1. **Invalid CSV**: Missing columns, non-numeric data
2. **Invalid JSON**: Malformed configuration file
3. **Parameter validation**: Negative standard deviations, invalid ranges
4. **MCMC failure**: Convergence issues, numerical instability

---

## 🔧 Configuration Options

### MCMC Backends:
- **`"mh"`**: Metropolis-Hastings (default, faster)
- **`"stan"`**: Stan MCMC (more robust, slower)

### Performance Tuning:
- **Samples**: 1000-4000 for quick results, 4000+ for publication
- **Burn-in**: 25-50% of samples typically
- **Chains**: 1 for web interface (performance), 4+ for rigorous analysis

---

## 📊 Output Features

### Plot Elements:
- **Data points**: Blue circles showing original data
- **Best fit curve**: Red line (mean parameters)
- **95% confidence bands**: Semi-transparent red region
- **EC50 line**: Green vertical line at EC50 value

### Results Display:
- Parameter estimates with uncertainty
- Confidence intervals (95%)
- MCMC diagnostics (acceptance rate)
- Download options (JSON coefficients, CSV results)

---

## 🔗 Integration Examples

### Python Integration:
```python
import requests

# Upload files
files = {
    'files': [
        ('files', open('data.csv', 'rb')),
        ('files', open('config.json', 'rb'))
    ]
}
response = requests.post('http://localhost:3001/upload-files', files=files)

# Run fitting
params = {
    'emin_mean': 1.0, 'emin_std': 0.1,
    'emax_mean': 1.3, 'emax_std': 0.1,
    'ec50_mean': 4.0, 'ec50_std': 1.0,
    'hillslope_mean': 1.0, 'hillslope_std': 1.0,
    'samples': 2000, 'burnin': 500, 'chains': 4,
    'sigma': 0.05, 'backend': 'mh'
}
response = requests.post('http://localhost:3001/update-parameters', data=params)

# Download results
results = requests.get('http://localhost:3001/download-csv')
with open('results.csv', 'wb') as f:
    f.write(results.content)
```

### JavaScript/Fetch Integration:
```javascript
// Upload files
const formData = new FormData();
formData.append('files', csvFile);
formData.append('files', jsonFile);

const uploadResponse = await fetch('/upload-files', {
    method: 'POST',
    body: formData
});

// Run fitting
const params = new URLSearchParams({
    emin_mean: 1.0,
    emin_std: 0.1,
    // ... other parameters
    samples: 2000,
    backend: 'mh'
});

const fitResponse = await fetch('/update-parameters', {
    method: 'POST',
    body: params
});
```
# EC50 Curve Fitting - HTMX Web Interface

This branch implements a complete HTMX-based web interface for EC50 curve fitting that mimics the functionality of the existing TUI.

## 🚀 Quick Start

### Build and Run the Server
```bash
# Build the HTMX server binary
cargo build --bin htmx-server

# Run the server
cargo run --bin htmx-server

# Or run the built binary directly
./target/debug/htmx-server
```

The server will start on **http://localhost:3001**

### Usage Steps

1. **Upload Data** (Step 1)
   - Upload a CSV file with `concentration` and `response` columns
   - The data will be automatically plotted for preview

2. **Upload Configuration** (Step 2)
   - Upload a `parameters.json` file (e.g., the existing `parameters.json`)
   - This loads MCMC settings and prior distributions

3. **Edit Parameters & Fit** (Step 3)
   - Adjust prior means/standard deviations in the interactive form
   - Modify MCMC settings (samples, burnin, chains, sigma, backend)
   - Click "Run Curve Fitting" to execute MCMC and see results

## 🎯 Features

### Complete TUI Feature Parity
- ✅ **File Loading**: CSV data and JSON configuration upload
- ✅ **Parameter Editing**: Interactive forms for all priors and MCMC settings
- ✅ **Real-time Fitting**: Dynamic MCMC execution with progress feedback
- ✅ **Plot Visualization**: Automatic plot generation with confidence intervals
- ✅ **Results Display**: Parameter estimates, diagnostics, and CI
- ✅ **Multiple Backends**: Support for both Metropolis-Hastings and Stan

### Web-Specific Enhancements
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **HTMX Reactivity**: Real-time updates without page reloads
- 🎨 **Bootstrap UI**: Professional, accessible interface
- 📊 **Inline Plots**: PNG plots embedded directly in the interface
- 💾 **Download Results**: Export fitting results as JSON

## 🏗️ Technical Architecture

### Backend (Rust + Axum)
- **Server**: Axum web framework with multipart file upload
- **State Management**: Arc<RwLock<>> for thread-safe shared state
- **MCMC Integration**: Direct integration with existing fit-rs backends
- **Plot Generation**: Server-side plotting using Plotters library
- **Templates**: Askama for server-side HTML rendering

### Frontend (HTMX + Bootstrap)
- **Reactivity**: HTMX for dynamic DOM updates
- **UI Framework**: Bootstrap 5 for responsive design
- **File Uploads**: Drag-and-drop file upload areas
- **Progress Indicators**: Loading spinners and status feedback
- **Form Validation**: Client and server-side validation

### File Structure
```
├── src/htmx_server.rs          # Main HTMX server implementation
├── templates/
│   ├── index.html              # Main interface template  
│   ├── parameter_form.html     # Parameter editing form
│   └── plot_update.html        # Results display template
└── static/                     # Static assets (CSS, JS)
```

## 🔧 Configuration

The server uses the same JSON configuration format as the CLI tool:

```json
{
  "input": {
    "file": "data/data.csv",
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

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing processes on port 3001
   lsof -ti:3001 | xargs kill -9
   ```

2. **Plot Generation Errors**
   - Ensure the data CSV has valid `concentration` and `response` columns
   - Check that concentration values are positive (log10 transformation is applied)

3. **MCMC Fitting Failures**
   - Verify prior parameters are reasonable
   - Try reducing sample count for faster iteration
   - Switch between MH and Stan backends if one fails

### Development

```bash
# Watch for changes and rebuild
cargo watch -x "build --bin htmx-server"

# Run with debug logging
RUST_LOG=debug cargo run --bin htmx-server
```

## 🔄 Comparison with TUI

| Feature | TUI | HTMX Web UI |
|---------|-----|-------------|
| File Loading | CLI args + interactive prompts | Drag-and-drop upload |
| Parameter Editing | Keyboard navigation | Web forms |
| Plot Display | Terminal graphics | PNG images |
| Real-time Updates | Key press triggers | HTMX automatic |
| Accessibility | Terminal only | Web accessible |
| Multi-user | Single user | Multi-user capable |
| Mobile Support | No | Yes |

The HTMX interface provides the same powerful functionality as the TUI while being more accessible and user-friendly for collaborative work and non-technical users.
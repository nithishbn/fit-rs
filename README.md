# fit-rs

**Bayesian EC50 Curve Fitting Tool**

A high-performance Rust implementation for fitting dose-response curves using Bayesian MCMC methods. Supports both Metropolis-Hastings and Stan backends with an interactive terminal interface for real-time parameter exploration.

## Features

### Core Functionality
- **Bayesian EC50 curve fitting** using 4-parameter logistic (LL4) model
- **Multiple MCMC backends**: Custom Metropolis-Hastings and Stan/BridgeStan
- **Multi-chain MCMC** with convergence diagnostics (R-hat statistics)
- **JSON configuration system** for reproducible analysis
- **Web API server** for remote curve fitting

### Interactive Terminal Interface
- **Real-time plotting** with ratatui-based TUI
- **Interactive parameter editing** with live curve updates
- **Multiple plot modes**: dose-response curves, MCMC traces, posterior distributions, diagnostics
- **Zoom, pan, and navigation controls**
- **Live confidence interval visualization**

### Output & Visualization
- **High-quality plots**: dose-response curves with 95% confidence intervals
- **MCMC diagnostics**: trace plots, posterior distributions, log-likelihood traces
- **Multiple output formats**: PNG plots, CSV/JSON results
- **Convergence assessment** with detailed R-hat reporting

## Installation

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Stan (for Stan backend, automatically downloaded)

### Build from Source
```bash
git clone https://github.com/nithishbn/fit-rs.git
cd fit-rs
cargo build --release
```

### Binary Installation
```bash
cargo install --path .
```

## Quick Start

### 1. Generate Default Configuration
```bash
fit-rs --generate-config
```

This creates `parameters.json` with sensible defaults.

### 2. Prepare Your Data
Create a CSV file with `concentration` and `response` columns:
```csv
concentration,response
1000,0.95
5000,1.05
10000,1.15
25000,1.25
50000,1.30
```

### 3. Run Curve Fitting
```bash
# Basic fitting with plots
fit-rs --config parameters.json

# Interactive terminal interface
fit-rs --config parameters.json --interactive

# Use Stan backend for better sampling
fit-rs --config parameters.json --backend stan
```

## Configuration

### JSON Configuration File
The tool uses JSON configuration for all parameters:

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
    "emin": {
      "type": "normal",
      "mean": 0.98,
      "std": 0.1,
      "description": "Lower asymptote"
    },
    "emax": {
      "type": "normal", 
      "mean": 1.31,
      "std": 0.1,
      "description": "Upper asymptote"
    },
    "ec50": {
      "type": "normal",
      "mean": 4.0,
      "std": 1.0,
      "description": "EC50 on log10 scale"
    },
    "hillslope": {
      "type": "normal",
      "mean": 1.0,
      "std": 1.0,
      "description": "Hill slope"
    }
  }
}
```

### Command Line Overrides
```bash
# Override specific parameters
fit-rs --samples 8000 --chains 6 --backend stan

# Override input/output
fit-rs --file my_data.csv --output results/

# Validate configuration
fit-rs --validate

# Show current configuration
fit-rs --show-config
```

## Interactive Terminal Interface

Launch the interactive interface with:
```bash
fit-rs --interactive
```

### Controls
- **Plot Modes**: `[d]` Dose-Response, `[t]` MCMC Traces, `[p]` Posteriors, `[i]` Diagnostics, `[e]` Parameter Editor
- **Navigation**: `[k/j]` or `[↑/↓]` select parameters or pan, `[←/→]` pan horizontally
- **Zoom**: `[+/-]` zoom in/out, `[r]` reset view
- **Parameter Editing**: `[e]` mode → `[Enter]` edit → `[f]` refit curve
- **Help**: `[h]` toggle help, `[q]` quit

### Parameter Editor Mode
- Edit prior distributions in real-time
- See immediate curve updates
- Refit with new parameters using `[f]`
- Compare different parameter scenarios

## Web API Server

Start the web server for remote access:
```bash
fit-server
```

### API Endpoints
- **POST /fit**: Upload CSV data and get fitted curves
- **GET /health**: Health check
- **POST /fit-with-config**: Fit with custom JSON configuration

### Example Usage
```bash
curl -X POST -F "file=@data.csv" http://localhost:3000/fit
```

## MCMC Backends

### Metropolis-Hastings (Default)
- Fast, lightweight implementation
- Good for quick analysis and testing
- Built-in adaptive proposal tuning

### Stan Backend
- High-quality NUTS sampling
- Better convergence for complex models
- Automatic compilation and caching
- Superior for final analysis

```bash
# Use Stan backend
fit-rs --backend stan
```

## Model Details

### 4-Parameter Logistic Model
```
Response = Emin + (Emax - Emin) / (1 + (concentration/EC50)^(-hillslope))
```

Where:
- **Emin**: Lower asymptote (baseline response)
- **Emax**: Upper asymptote (maximum response)  
- **EC50**: Half-maximal effective concentration
- **hillslope**: Steepness of the dose-response curve

### Prior Distributions
All parameters use Normal priors by default, configurable via JSON:
- Flexible prior specification (Normal, Uniform, etc.)
- Informative or non-informative priors
- Parameter bounds support

## Output Files

The tool generates comprehensive output in the specified directory:

- **dose_response_curve.png**: Main fitted curve with confidence bands
- **mcmc_traces.png**: Parameter convergence diagnostics
- **posterior_distributions.png**: Parameter uncertainty
- **log_likelihood_trace.png**: Sampling efficiency
- **coefficients.csv**: Fitted parameters summary
- **coefficients.json**: Detailed results with diagnostics

## Performance & Scalability

- **Multi-threaded**: Parallel MCMC chains
- **Memory efficient**: Streaming data processing
- **Fast compilation**: Stan model caching
- **Large datasets**: Efficient for 1000+ data points

## Convergence Diagnostics

- **R-hat statistics**: Automated convergence assessment
- **Effective sample size**: Sampling efficiency metrics
- **Visual diagnostics**: Trace plots and posterior distributions
- **Multi-chain comparison**: Between/within chain variance analysis

## Examples

### Basic Analysis
```bash
# Generate config and run analysis
fit-rs --generate-config
fit-rs
```

### High-Quality Analysis
```bash
# Use Stan with more samples
fit-rs --backend stan --samples 8000 --chains 6
```

### Interactive Exploration
```bash
# Launch interactive interface
fit-rs --interactive
```

### Server Mode
```bash
# Start web server
fit-server &

# Submit job via API
curl -X POST -F "file=@data.csv" http://localhost:3000/fit
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use fit-rs in your research, please cite:

```bibtex
@software{fit-rs,
  title = {fit-rs: Bayesian EC50 Curve Fitting Tool},
  author = {Nithish Narasimman},
  url = {https://github.com/nithishbn/fit-rs},
  year = {2024}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/nithishbn/fit-rs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nithishbn/fit-rs/discussions)

---

**fit-rs** - Fast, accurate, and interactive Bayesian curve fitting for dose-response analysis.
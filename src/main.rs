use anyhow::Result;
use clap::Parser;
use fit_rs::{io::load_csv, BayesianEC50Fitter, MCMCSampler, StanSampler, Config, TuiPlotter};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ec50-fit")]
#[command(about = "Bayesian EC50 curve fitting tool with JSON configuration")]
#[command(version = "2.0")]
struct Cli {
    /// Path to JSON configuration file
    #[arg(short, long, default_value = "parameters.json")]
    config: PathBuf,

    /// Generate a default configuration file and exit
    #[arg(long)]
    generate_config: bool,

    /// Override: Input CSV file with 'concentration' and 'response' columns
    #[arg(long)]
    file: Option<PathBuf>,

    /// Override: Output directory for results and plots
    #[arg(long)]
    output: Option<PathBuf>,

    /// Override: Number of MCMC samples per chain
    #[arg(long)]
    samples: Option<usize>,

    /// Override: Number of burnin samples per chain
    #[arg(long)]
    burnin: Option<usize>,

    /// Override: Number of parallel chains for convergence diagnostics
    #[arg(long)]
    chains: Option<usize>,

    /// Override: MCMC Backend ('mh' for Metropolis-Hastings, 'stan' for Stan/BridgeStan)
    #[arg(long)]
    backend: Option<String>,

    /// Override: Verbose output
    #[arg(short, long)]
    verbose: Option<bool>,

    /// Validate configuration file and exit
    #[arg(long)]
    validate: bool,

    /// Print configuration summary and exit
    #[arg(long)]
    show_config: bool,

    /// Launch interactive terminal plotting interface
    #[arg(long)]
    interactive: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Handle special commands
    if args.generate_config {
        return generate_default_config(&args.config);
    }

    if args.validate {
        return validate_config(&args.config);
    }

    if args.show_config {
        return show_config(&args.config);
    }

    // Load and process configuration
    let mut config = load_config(&args.config)?;
    
    // Apply CLI overrides
    apply_cli_overrides(&mut config, &args);
    
    // Validate final configuration
    config.validate()?;
    
    // Convert configuration to internal types
    let prior = config.to_prior()?;
    let verbose = config.plotting.verbose;
    
    if verbose {
        println!("EC50 Bayesian Curve Fitting Tool (JSON Configuration)");
        println!("=====================================================");
        println!("Configuration file: {}", args.config.display());
        println!("Input file: {}", config.input.file.display());
        println!("Output directory: {}", config.input.output_dir.display());
        println!("MCMC backend: {}", config.mcmc.backend);
        println!("MCMC chains: {}", config.mcmc.chains);
        println!("MCMC samples per chain: {} (burnin: {})", config.mcmc.samples, config.mcmc.burnin);
        println!("Error sigma: {}", config.mcmc.sigma);
        println!();
        print_prior_summary(&config);
        println!();
    }

    // Load data
    if verbose {
        println!("Loading data from {}...", config.input.file.display());
    }
    let data = load_csv(&config.input.file)?;
    println!("Loaded {} data points", data.len());

    // Set up plotting bounds
    let bounds = config.get_plotting_bounds();
    if verbose {
        match bounds {
            Some((x_min, x_max, y_min, y_max)) => {
                println!("Using custom bounds: x=[{:.3}, {:.3}], y=[{:.3}, {:.3}]", 
                         x_min, x_max, y_min, y_max);
            }
            None => {
                println!("Using automatic bounds");
            }
        }
    }

    // Run MCMC chains
    println!("Running {} chains with {} samples each...", config.mcmc.chains, config.mcmc.samples);
    let chain_results = run_multiple_chains_improved(
        data,
        config.mcmc.chains,
        config.mcmc.samples,
        config.mcmc.burnin,
        prior.clone(),
        config.mcmc.sigma,
        &config.mcmc.backend,
        verbose,
    )?;

    // Use the best chain for plotting (highest log-likelihood)
    let best_chain_idx = find_best_chain(&chain_results);
    let (best_result, best_summary) = &chain_results[best_chain_idx];

    println!("\nUsing chain {} (highest log-likelihood) for plotting", best_chain_idx + 1);

    // Load data again for plotting
    let plot_data = load_csv(&config.input.file)?;
    let plot_fitter = BayesianEC50Fitter::new(plot_data)
        .with_prior(prior.clone())
        .with_sigma(config.mcmc.sigma);

    // Generate plots or launch interactive interface
    if args.interactive {
        println!("Launching interactive terminal plotting interface...");
        println!("Press 'h' for help, 'q' to quit");
        
        let plot_data_copy = load_csv(&config.input.file)?;
        let plot_fitter_copy = BayesianEC50Fitter::new(plot_data_copy.clone())
            .with_prior(prior.clone())
            .with_sigma(config.mcmc.sigma);
        
        let tui_plotter = TuiPlotter::new(plot_fitter_copy, plot_data_copy)
            .with_results(best_result.clone(), best_summary.clone());
        
        tui_plotter.run_interactive_plot()?;
    } else {
        println!("Generating diagnostic plots...");
        let visualizer = fit_rs::visualization::EC50Visualizer::new(&plot_fitter);
        visualizer.generate_all_plots(best_result, best_summary, &config.input.output_dir.to_string_lossy(), bounds)?;
    }

    // Calculate and display R-hat diagnostics
    if !args.interactive {
        calculate_rhat_improved(&chain_results);

        // Display final results
        display_final_results(best_summary, &config.input.output_dir.to_string_lossy());
    }

    Ok(())
}

fn generate_default_config(config_path: &PathBuf) -> Result<()> {
    let config = Config::default();
    config.to_file(config_path)?;
    println!("Generated default configuration file: {}", config_path.display());
    println!("Edit this file to customize your analysis parameters.");
    Ok(())
}

fn validate_config(config_path: &PathBuf) -> Result<()> {
    match Config::from_file(config_path) {
        Ok(_) => {
            println!("✅ Configuration file '{}' is valid", config_path.display());
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Configuration file '{}' is invalid: {}", config_path.display(), e);
            std::process::exit(1);
        }
    }
}

fn show_config(config_path: &PathBuf) -> Result<()> {
    let config = Config::from_file(config_path)?;
    
    println!("Configuration Summary");
    println!("===================");
    println!("File: {}", config_path.display());
    println!();
    println!("Input:");
    println!("  Data file: {}", config.input.file.display());
    println!("  Output directory: {}", config.input.output_dir.display());
    println!();
    println!("MCMC:");
    println!("  Backend: {}", config.mcmc.backend);
    println!("  Samples per chain: {}", config.mcmc.samples);
    println!("  Burnin samples: {}", config.mcmc.burnin);
    println!("  Number of chains: {}", config.mcmc.chains);
    println!("  Error sigma: {}", config.mcmc.sigma);
    println!();
    println!("Priors:");
    println!("  Emin: {} (μ={}, σ={})", config.priors.emin.prior_type, config.priors.emin.mean, config.priors.emin.std);
    println!("  Emax: {} (μ={}, σ={})", config.priors.emax.prior_type, config.priors.emax.mean, config.priors.emax.std);
    println!("  EC50: {} (μ={}, σ={})", config.priors.ec50.prior_type, config.priors.ec50.mean, config.priors.ec50.std);
    println!("  Hill slope: {} (μ={}, σ={})", config.priors.hillslope.prior_type, config.priors.hillslope.mean, config.priors.hillslope.std);
    println!();
    println!("Plotting:");
    println!("  Verbose: {}", config.plotting.verbose);
    if let Some((x_min, x_max, y_min, y_max)) = config.get_plotting_bounds() {
        println!("  Custom bounds: x=[{:.3}, {:.3}], y=[{:.3}, {:.3}]", x_min, x_max, y_min, y_max);
    } else {
        println!("  Bounds: automatic");
    }
    
    Ok(())
}

fn load_config(config_path: &PathBuf) -> Result<Config> {
    if !config_path.exists() {
        eprintln!("Configuration file '{}' not found.", config_path.display());
        eprintln!("Generate a default configuration with: {} --generate-config", 
                  std::env::args().next().unwrap_or_else(|| "fit-rs".to_string()));
        std::process::exit(1);
    }
    
    Config::from_file(config_path)
}

fn apply_cli_overrides(config: &mut Config, args: &Cli) {
    if let Some(ref file) = args.file {
        config.input.file = file.clone();
    }
    if let Some(ref output) = args.output {
        config.input.output_dir = output.clone();
    }
    if let Some(samples) = args.samples {
        config.mcmc.samples = samples;
    }
    if let Some(burnin) = args.burnin {
        config.mcmc.burnin = burnin;
    }
    if let Some(chains) = args.chains {
        config.mcmc.chains = chains;
    }
    if let Some(ref backend) = args.backend {
        config.mcmc.backend = backend.clone();
    }
    if let Some(verbose) = args.verbose {
        config.plotting.verbose = verbose;
    }
}

fn print_prior_summary(config: &Config) {
    println!("Prior specifications:");
    println!("  Emin ~ {} (μ={:.3}, σ={:.3}) - {}", 
             config.priors.emin.prior_type, 
             config.priors.emin.mean, 
             config.priors.emin.std,
             config.priors.emin.description.as_deref().unwrap_or(""));
    println!("  Emax ~ {} (μ={:.3}, σ={:.3}) - {}", 
             config.priors.emax.prior_type, 
             config.priors.emax.mean, 
             config.priors.emax.std,
             config.priors.emax.description.as_deref().unwrap_or(""));
    println!("  EC50 ~ {} (μ={:.3}, σ={:.3}) - {}", 
             config.priors.ec50.prior_type, 
             config.priors.ec50.mean, 
             config.priors.ec50.std,
             config.priors.ec50.description.as_deref().unwrap_or(""));
    println!("  Hill slope ~ {} (μ={:.3}, σ={:.3}) - {}", 
             config.priors.hillslope.prior_type, 
             config.priors.hillslope.mean, 
             config.priors.hillslope.std,
             config.priors.hillslope.description.as_deref().unwrap_or(""));
}

fn run_multiple_chains_improved(
    data: Vec<fit_rs::DoseResponse>,
    n_chains: usize,
    n_samples: usize,
    burnin: usize,
    priors: fit_rs::Prior,
    sigma: f64,
    backend: &str,
    verbose: bool,
) -> Result<Vec<(fit_rs::MCMCResult, fit_rs::ParameterSummary)>> {
    let mut all_results = Vec::new();

    for chain_id in 0..n_chains {
        if verbose {
            println!("Running chain {}/{}...", chain_id + 1, n_chains);
        } else {
            print!("Chain {}... ", chain_id + 1);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        let (result, summary) = match backend {
            "stan" => {
                let stan_sampler = StanSampler::new(data.clone(), priors.clone())?;
                let result = stan_sampler.fit(n_samples, burnin, Some(1))?;
                
                // Create a temporary fitter just for summarizing results
                let temp_fitter = BayesianEC50Fitter::new(data.clone())
                    .with_prior(priors.clone())
                    .with_sigma(sigma);
                let summary = temp_fitter.summarize_results(&result);
                
                if verbose {
                    println!("  Using {} backend", stan_sampler.get_name());
                }
                
                (result, summary)
            }
            _ => {
                // Default to Metropolis-Hastings
                let fitter = BayesianEC50Fitter::new(data.clone())
                    .with_prior(priors.clone())
                    .with_sigma(sigma);
                
                let result = fitter.fit(n_samples, burnin);
                let summary = fitter.summarize_results(&result);
                
                if verbose {
                    println!("  Using {} backend", fitter.get_name());
                }
                
                (result, summary)
            }
        };

        if verbose {
            println!("  Acceptance rate: {:.1}%", summary.acceptance_rate * 100.0);
            println!("  EC50 estimate: {:.3} ± {:.3}", summary.ec50.mean, summary.ec50.std);
            println!("  Max log-likelihood: {:.2}",
                     result.log_likelihood.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }

        all_results.push((result, summary));
    }

    if !verbose {
        println!("Done!");
    }

    Ok(all_results)
}

fn find_best_chain(results: &[(fit_rs::MCMCResult, fit_rs::ParameterSummary)]) -> usize {
    results
        .iter()
        .enumerate()
        .max_by(|(_, (result1, _)), (_, (result2, _))| {
            let max_ll1 = result1.log_likelihood.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let max_ll2 = result2.log_likelihood.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            max_ll1.partial_cmp(&max_ll2).unwrap()
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn calculate_rhat_improved(results: &[(fit_rs::MCMCResult, fit_rs::ParameterSummary)]) {
    if results.len() < 2 {
        println!("⚠️  Need at least 2 chains for R-hat calculation");
        return;
    }

    println!("\n=== Multi-Chain Convergence Diagnostics ===");
    
    let emin_rhat = calculate_rhat_for_param(results, |p| p.emin);
    let emax_rhat = calculate_rhat_for_param(results, |p| p.emax);
    let ec50_rhat = calculate_rhat_for_param(results, |p| p.ec50);
    let hillslope_rhat = calculate_rhat_for_param(results, |p| p.hillslope);

    let status_icon = |rhat: f64| {
        if rhat < 1.01 { "✅" }
        else if rhat < 1.05 { "⚠️ " }
        else { "❌" }
    };

    println!("  emin: R-hat = {:.3} {} {}", emin_rhat, status_icon(emin_rhat), 
             get_convergence_message(emin_rhat));
    println!("  emax: R-hat = {:.3} {} {}", emax_rhat, status_icon(emax_rhat),
             get_convergence_message(emax_rhat));
    println!("  ec50: R-hat = {:.3} {} {}", ec50_rhat, status_icon(ec50_rhat),
             get_convergence_message(ec50_rhat));
    println!("  hillslope: R-hat = {:.3} {} {}", hillslope_rhat, status_icon(hillslope_rhat),
             get_convergence_message(hillslope_rhat));

    let max_rhat = [emin_rhat, emax_rhat, ec50_rhat, hillslope_rhat]
        .iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\nOverall convergence assessment:");
    if max_rhat < 1.01 {
        println!("✅ Excellent convergence - chains have mixed well");
    } else if max_rhat < 1.05 {
        println!("⚠️  Good convergence - consider running more samples for final analysis");
    } else {
        println!("❌ Poor convergence detected - increase sample size or check model");
    }
}

fn calculate_rhat_for_param(
    results: &[(fit_rs::MCMCResult, fit_rs::ParameterSummary)],
    extract_param: fn(&fit_rs::LL4Parameters) -> f64,
) -> f64 {
    let n_chains = results.len();
    if n_chains < 2 { return 1.0; }
    
    let chain_means: Vec<f64> = results.iter()
        .map(|(result, _)| {
            let params: Vec<f64> = result.samples.iter().map(extract_param).collect();
            params.iter().sum::<f64>() / params.len() as f64
        })
        .collect();
    
    let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;
    
    let chain_vars: Vec<f64> = results.iter()
        .map(|(result, _)| {
            let params: Vec<f64> = result.samples.iter().map(extract_param).collect();
            let mean = params.iter().sum::<f64>() / params.len() as f64;
            params.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (params.len() - 1) as f64
        })
        .collect();
    
    let n = results[0].0.samples.len();
    let w = chain_vars.iter().sum::<f64>() / n_chains as f64;
    let b = chain_means.iter().map(|&mean| (mean - overall_mean).powi(2)).sum::<f64>() 
            * n as f64 / (n_chains - 1) as f64;
    
    let var_plus = ((n - 1) as f64 * w + b) / n as f64;
    (var_plus / w).sqrt()
}

fn get_convergence_message(rhat: f64) -> &'static str {
    if rhat < 1.01 {
        "excellent convergence"
    } else if rhat < 1.05 {
        "good convergence"
    } else if rhat < 1.1 {
        "acceptable - consider more samples"
    } else if rhat < 1.2 {
        "poor - increase samples"
    } else {
        "very poor - increase samples significantly"
    }
}

fn display_final_results(summary: &fit_rs::ParameterSummary, output_dir: &str) {
    println!("\n=== Final Results (Best Chain) ===");
    println!("Acceptance rate: {:.1}%", summary.acceptance_rate * 100.0);
    println!();
    println!("Parameter estimates (Mean ± SD [95% CI]):");
    println!("  Emin:      {:.4} ± {:.4} [{:.4}, {:.4}]",
             summary.emin.mean, summary.emin.std, summary.emin.ci_lower, summary.emin.ci_upper);
    println!("  Emax:      {:.4} ± {:.4} [{:.4}, {:.4}]",
             summary.emax.mean, summary.emax.std, summary.emax.ci_lower, summary.emax.ci_upper);
    println!("  EC50:      {:.4} ± {:.4} [{:.4}, {:.4}] (log10)",
             summary.ec50.mean, summary.ec50.std, summary.ec50.ci_lower, summary.ec50.ci_upper);
    println!("  EC50:      {:.2e} ({:.2e}, {:.2e}) (linear)",
             10.0_f64.powf(summary.ec50.mean),
             10.0_f64.powf(summary.ec50.ci_lower),
             10.0_f64.powf(summary.ec50.ci_upper));
    println!("  Hill:      {:.4} ± {:.4} [{:.4}, {:.4}]",
             summary.hillslope.mean, summary.hillslope.std, 
             summary.hillslope.ci_lower, summary.hillslope.ci_upper);
    
    println!("\nOutput files generated in '{}':", output_dir);
    println!("  dose_response_curve.png - Main fitted curve");
    println!("  mcmc_traces.png - Parameter convergence diagnostics");
    println!("  posterior_distributions.png - Parameter uncertainty");
    println!("  log_likelihood_trace.png - Sampling efficiency");
    println!("  coefficients.csv - Model coefficients");
    println!("  coefficients.json - Detailed results with diagnostics");
}
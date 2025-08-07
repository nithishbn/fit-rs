use anyhow::Result;
use clap::Parser;
use fit_rs::{io::load_csv, BayesianEC50Fitter, MCMCSampler, Prior, PriorType, StanSampler};

#[derive(Parser)]
#[command(name = "ec50-fit")]
#[command(about = "Bayesian EC50 curve fitting tool")]
#[command(version = "1.0")]
struct Cli {
    /// Input CSV file with 'concentration' and 'response' columns
    #[arg(short, long)]
    file: String,

    /// Output directory for results and plots
    #[arg(short, long, default_value = "ec50_output")]
    output: String,

    /// Number of MCMC samples per chain
    #[arg(long, default_value = "2000")]
    samples: usize,

    /// Number of burnin samples per chain
    #[arg(long, default_value = "1000")]
    burnin: usize,

    /// Number of parallel chains for convergence diagnostics
    #[arg(long, default_value = "4")]
    chains: usize,

    /// Error standard deviation (sigma)
    #[arg(long, default_value = "0.05")]
    sigma: f64,

    // Emin prior parameters
    /// Emin prior mean
    #[arg(long)]
    emin_mean: f64,

    /// Emin prior standard deviation
    #[arg(long)]
    emin_std: f64,

    // Emax prior parameters
    /// Emax prior mean
    #[arg(long)]
    emax_mean: f64,

    /// Emax prior standard deviation
    #[arg(long)]
    emax_std: f64,

    // EC50 prior parameters
    /// EC50 prior mean (log10 scale)
    #[arg(long)]
    ec50_mean: f64,

    /// EC50 prior standard deviation (log10 scale)
    #[arg(long)]
    ec50_std: f64,

    // Hill slope prior parameters
    /// Hill slope prior mean
    #[arg(long)]
    hillslope_mean: f64,

    /// Hill slope prior standard deviation
    #[arg(long)]
    hillslope_std: f64,

    /// Custom x-axis minimum (optional)
    #[arg(long)]
    x_min: Option<f64>,

    /// Custom x-axis maximum (optional)
    #[arg(long)]
    x_max: Option<f64>,

    /// Custom y-axis minimum (optional)
    #[arg(long)]
    y_min: Option<f64>,

    /// Custom y-axis maximum (optional)
    #[arg(long)]
    y_max: Option<f64>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// MCMC Backend ('mh' for Metropolis-Hastings, 'stan' for Stan/BridgeStan)
    #[arg(long, default_value = "mh")]
    backend: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    if args.verbose {
        println!("EC50 Bayesian Curve Fitting Tool (Multi-Chain)");
        println!("===============================================");
        println!("Input file: {}", args.file);
        println!("Output directory: {}", args.output);
        println!("MCMC chains: {}", args.chains);
        println!(
            "MCMC samples per chain: {} (burnin: {})",
            args.samples, args.burnin
        );
        println!("Error sigma: {}", args.sigma);
        println!();
    }

    // Load data
    if args.verbose {
        println!("Loading data from {}...", args.file);
    }
    let data = load_csv(&args.file)?;
    println!("Loaded {} data points", data.len());

    // Set up priors from command line arguments
    let priors = Prior {
        emin: PriorType::Normal {
            mean: args.emin_mean,
            std: args.emin_std,
        },
        emax: PriorType::Normal {
            mean: args.emax_mean,
            std: args.emax_std,
        },
        ec50: PriorType::Normal {
            mean: args.ec50_mean,
            std: args.ec50_std,
        },
        hillslope: PriorType::Normal {
            mean: args.hillslope_mean,
            std: args.hillslope_std,
        },
    };

    if args.verbose {
        println!("Prior specifications:");
        println!(
            "  Emin ~ Normal({:.3}, {:.3})",
            args.emin_mean, args.emin_std
        );
        println!(
            "  Emax ~ Normal({:.3}, {:.3})",
            args.emax_mean, args.emax_std
        );
        println!(
            "  EC50 ~ Normal({:.3}, {:.3}) [log10 scale]",
            args.ec50_mean, args.ec50_std
        );
        println!(
            "  Hill slope ~ Normal({:.3}, {:.3}) [lb = 0]",
            args.hillslope_mean, args.hillslope_std
        );
        println!();
    }

    // Set up plotting bounds if provided
    let bounds = match (args.x_min, args.x_max, args.y_min, args.y_max) {
        (Some(x_min), Some(x_max), Some(y_min), Some(y_max)) => {
            if args.verbose {
                println!(
                    "Using custom bounds: x=[{:.3}, {:.3}], y=[{:.3}, {:.3}]",
                    x_min, x_max, y_min, y_max
                );
            }
            Some((x_min, x_max, y_min, y_max))
        }
        _ => {
            if args.verbose {
                println!("Using automatic bounds");
            }
            None
        }
    };

    // Run multiple chains for better diagnostics
    println!(
        "Running {} chains with {} samples each...",
        args.chains, args.samples
    );
    let chain_results = run_multiple_chains_improved(
        data,
        args.chains,
        args.samples,
        args.burnin,
        priors,
        args.sigma,
        &args.backend,
        args.verbose,
    )?;

    // Use the best chain for plotting (highest log-likelihood)
    let best_chain_idx = find_best_chain(&chain_results);
    let (best_result, best_summary) = &chain_results[best_chain_idx];

    println!(
        "\nUsing chain {} (highest log-likelihood) for plotting",
        best_chain_idx + 1
    );

    // Load data again for plotting
    let plot_data = load_csv(&args.file)?;
    let plot_fitter = BayesianEC50Fitter::new(plot_data)
        .with_prior(Prior {
            emin: PriorType::Normal {
                mean: args.emin_mean,
                std: args.emin_std,
            },
            emax: PriorType::Normal {
                mean: args.emax_mean,
                std: args.emax_std,
            },
            ec50: PriorType::Normal {
                mean: args.ec50_mean,
                std: args.ec50_std,
            },
            hillslope: PriorType::Normal {
                mean: args.hillslope_mean,
                std: args.hillslope_std,
            },
        })
        .with_sigma(args.sigma);

    // Generate plots
    println!("Generating diagnostic plots...");
    let visualizer = fit_rs::visualization::EC50Visualizer::new(&plot_fitter);
    visualizer.generate_all_plots(best_result, best_summary, &args.output, bounds)?;

    // Calculate and display R-hat diagnostics
    calculate_rhat_improved(&chain_results);

    // Display final results from best chain
    display_final_results(best_summary, &args.output);

    Ok(())
}

fn run_multiple_chains_improved(
    data: Vec<fit_rs::DoseResponse>,
    n_chains: usize,
    n_samples: usize,
    burnin: usize,
    priors: Prior,
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
            println!(
                "  EC50 estimate: {:.3} ± {:.3}",
                summary.ec50.mean, summary.ec50.std
            );
            println!(
                "  Max log-likelihood: {:.2}",
                result
                    .log_likelihood
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            );
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
            let max_ll1 = result1
                .log_likelihood
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let max_ll2 = result2
                .log_likelihood
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
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

    // Calculate R-hat for each parameter
    let parameters = ["emin", "emax", "ec50", "hillslope"];

    for (param_idx, param_name) in parameters.iter().enumerate() {
        let mut chain_samples = Vec::new();

        for (result, _) in results {
            let param_samples: Vec<f64> = match param_idx {
                0 => result.samples.iter().map(|p| p.emin).collect(),
                1 => result.samples.iter().map(|p| p.emax).collect(),
                2 => result.samples.iter().map(|p| p.ec50).collect(),
                3 => result.samples.iter().map(|p| p.hillslope).collect(),
                _ => unreachable!(),
            };
            chain_samples.push(param_samples);
        }

        let rhat = calculate_rhat_for_parameter(&chain_samples);

        print!("  {}: R-hat = {:.3} ", param_name, rhat);
        if rhat < 1.01 {
            println!("✅ Excellent");
        } else if rhat < 1.05 {
            println!("✅ Good");
        } else if rhat < 1.1 {
            println!("⚠️  Acceptable");
        } else if rhat < 1.2 {
            println!("⚠️  Poor - consider more samples");
        } else {
            println!("❌ Very poor - increase samples significantly");
        }
    }

    // Overall assessment
    let all_rhats: Vec<f64> = parameters
        .iter()
        .enumerate()
        .map(|(param_idx, _)| {
            let mut chain_samples = Vec::new();
            for (result, _) in results {
                let param_samples: Vec<f64> = match param_idx {
                    0 => result.samples.iter().map(|p| p.emin).collect(),
                    1 => result.samples.iter().map(|p| p.emax).collect(),
                    2 => result.samples.iter().map(|p| p.ec50).collect(),
                    3 => result.samples.iter().map(|p| p.hillslope).collect(),
                    _ => unreachable!(),
                };
                chain_samples.push(param_samples);
            }
            calculate_rhat_for_parameter(&chain_samples)
        })
        .collect();

    let max_rhat = all_rhats.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\nOverall convergence assessment:");
    if max_rhat < 1.01 {
        println!("✅ Excellent convergence across all parameters");
    } else if max_rhat < 1.05 {
        println!("✅ Good convergence across all parameters");
    } else if max_rhat < 1.1 {
        println!("⚠️  Acceptable convergence, but consider more samples for publication");
    } else {
        println!("❌ Poor convergence detected - increase sample size or check model");
    }
}

fn calculate_rhat_for_parameter(chain_samples: &[Vec<f64>]) -> f64 {
    let n_chains = chain_samples.len();
    let n_samples = chain_samples[0].len();

    // Chain means
    let chain_means: Vec<f64> = chain_samples
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / chain.len() as f64)
        .collect();

    // Overall mean
    let overall_mean = chain_means.iter().sum::<f64>() / chain_means.len() as f64;

    // Between-chain variance
    let b = n_samples as f64
        * chain_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>()
        / (n_chains - 1) as f64;

    // Within-chain variance
    let w = chain_samples
        .iter()
        .map(|chain| {
            let mean = chain.iter().sum::<f64>() / chain.len() as f64;
            chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (chain.len() - 1) as f64
        })
        .sum::<f64>()
        / n_chains as f64;

    // R-hat estimate
    let var_plus = ((n_samples - 1) as f64 * w + b) / n_samples as f64;
    (var_plus / w).sqrt()
}

fn display_final_results(summary: &fit_rs::ParameterSummary, output_dir: &str) {
    println!("\n=== Final Results (Best Chain) ===");
    println!("Acceptance rate: {:.1}%", summary.acceptance_rate * 100.0);
    println!();
    println!("Parameter estimates (Mean ± SD [95% CI]):");
    println!(
        "  Emin:      {:.4} ± {:.4} [{:.4}, {:.4}]",
        summary.emin.mean, summary.emin.std, summary.emin.ci_lower, summary.emin.ci_upper
    );
    println!(
        "  Emax:      {:.4} ± {:.4} [{:.4}, {:.4}]",
        summary.emax.mean, summary.emax.std, summary.emax.ci_lower, summary.emax.ci_upper
    );
    println!(
        "  EC50:      {:.4} ± {:.4} [{:.4}, {:.4}] (log10)",
        summary.ec50.mean, summary.ec50.std, summary.ec50.ci_lower, summary.ec50.ci_upper
    );
    println!(
        "  EC50:      {:.2e} ({:.2e}, {:.2e}) (linear)",
        10.0_f64.powf(summary.ec50.mean),
        10.0_f64.powf(summary.ec50.ci_lower),
        10.0_f64.powf(summary.ec50.ci_upper)
    );
    println!(
        "  Hill:      {:.4} ± {:.4} [{:.4}, {:.4}]",
        summary.hillslope.mean,
        summary.hillslope.std,
        summary.hillslope.ci_lower,
        summary.hillslope.ci_upper
    );

    println!("\nOutput files generated in '{}':", output_dir);
    println!("  dose_response_curve.png - Main fitted curve");
    println!("  mcmc_traces.png - Parameter convergence diagnostics");
    println!("  posterior_distributions.png - Parameter uncertainty");
    println!("  log_likelihood_trace.png - Sampling efficiency");
    println!("  coefficients.csv - Model coefficients");
    println!("  coefficients.json - Detailed results with diagnostics");
}

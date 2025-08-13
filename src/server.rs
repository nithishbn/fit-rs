use axum::{extract::Multipart, http::{StatusCode, Method}, response::Json, routing::post, Router};
use fit_rs::{io::load_csv, BayesianEC50Fitter, MCMCSampler, Prior, PriorType, StanSampler, Config};
use serde::{Deserialize, Serialize};
use std::io::Write;
use tempfile::NamedTempFile;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FitRequest {
    // Legacy format for backward compatibility
    Legacy {
        // Prior parameters
        emin_mean: f64,
        emin_std: f64,
        emax_mean: f64,
        emax_std: f64,
        ec50_mean: f64,
        ec50_std: f64,
        hillslope_mean: f64,
        hillslope_std: f64,

        // MCMC parameters
        #[serde(default = "default_samples")]
        samples: usize,
        #[serde(default = "default_burnin")]
        burnin: usize,
        #[serde(default = "default_chains")]
        chains: usize,
        #[serde(default = "default_sigma")]
        sigma: f64,
        #[serde(default = "default_backend")]
        backend: String,
    },
    // New JSON config format
    JsonConfig(Config),
}

fn default_samples() -> usize {
    2000
}
fn default_burnin() -> usize {
    1000
}
fn default_chains() -> usize {
    4
}
fn default_sigma() -> f64 {
    0.05
}
fn default_backend() -> String {
    "mh".to_string()
}

#[derive(Debug, Serialize)]
struct FitResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_info: Option<ModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Parameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<Diagnostics>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    formula: String,
    n_samples_per_chain: usize,
    n_chains: usize,
    total_samples: usize,
    data_points: usize,
}

#[derive(Debug, Serialize)]
struct Parameters {
    emin: ParameterEstimate,
    emax: ParameterEstimate,
    ec50_log10: ParameterEstimate,
    ec50_linear: ParameterEstimate,
    hillslope: ParameterEstimate,
}

#[derive(Debug, Serialize)]
struct ParameterEstimate {
    mean: f64,
    median: f64,
    std: f64,
    ci_lower: f64,
    ci_upper: f64,
    description: String,
}

#[derive(Debug, Serialize)]
struct Diagnostics {
    acceptance_rate: f64,
    effective_sample_size: f64,
    rhat_estimates: RhatEstimates,
    convergence_warnings: Vec<String>,
    model_quality: String,
}

#[derive(Debug, Serialize)]
struct RhatEstimates {
    emin: f64,
    emax: f64,
    ec50: f64,
    hillslope: f64,
    max_rhat: f64,
    convergence_status: String,
}

async fn fit_curve(mut multipart: Multipart) -> Result<Json<FitResponse>, StatusCode> {
    let mut csv_data: Option<Vec<u8>> = None;
    let mut fit_params: Option<FitRequest> = None;

    // Parse multipart form data
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
    {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                csv_data = Some(data.to_vec());
            }
            "parameters" | "config" => {
                let data = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                fit_params = serde_json::from_str(&data).map_err(|_| StatusCode::BAD_REQUEST)?;
            }
            _ => {
                // Skip unknown fields
            }
        }
    }

    // Validate required fields
    let csv_data = csv_data.ok_or(StatusCode::BAD_REQUEST)?;
    let params = fit_params.ok_or(StatusCode::BAD_REQUEST)?;

    // Process the request
    match process_fit_request(csv_data, params).await {
        Ok(response) => Ok(Json(response)),
        Err(error_msg) => Ok(Json(FitResponse {
            success: false,
            error: Some(error_msg),
            model_info: None,
            parameters: None,
            diagnostics: None,
        })),
    }
}

async fn process_fit_request(csv_data: Vec<u8>, params: FitRequest) -> Result<FitResponse, String> {
    // Write CSV data to temporary file
    let mut temp_file =
        NamedTempFile::new().map_err(|e| format!("Failed to create temp file: {}", e))?;

    temp_file
        .write_all(&csv_data)
        .map_err(|e| format!("Failed to write CSV data: {}", e))?;

    // Load data from CSV
    let data = load_csv(temp_file.path()).map_err(|e| format!("Failed to parse CSV: {}", e))?;

    if data.is_empty() {
        return Err("No valid data points found in CSV".to_string());
    }

    // Extract parameters based on request type
    let (priors, mcmc_config) = match params {
        FitRequest::Legacy {
            emin_mean, emin_std, emax_mean, emax_std,
            ec50_mean, ec50_std, hillslope_mean, hillslope_std,
            samples, burnin, chains, sigma, backend
        } => {
            let priors = Prior {
                emin: PriorType::Normal { mean: emin_mean, std: emin_std },
                emax: PriorType::Normal { mean: emax_mean, std: emax_std },
                ec50: PriorType::Normal { mean: ec50_mean, std: ec50_std },
                hillslope: PriorType::Normal { mean: hillslope_mean, std: hillslope_std },
            };
            let mcmc_config = (samples, burnin, chains, sigma, backend);
            (priors, mcmc_config)
        },
        FitRequest::JsonConfig(config) => {
            // Validate the config
            config.validate().map_err(|e| format!("Invalid configuration: {}", e))?;
            
            let priors = config.to_prior().map_err(|e| format!("Failed to convert priors: {}", e))?;
            let mcmc_config = (
                config.mcmc.samples,
                config.mcmc.burnin,
                config.mcmc.chains,
                config.mcmc.sigma,
                config.mcmc.backend
            );
            (priors, mcmc_config)
        }
    };

    let (samples, burnin, chains, sigma, backend) = mcmc_config;

    // Run multiple chains
    let chain_results = run_multiple_chains(
        data.clone(),
        chains,
        samples,
        burnin,
        priors,
        sigma,
        &backend,
    )
    .map_err(|e| format!("MCMC fitting failed: {}", e))?;

    // Find best chain and calculate diagnostics
    let best_chain_idx = find_best_chain(&chain_results);
    let (_best_result, best_summary) = &chain_results[best_chain_idx];

    // Calculate R-hat for all parameters
    let rhat_estimates = calculate_rhat_diagnostics(&chain_results);

    // Create response
    let response = FitResponse {
        success: true,
        error: None,
        model_info: Some(ModelInfo {
            formula: "response ~ (emin + (emax - emin) / (1 + 10**(hillslope * (ec50 - log_concentration))))".to_string(),
            n_samples_per_chain: samples,
            n_chains: chains,
            total_samples: samples * chains,
            data_points: data.len(),
        }),
        parameters: Some(Parameters {
            emin: ParameterEstimate {
                mean: best_summary.emin.mean,
                median: best_summary.emin.median,
                std: best_summary.emin.std,
                ci_lower: best_summary.emin.ci_lower,
                ci_upper: best_summary.emin.ci_upper,
                description: "Lower asymptote (baseline response)".to_string(),
            },
            emax: ParameterEstimate {
                mean: best_summary.emax.mean,
                median: best_summary.emax.median,
                std: best_summary.emax.std,
                ci_lower: best_summary.emax.ci_lower,
                ci_upper: best_summary.emax.ci_upper,
                description: "Upper asymptote (maximum response)".to_string(),
            },
            ec50_log10: ParameterEstimate {
                mean: best_summary.ec50.mean,
                median: best_summary.ec50.median,
                std: best_summary.ec50.std,
                ci_lower: best_summary.ec50.ci_lower,
                ci_upper: best_summary.ec50.ci_upper,
                description: "EC50 on log10 scale".to_string(),
            },
            ec50_linear: ParameterEstimate {
                mean: 10.0_f64.powf(best_summary.ec50.mean),
                median: 10.0_f64.powf(best_summary.ec50.median),
                std: 10.0_f64.powf(best_summary.ec50.mean + best_summary.ec50.std) - 10.0_f64.powf(best_summary.ec50.mean),
                ci_lower: 10.0_f64.powf(best_summary.ec50.ci_lower),
                ci_upper: 10.0_f64.powf(best_summary.ec50.ci_upper),
                description: "EC50 on linear scale (concentration units)".to_string(),
            },
            hillslope: ParameterEstimate {
                mean: best_summary.hillslope.mean,
                median: best_summary.hillslope.median,
                std: best_summary.hillslope.std,
                ci_lower: best_summary.hillslope.ci_lower,
                ci_upper: best_summary.hillslope.ci_upper,
                description: "Hill slope (steepness of curve)".to_string(),
            },
        }),
        diagnostics: Some(Diagnostics {
            acceptance_rate: best_summary.acceptance_rate,
            effective_sample_size: best_summary.n_samples as f64 * best_summary.acceptance_rate,
            rhat_estimates,
            convergence_warnings: get_convergence_warnings(best_summary),
            model_quality: assess_model_quality(best_summary),
        }),
    };

    Ok(response)
}

fn run_multiple_chains(
    data: Vec<fit_rs::DoseResponse>,
    n_chains: usize,
    n_samples: usize,
    burnin: usize,
    priors: Prior,
    sigma: f64,
    backend: &str,
) -> Result<Vec<(fit_rs::MCMCResult, fit_rs::ParameterSummary)>, String> {
    let mut all_results = Vec::new();

    for _chain_id in 0..n_chains {
        let (result, summary) = match backend {
            "stan" => {
                let stan_sampler = StanSampler::new(data.clone(), priors.clone())
                    .map_err(|e| format!("Failed to create Stan sampler: {}", e))?;
                let result = stan_sampler.fit(n_samples, burnin, Some(1))
                    .map_err(|e| format!("Stan fitting failed: {}", e))?;
                
                // Create a temporary fitter just for summarizing results
                let temp_fitter = BayesianEC50Fitter::new(data.clone())
                    .with_prior(priors.clone())
                    .with_sigma(sigma);
                let summary = temp_fitter.summarize_results(&result);
                
                (result, summary)
            }
            _ => {
                // Default to Metropolis-Hastings
                let fitter = BayesianEC50Fitter::new(data.clone())
                    .with_prior(priors.clone())
                    .with_sigma(sigma);

                let result = fitter.fit(n_samples, burnin);
                let summary = fitter.summarize_results(&result);
                
                (result, summary)
            }
        };

        all_results.push((result, summary));
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

fn calculate_rhat_diagnostics(
    results: &[(fit_rs::MCMCResult, fit_rs::ParameterSummary)],
) -> RhatEstimates {
    if results.len() < 2 {
        return RhatEstimates {
            emin: 1.0,
            emax: 1.0,
            ec50: 1.0,
            hillslope: 1.0,
            max_rhat: 1.0,
            convergence_status: "Single chain - R-hat not available".to_string(),
        };
    }

    let emin_rhat = calculate_rhat_for_param(results, |p| p.emin);
    let emax_rhat = calculate_rhat_for_param(results, |p| p.emax);
    let ec50_rhat = calculate_rhat_for_param(results, |p| p.ec50);
    let hillslope_rhat = calculate_rhat_for_param(results, |p| p.hillslope);

    let max_rhat = [emin_rhat, emax_rhat, ec50_rhat, hillslope_rhat]
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let convergence_status = if max_rhat < 1.01 {
        "Excellent".to_string()
    } else if max_rhat < 1.05 {
        "Good".to_string()
    } else if max_rhat < 1.1 {
        "Acceptable".to_string()
    } else if max_rhat < 1.2 {
        "Poor".to_string()
    } else {
        "Very Poor".to_string()
    };

    RhatEstimates {
        emin: emin_rhat,
        emax: emax_rhat,
        ec50: ec50_rhat,
        hillslope: hillslope_rhat,
        max_rhat,
        convergence_status,
    }
}

fn calculate_rhat_for_param<F>(
    results: &[(fit_rs::MCMCResult, fit_rs::ParameterSummary)],
    param_extractor: F,
) -> f64
where
    F: Fn(&fit_rs::LL4Parameters) -> f64,
{
    let chain_samples: Vec<Vec<f64>> = results
        .iter()
        .map(|(result, _)| result.samples.iter().map(|p| param_extractor(p)).collect())
        .collect();

    let n_chains = chain_samples.len();
    let n_samples = chain_samples[0].len();

    let chain_means: Vec<f64> = chain_samples
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / chain.len() as f64)
        .collect();

    let overall_mean = chain_means.iter().sum::<f64>() / chain_means.len() as f64;

    let b = n_samples as f64
        * chain_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>()
        / (n_chains - 1) as f64;

    let w = chain_samples
        .iter()
        .map(|chain| {
            let mean = chain.iter().sum::<f64>() / chain.len() as f64;
            chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (chain.len() - 1) as f64
        })
        .sum::<f64>()
        / n_chains as f64;

    let var_plus = ((n_samples - 1) as f64 * w + b) / n_samples as f64;
    (var_plus / w).sqrt()
}

fn get_convergence_warnings(summary: &fit_rs::ParameterSummary) -> Vec<String> {
    let mut warnings = Vec::new();

    if summary.acceptance_rate < 0.2 {
        warnings.push(format!(
            "Low acceptance rate ({:.1}%) - consider adjusting priors",
            summary.acceptance_rate * 100.0
        ));
    } else if summary.acceptance_rate > 0.7 {
        warnings.push(format!(
            "High acceptance rate ({:.1}%) - proposals might be too small",
            summary.acceptance_rate * 100.0
        ));
    }

    let ec50_cv = summary.ec50.std / summary.ec50.mean.abs();
    if ec50_cv > 0.5 {
        warnings.push(format!(
            "High uncertainty in EC50 estimate (CV = {:.1}%)",
            ec50_cv * 100.0
        ));
    }

    let eff_sample_size = summary.n_samples as f64 * summary.acceptance_rate;
    if eff_sample_size < 400.0 {
        warnings.push(format!(
            "Low effective sample size (~{:.0}) - consider more samples",
            eff_sample_size
        ));
    }

    warnings
}

fn assess_model_quality(summary: &fit_rs::ParameterSummary) -> String {
    let warnings = get_convergence_warnings(summary);

    if warnings.is_empty() {
        "Good - No major issues detected".to_string()
    } else if warnings.len() <= 2 {
        "Fair - Minor issues detected".to_string()
    } else {
        "Poor - Multiple issues detected".to_string()
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "ec50-fitting-api",
        "version": "1.0.0"
    }))
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt().init();
    // Build the router
    let app = Router::new()
        .route("/fit_curve", post(fit_curve))
        .route("/health", axum::routing::get(health_check))
        .layer(
            ServiceBuilder::new()
                .layer(
                    CorsLayer::new()
                        .allow_origin(tower_http::cors::Any)
                        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                        .allow_headers(tower_http::cors::Any)
                        .allow_credentials(false)
                )
                .into_inner(),
        );

    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind to port 3000");

    println!("🚀 EC50 Fitting API server starting on http://0.0.0.0:3000");
    println!("📊 POST /fit_curve - Fit EC50 curves with uploaded CSV data");
    println!("❤️  GET /health - Health check endpoint");

    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}

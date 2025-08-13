use askama::{Template};
use axum::{
    extract::{Multipart, State},
    http::header,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Router,
};
use base64::Engine;
use chrono;
use fit_rs::{io::load_csv, BayesianEC50Fitter, Config, DoseResponse, LL4Parameters, MCMCSampler, Prior, StanSampler};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, sync::Arc};
use tempfile::NamedTempFile;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    has_data: bool,
    has_config: bool,
    current_config: Option<String>,
    plot_data: Option<String>,
}

#[derive(Template)]
#[template(path = "parameter_form.html")]
struct ParameterFormTemplate {
    config: Config,
}

#[derive(Template)]
#[template(path = "plot_update.html")]
struct PlotUpdateTemplate {
    plot_data: String,
    fit_results: Option<FitResultsDisplay>,
}

#[derive(Serialize, Deserialize, Clone)]
struct FitResultsDisplay {
    emin: ParamDisplay,
    emax: ParamDisplay,
    ec50_log: ParamDisplay,
    ec50_linear: ParamDisplay,
    hillslope: ParamDisplay,
    acceptance_rate: f64,
    diagnostics: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ParamDisplay {
    mean: f64,
    std: f64,
    ci_lower: f64,
    ci_upper: f64,
}

#[derive(Clone)]
struct AppState {
    data: Arc<RwLock<Option<Vec<DoseResponse>>>>,
    config: Arc<RwLock<Option<Config>>>,
    fit_results: Arc<RwLock<Option<FitResultsDisplay>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            data: Arc::new(RwLock::new(None)),
            config: Arc::new(RwLock::new(None)),
            fit_results: Arc::new(RwLock::new(None)),
        }
    }
}

async fn index(State(state): State<AppState>) -> impl IntoResponse {
    let data = state.data.read().await;
    let config = state.config.read().await;
    
    let has_data = data.is_some();
    let has_config = config.is_some();
    let current_config = config.as_ref().map(|c| serde_json::to_string_pretty(c).unwrap_or_default());
    
    // Generate a simple plot if we have data
    let plot_data = if let Some(ref data_vec) = *data {
        match generate_data_plot(data_vec).await {
            Ok(plot) if !plot.is_empty() => Some(plot),
            Ok(_) => None,
            Err(e) => {
                eprintln!("Error generating plot: {}", e);
                None
            }
        }
    } else {
        None
    };

    let template = IndexTemplate {
        has_data,
        has_config,
        current_config,
        plot_data,
    };
    
    Html(template.render().unwrap())
}

async fn upload_data(State(state): State<AppState>, mut multipart: Multipart) -> impl IntoResponse {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "data_file" {
            let data = field.bytes().await.unwrap();
            
            // Write to temp file and parse
            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(&data).unwrap();
            
            match load_csv(temp_file.path()) {
                Ok(parsed_data) => {
                    *state.data.write().await = Some(parsed_data.clone());
                    
                    // Generate initial plot
                    let plot_data = generate_data_plot(&parsed_data).await.unwrap_or_else(|e| {
                        eprintln!("Error generating plot: {}", e);
                        String::new()
                    });
                    
                    return Html(format!(
                        r#"
                        <div id="data-status" class="alert alert-success">
                            ✅ Data uploaded successfully! {} data points loaded.
                        </div>
                        <div id="plot-container">
                            <img src="data:image/png;base64,{}" alt="Data Plot" class="img-fluid">
                        </div>
                        "#,
                        parsed_data.len(),
                        plot_data
                    ));
                }
                Err(e) => {
                    return Html(format!(
                        r#"<div id="data-status" class="alert alert-danger">❌ Error parsing CSV: {}</div>"#,
                        e
                    ));
                }
            }
        }
    }
    
    Html(r#"<div id="data-status" class="alert alert-danger">❌ No data file received</div>"#.to_string())
}

async fn upload_config(State(state): State<AppState>, mut multipart: Multipart) -> impl IntoResponse {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "config_file" {
            let data = field.text().await.unwrap();
            
            match serde_json::from_str::<Config>(&data) {
                Ok(config) => {
                    if let Err(e) = config.validate() {
                        return Html(format!(
                            r#"<div id="config-status" class="alert alert-danger">❌ Invalid config: {}</div>"#,
                            e
                        ));
                    }
                    
                    *state.config.write().await = Some(config.clone());
                    
                    return Html(format!(
                        r#"
                        <div id="config-status" class="alert alert-success">
                            ✅ Configuration uploaded successfully!
                        </div>
                        <div id="parameter-form-container" hx-get="/parameter-form" hx-trigger="load">
                            Loading parameter form...
                        </div>
                        "#
                    ));
                }
                Err(e) => {
                    return Html(format!(
                        r#"<div id="config-status" class="alert alert-danger">❌ Error parsing JSON: {}</div>"#,
                        e
                    ));
                }
            }
        }
    }
    
    Html(r#"<div id="config-status" class="alert alert-danger">❌ No config file received</div>"#.to_string())
}

async fn parameter_form(State(state): State<AppState>) -> impl IntoResponse {
    let config_guard = state.config.read().await;
    
    if let Some(config) = config_guard.as_ref() {
        let template = ParameterFormTemplate {
            config: config.clone(),
        };
        Html(template.render().unwrap())
    } else {
        Html("<div class=\"alert alert-warning\">No configuration loaded. Please upload a parameters.json file first.</div>".to_string())
    }
}

#[derive(Deserialize)]
struct UpdateParametersForm {
    emin_mean: f64,
    emin_std: f64,
    emax_mean: f64,
    emax_std: f64,
    ec50_mean: f64,
    ec50_std: f64,
    hillslope_mean: f64,
    hillslope_std: f64,
    samples: usize,
    burnin: usize,
    chains: usize,
    sigma: f64,
    backend: String,
}

async fn update_parameters(
    State(state): State<AppState>,
    axum::extract::Form(form): axum::extract::Form<UpdateParametersForm>,
) -> impl IntoResponse {
    let data_guard = state.data.read().await;
    let mut config_guard = state.config.write().await;
    
    let Some(data) = data_guard.as_ref() else {
        return Html(r#"<div class="alert alert-danger">❌ No data loaded. Please upload a CSV file first.</div>"#.to_string());
    };
    
    let Some(config) = config_guard.as_mut() else {
        return Html(r#"<div class="alert alert-danger">❌ No configuration loaded. Please upload a parameters.json file first.</div>"#.to_string());
    };
    
    // Update config with new parameters
    config.priors.emin.mean = form.emin_mean;
    config.priors.emin.std = form.emin_std;
    config.priors.emax.mean = form.emax_mean;
    config.priors.emax.std = form.emax_std;
    config.priors.ec50.mean = form.ec50_mean;
    config.priors.ec50.std = form.ec50_std;
    config.priors.hillslope.mean = form.hillslope_mean;
    config.priors.hillslope.std = form.hillslope_std;
    
    config.mcmc.samples = form.samples;
    config.mcmc.burnin = form.burnin;
    config.mcmc.chains = form.chains;
    config.mcmc.sigma = form.sigma;
    config.mcmc.backend = form.backend;
    
    if let Err(e) = config.validate() {
        return Html(format!(
            r#"<div class="alert alert-danger">❌ Invalid parameters: {}</div>"#,
            e
        ));
    }
    
    // Run the fitting process
    let priors = match config.to_prior() {
        Ok(p) => p,
        Err(e) => {
            return Html(format!(
                r#"<div class="alert alert-danger">❌ Error converting priors: {}</div>"#,
                e
            ));
        }
    };
    
    let fitting_result = run_mcmc_fitting(
        data.clone(),
        priors,
        config.mcmc.samples,
        config.mcmc.burnin,
        config.mcmc.chains,
        config.mcmc.sigma,
        &config.mcmc.backend,
    ).await;
    
    match fitting_result {
        Ok((plot_data, fit_results)) => {
            *state.fit_results.write().await = Some(fit_results.clone());
            
            let template = PlotUpdateTemplate {
                plot_data,
                fit_results: Some(fit_results),
            };
            Html(template.render().unwrap())
        }
        Err(e) => {
            Html(format!(
                r#"<div class="alert alert-danger">❌ Fitting failed: {}</div>"#,
                e
            ))
        }
    }
}

async fn run_mcmc_fitting(
    data: Vec<DoseResponse>,
    priors: Prior,
    samples: usize,
    burnin: usize,
    _chains: usize,
    sigma: f64,
    backend: &str,
) -> Result<(String, FitResultsDisplay), String> {
    // Run a single chain for speed in the web interface
    let (result, summary) = match backend {
        "stan" => {
            let stan_sampler = StanSampler::new(data.clone(), priors.clone())
                .map_err(|e| format!("Failed to create Stan sampler: {}", e))?;
            let result = stan_sampler.fit(samples, burnin, Some(1))
                .map_err(|e| format!("Stan fitting failed: {}", e))?;
            
            let temp_fitter = BayesianEC50Fitter::new(data.clone())
                .with_prior(priors.clone())
                .with_sigma(sigma);
            let summary = temp_fitter.summarize_results(&result);
            
            (result, summary)
        }
        _ => {
            let fitter = BayesianEC50Fitter::new(data.clone())
                .with_prior(priors.clone())
                .with_sigma(sigma);
            
            let result = fitter.fit(samples, burnin);
            let summary = fitter.summarize_results(&result);
            
            (result, summary)
        }
    };
    
    // Generate plot with fitted curve
    let plot_data = generate_fitted_plot(&data, &result, &summary).await
        .map_err(|e| format!("Failed to generate plot: {}", e))?;
    
    let fit_display = FitResultsDisplay {
        emin: ParamDisplay {
            mean: summary.emin.mean,
            std: summary.emin.std,
            ci_lower: summary.emin.ci_lower,
            ci_upper: summary.emin.ci_upper,
        },
        emax: ParamDisplay {
            mean: summary.emax.mean,
            std: summary.emax.std,
            ci_lower: summary.emax.ci_lower,
            ci_upper: summary.emax.ci_upper,
        },
        ec50_log: ParamDisplay {
            mean: summary.ec50.mean,
            std: summary.ec50.std,
            ci_lower: summary.ec50.ci_lower,
            ci_upper: summary.ec50.ci_upper,
        },
        ec50_linear: ParamDisplay {
            mean: 10.0_f64.powf(summary.ec50.mean),
            std: 10.0_f64.powf(summary.ec50.mean + summary.ec50.std) - 10.0_f64.powf(summary.ec50.mean),
            ci_lower: 10.0_f64.powf(summary.ec50.ci_lower),
            ci_upper: 10.0_f64.powf(summary.ec50.ci_upper),
        },
        hillslope: ParamDisplay {
            mean: summary.hillslope.mean,
            std: summary.hillslope.std,
            ci_lower: summary.hillslope.ci_lower,
            ci_upper: summary.hillslope.ci_upper,
        },
        acceptance_rate: summary.acceptance_rate * 100.0,
        diagnostics: format!("Effective sample size: ~{:.0}", summary.n_samples as f64 * summary.acceptance_rate),
    };
    
    Ok((plot_data, fit_display))
}

async fn generate_data_plot(data: &[DoseResponse]) -> Result<String, Box<dyn std::error::Error>> {
    let temp_file = tempfile::NamedTempFile::with_suffix(".png")?;
    let temp_path = temp_file.path();
    
    {
        let root = BitMapBackend::new(temp_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let data_points: Vec<(f64, f64)> = data
            .iter()
            .map(|d| (d.concentration.log10(), d.response))
            .collect();
        
        if data_points.is_empty() {
            return Ok(String::new());
        }
        
        let x_min = data_points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min) - 0.5;
        let x_max = data_points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max) + 0.5;
        let y_min = data_points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min) - 0.1;
        let y_max = data_points.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) + 0.1;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Uploaded Data", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
        
        chart
            .configure_mesh()
            .x_desc("Log10(Concentration)")
            .y_desc("Response")
            .draw()?;
        
        chart.draw_series(
            data_points.iter().map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled()))
        )?
        .label("Data Points")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));
        
        chart.configure_series_labels().draw()?;
        root.present()?;
        drop(root);
    }
    
    // Read the file and encode as base64
    let image_data = std::fs::read(temp_path)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&image_data))
}

async fn generate_fitted_plot(
    data: &[DoseResponse],
    _results: &fit_rs::MCMCResult,
    summary: &fit_rs::ParameterSummary,
) -> Result<String, Box<dyn std::error::Error>> {
    let temp_file = tempfile::NamedTempFile::with_suffix(".png")?;
    let temp_path = temp_file.path();
    
    {
        let root = BitMapBackend::new(temp_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let data_points: Vec<(f64, f64)> = data
            .iter()
            .map(|d| (d.concentration.log10(), d.response))
            .collect();
        
        let x_min = data_points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min) - 1.0;
        let x_max = data_points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max) + 1.0;
        let y_min = data_points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min) - 0.1;
        let y_max = data_points.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) + 0.1;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("EC50 Curve Fit", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
        
        chart
            .configure_mesh()
            .x_desc("Log10(Concentration)")
            .y_desc("Response")
            .draw()?;
        
        // Generate fitted curve
        let params = LL4Parameters {
            emin: summary.emin.mean,
            emax: summary.emax.mean,
            ec50: summary.ec50.mean,
            hillslope: summary.hillslope.mean,
        };
        
        let curve_points: Vec<(f64, f64)> = (0..200)
            .map(|i| {
                let log_conc = x_min + (x_max - x_min) * i as f64 / 199.0;
                let conc = 10.0_f64.powf(log_conc);
                let response = ll4_model(conc, &params);
                (log_conc, response)
            })
            .collect();
        
        // Draw fitted curve
        chart.draw_series(LineSeries::new(curve_points.iter().cloned(), &RED))?
            .label("Fitted Curve")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));
        
        // Draw data points
        chart.draw_series(
            data_points.iter().map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled()))
        )?
        .label("Data Points")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));
        
        // Draw EC50 line
        let ec50_line = vec![(summary.ec50.mean, y_min), (summary.ec50.mean, y_max)];
        chart.draw_series(LineSeries::new(ec50_line.iter().cloned(), GREEN.stroke_width(2)))?
            .label(&format!("EC50: {:.2}", 10.0_f64.powf(summary.ec50.mean)))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], GREEN));
        
        chart.configure_series_labels().draw()?;
        root.present()?;
        drop(root);
    }
    
    // Read the file and encode as base64
    let image_data = std::fs::read(temp_path)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&image_data))
}

fn ll4_model(concentration: f64, params: &LL4Parameters) -> f64 {
    params.emin + (params.emax - params.emin) / (1.0 + 10.0_f64.powf(params.hillslope * (params.ec50 - concentration.log10())))
}

async fn download_csv(State(state): State<AppState>) -> impl IntoResponse {
    let fit_results = state.fit_results.read().await;
    
    if let Some(results) = fit_results.as_ref() {
        let csv_content = format!(
            "Parameter,Mean,Std_Dev,CI_Lower,CI_Upper\n\
            Emin,{},{},{},{}\n\
            Emax,{},{},{},{}\n\
            EC50_log10,{},{},{},{}\n\
            EC50_linear,{},{},{},{}\n\
            Hillslope,{},{},{},{}\n\
            Acceptance_Rate_Percent,{},,,\n",
            results.emin.mean, results.emin.std, results.emin.ci_lower, results.emin.ci_upper,
            results.emax.mean, results.emax.std, results.emax.ci_lower, results.emax.ci_upper,
            results.ec50_log.mean, results.ec50_log.std, results.ec50_log.ci_lower, results.ec50_log.ci_upper,
            results.ec50_linear.mean, results.ec50_linear.std, results.ec50_linear.ci_lower, results.ec50_linear.ci_upper,
            results.hillslope.mean, results.hillslope.std, results.hillslope.ci_lower, results.hillslope.ci_upper,
            results.acceptance_rate
        );
        
        let filename = format!("ec50_results_{}.csv", 
                              chrono::Utc::now().format("%Y%m%d_%H%M%S"));
        
        Response::builder()
            .header(header::CONTENT_TYPE, "text/csv")
            .header(header::CONTENT_DISPOSITION, 
                   format!("attachment; filename=\"{}\"", filename))
            .body(csv_content)
            .unwrap()
    } else {
        Response::builder()
            .status(404)
            .body("No results available for download".to_string())
            .unwrap()
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    let state = AppState::default();
    
    let app = Router::new()
        .route("/", get(index))
        .route("/upload-data", post(upload_data))
        .route("/upload-config", post(upload_config))
        .route("/parameter-form", get(parameter_form))
        .route("/update-parameters", post(update_parameters))
        .route("/download-csv", get(download_csv))
        .nest_service("/static", ServeDir::new("static"))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        )
        .with_state(state);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001")
        .await
        .expect("Failed to bind to port 3001");
    
    println!("🚀 HTMX EC50 Fitting Server starting on http://0.0.0.0:3001");
    println!("📊 Web interface for EC50 curve fitting with real-time updates");
    
    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}
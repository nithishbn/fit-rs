use askama::Template;
use axum::{
    extract::{Multipart, State},
    http::header,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Router,
};
use chrono;
use fit_rs::{
    io::load_csv, BayesianEC50Fitter, Config, DoseResponse, LL4Parameters, MCMCSampler, Prior,
    StanSampler,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
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
    plot_json: Option<String>,
}

#[derive(Template)]
#[template(path = "parameter_form.html")]
struct ParameterFormTemplate {
    config: Config,
}

#[derive(Template)]
#[template(path = "fit_results.html")]
struct FitResultsTemplate {
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
    acceptance_rate_formatted: String,
    diagnostics: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ParamDisplay {
    mean: f64,
    std: f64,
    ci_lower: f64,
    ci_upper: f64,
    mean_formatted: String,
    std_formatted: String,
    ci_lower_formatted: String,
    ci_upper_formatted: String,
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
    let current_config = config
        .as_ref()
        .map(|c| serde_json::to_string_pretty(c).unwrap_or_default());

    // Generate plot data JSON if we have data
    let plot_json = if let Some(ref data_vec) = *data {
        match generate_data_plot_json(data_vec).await {
            Ok(plot) if !plot.is_empty() => Some(plot),
            Ok(_) => None,
            Err(e) => {
                eprintln!("Error generating plot data: {}", e);
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
        plot_json,
    };

    Html(template.render().unwrap())
}

async fn upload_files(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut csv_data: Option<Vec<u8>> = None;
    let mut config_data: Option<String> = None;
    let mut data_loaded = false;
    let mut config_loaded = false;

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();

        if name == "files" {
            let filename = field.file_name().unwrap_or("").to_string();
            let data = field.bytes().await.unwrap();

            if filename.ends_with(".csv") {
                csv_data = Some(data.to_vec());
            } else if filename.ends_with(".json") {
                config_data = Some(String::from_utf8_lossy(&data).to_string());
            }
        }
    }

    let mut responses = Vec::new();

    // Process CSV data if uploaded
    if let Some(csv_bytes) = csv_data {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&csv_bytes).unwrap();

        match load_csv(temp_file.path()) {
            Ok(parsed_data) => {
                *state.data.write().await = Some(parsed_data.clone());
                data_loaded = true;

                // Generate initial plot data and update main plot container
                let plot_json = generate_data_plot_json(&parsed_data)
                    .await
                    .unwrap_or_else(|e| {
                        eprintln!("Error generating plot data: {}", e);
                        String::new()
                    });

                if !plot_json.is_empty() {
                    responses.push(format!(
                        r#"<div hx-swap-oob="innerHTML:#main-plot-container">
                            <div id="plotly-div" style="width:100%;height:400px;"></div>
                            <script>
                                Plotly.newPlot('plotly-div', {});
                            </script>
                        </div>"#,
                        plot_json
                    ));
                }
            }
            Err(e) => {
                responses.push(format!("❌ Error parsing CSV: {}", e));
            }
        }
    }

    // Process JSON config if uploaded
    if let Some(json_data) = config_data {
        match serde_json::from_str::<Config>(&json_data) {
            Ok(config) => {
                if let Err(e) = config.validate() {
                    responses.push(format!("❌ Invalid config: {}", e));
                } else {
                    *state.config.write().await = Some(config.clone());
                    config_loaded = true;
                }
            }
            Err(e) => {
                responses.push(format!("❌ Error parsing JSON: {}", e));
            }
        }
    }

    // Update status and trigger parameter form load if both files uploaded
    let status_html = if data_loaded && config_loaded {
        "<small class=\"text-success\">✅ Data & Config loaded</small>\n        <div hx-get=\"/parameter-form\" hx-target=\"#parameter-form-container\" hx-trigger=\"load\"></div>".to_string()
    } else if data_loaded {
        r#"<small class="text-warning">⚠️ Data loaded, need config</small>"#.to_string()
    } else if config_loaded {
        r#"<small class="text-warning">⚠️ Config loaded, need data</small>"#.to_string()
    } else {
        r#"<small class="text-danger">❌ No valid files uploaded</small>"#.to_string()
    };

    // Return combined response targeting both plot area and status
    let combined_response = if !responses.is_empty() && data_loaded {
        format!(
            r#"<div hx-swap-oob="innerHTML:#upload-status">{}</div>
            <div hx-swap-oob="innerHTML:#parameter-form-container">
                {}
            </div>
            {}"#,
            status_html,
            if config_loaded {
                r#"<div hx-get="/parameter-form" hx-trigger="load"></div>"#
            } else {
                ""
            },
            responses.join("")
        )
    } else {
        format!(
            r#"<div hx-swap-oob="innerHTML:#upload-status">{}</div>"#,
            status_html
        )
    };

    Html(combined_response)
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

                    // Generate initial plot data
                    let plot_json =
                        generate_data_plot_json(&parsed_data)
                            .await
                            .unwrap_or_else(|e| {
                                eprintln!("Error generating plot data: {}", e);
                                String::new()
                            });

                    return Html(format!(
                        r#"
                        <div id="data-status" class="alert alert-success">
                            ✅ Data uploaded successfully! {} data points loaded.
                        </div>
                        <div id="plot-container">
                            <div id="plotly-div" style="width:100%;height:400px;"></div>
                            <script>
                                Plotly.newPlot('plotly-div', {});
                            </script>
                        </div>
                        "#,
                        parsed_data.len(),
                        plot_json
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

    Html(
        r#"<div id="data-status" class="alert alert-danger">❌ No data file received</div>"#
            .to_string(),
    )
}

async fn upload_config(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
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

    Html(
        r#"<div id="config-status" class="alert alert-danger">❌ No config file received</div>"#
            .to_string(),
    )
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
) -> Response {
    let data_guard = state.data.read().await;
    let mut config_guard = state.config.write().await;

    let Some(data) = data_guard.as_ref() else {
        return Html(r#"<div class="alert alert-danger">❌ No data loaded. Please upload a CSV file first.</div>"#).into_response();
    };

    let Some(config) = config_guard.as_mut() else {
        return Html(r#"<div class="alert alert-danger">❌ No configuration loaded. Please upload a parameters.json file first.</div>"#).into_response();
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
        ))
        .into_response();
    }

    // Run the fitting process
    let priors = match config.to_prior() {
        Ok(p) => p,
        Err(e) => {
            return Html(format!(
                r#"<div class="alert alert-danger">❌ Error converting priors: {}</div>"#,
                e
            ))
            .into_response();
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
    )
    .await;

    match fitting_result {
        Ok((plot_json, fit_results)) => {
            *state.fit_results.write().await = Some(fit_results.clone());

            // Create the results template
            let template = FitResultsTemplate {
                fit_results: Some(fit_results),
            };

            // Return response with HX-Trigger header containing plot data
            let html_content = template.render().unwrap();
            Response::builder()
                .header("HX-Trigger", format!("{{\"updatePlot\": {}}}", plot_json))
                .header("Content-Type", "text/html")
                .body(html_content.into())
                .unwrap()
        }
        Err(e) => Html(format!(
            r#"<div class="alert alert-danger">❌ Fitting failed: {}</div>"#,
            e
        ))
        .into_response(),
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
            let result = stan_sampler
                .fit(samples, burnin, Some(1))
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

    // Generate plot data with fitted curve
    let plot_json = generate_fitted_plot_json(&data, &result, &summary)
        .await
        .map_err(|e| format!("Failed to generate plot data: {}", e))?;

    let ec50_linear_mean = 10.0_f64.powf(summary.ec50.mean);
    let ec50_linear_std =
        10.0_f64.powf(summary.ec50.mean + summary.ec50.std) - 10.0_f64.powf(summary.ec50.mean);
    let ec50_linear_ci_lower = 10.0_f64.powf(summary.ec50.ci_lower);
    let ec50_linear_ci_upper = 10.0_f64.powf(summary.ec50.ci_upper);
    let acceptance_rate_percent = summary.acceptance_rate * 100.0;

    let fit_display = FitResultsDisplay {
        emin: ParamDisplay {
            mean: summary.emin.mean,
            std: summary.emin.std,
            ci_lower: summary.emin.ci_lower,
            ci_upper: summary.emin.ci_upper,
            mean_formatted: format_4_sig_figs(summary.emin.mean),
            std_formatted: format_4_sig_figs(summary.emin.std),
            ci_lower_formatted: format_4_sig_figs(summary.emin.ci_lower),
            ci_upper_formatted: format_4_sig_figs(summary.emin.ci_upper),
        },
        emax: ParamDisplay {
            mean: summary.emax.mean,
            std: summary.emax.std,
            ci_lower: summary.emax.ci_lower,
            ci_upper: summary.emax.ci_upper,
            mean_formatted: format_4_sig_figs(summary.emax.mean),
            std_formatted: format_4_sig_figs(summary.emax.std),
            ci_lower_formatted: format_4_sig_figs(summary.emax.ci_lower),
            ci_upper_formatted: format_4_sig_figs(summary.emax.ci_upper),
        },
        ec50_log: ParamDisplay {
            mean: summary.ec50.mean,
            std: summary.ec50.std,
            ci_lower: summary.ec50.ci_lower,
            ci_upper: summary.ec50.ci_upper,
            mean_formatted: format_4_sig_figs(summary.ec50.mean),
            std_formatted: format_4_sig_figs(summary.ec50.std),
            ci_lower_formatted: format_4_sig_figs(summary.ec50.ci_lower),
            ci_upper_formatted: format_4_sig_figs(summary.ec50.ci_upper),
        },
        ec50_linear: ParamDisplay {
            mean: ec50_linear_mean,
            std: ec50_linear_std,
            ci_lower: ec50_linear_ci_lower,
            ci_upper: ec50_linear_ci_upper,
            mean_formatted: format_4_sig_figs(ec50_linear_mean),
            std_formatted: format_4_sig_figs(ec50_linear_std),
            ci_lower_formatted: format_4_sig_figs(ec50_linear_ci_lower),
            ci_upper_formatted: format_4_sig_figs(ec50_linear_ci_upper),
        },
        hillslope: ParamDisplay {
            mean: summary.hillslope.mean,
            std: summary.hillslope.std,
            ci_lower: summary.hillslope.ci_lower,
            ci_upper: summary.hillslope.ci_upper,
            mean_formatted: format_4_sig_figs(summary.hillslope.mean),
            std_formatted: format_4_sig_figs(summary.hillslope.std),
            ci_lower_formatted: format_4_sig_figs(summary.hillslope.ci_lower),
            ci_upper_formatted: format_4_sig_figs(summary.hillslope.ci_upper),
        },
        acceptance_rate: acceptance_rate_percent,
        acceptance_rate_formatted: format_4_sig_figs(acceptance_rate_percent),
        diagnostics: format!(
            "Effective sample size: ~{:.0}",
            summary.n_samples as f64 * summary.acceptance_rate
        ),
    };

    Ok((plot_json, fit_display))
}

async fn generate_data_plot_json(
    data: &[DoseResponse],
) -> Result<String, Box<dyn std::error::Error>> {
    let data_points: Vec<(f64, f64)> = data
        .iter()
        .map(|d| (d.concentration.log10(), d.response))
        .collect();

    if data_points.is_empty() {
        return Ok(String::new());
    }

    let x_values: Vec<f64> = data_points.iter().map(|(x, _)| *x).collect();
    let y_values: Vec<f64> = data_points.iter().map(|(_, y)| *y).collect();

    let plot_data = json!({
        "data": [{
            "x": x_values,
            "y": y_values,
            "mode": "markers",
            "type": "scatter",
            "name": "Data Points",
            "marker": {
                "color": "blue",
                "size": 8
            }
        }],
        "layout": {
            "xaxis": {
                "title": "log10(Concentration)",
                "showgrid": true
            },
            "yaxis": {
                "title": "Response",
                "showgrid": true
            },
            "showlegend": true,
            "hovermode": "closest"
        },
        "config": {
            "displayModeBar": true,
            "responsive": true
        }
    });

    Ok(plot_data.to_string())
}

async fn generate_fitted_plot_json(
    data: &[DoseResponse],
    results: &fit_rs::MCMCResult,
    summary: &fit_rs::ParameterSummary,
) -> Result<String, Box<dyn std::error::Error>> {
    let data_points: Vec<(f64, f64)> = data
        .iter()
        .map(|d| (d.concentration.log10(), d.response))
        .collect();

    let x_min = data_points
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min)
        - 1.0;
    let x_max = data_points
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max)
        + 1.0;

    // Generate fitted curve using mean parameters
    let params_mean = LL4Parameters {
        emin: summary.emin.mean,
        emax: summary.emax.mean,
        ec50: summary.ec50.mean,
        hillslope: summary.hillslope.mean,
    };

    let x_points: Vec<f64> = (0..100)
        .map(|i| x_min + (x_max - x_min) * i as f64 / 99.0)
        .collect();

    let curve_points: Vec<f64> = x_points
        .iter()
        .map(|&log_conc| {
            let conc = 10.0_f64.powf(log_conc);
            ll4_model(conc, &params_mean)
        })
        .collect();

    // Generate confidence bands using MCMC samples
    let sample_step = (results.samples.len() / 100).max(1);
    let mut curve_predictions: Vec<Vec<f64>> = vec![Vec::new(); x_points.len()];

    for (i, sample) in results.samples.iter().step_by(sample_step).enumerate() {
        if i >= 50 {
            break;
        } // Limit to 50 samples for performance

        for (j, &log_conc) in x_points.iter().enumerate() {
            let conc = 10.0_f64.powf(log_conc);
            let response = ll4_model(conc, sample);
            curve_predictions[j].push(response);
        }
    }

    // Calculate percentiles for confidence bands
    let mut upper_bounds = Vec::new();
    let mut lower_bounds = Vec::new();

    for predictions in curve_predictions.iter() {
        if !predictions.is_empty() {
            let mut sorted_predictions = predictions.clone();
            sorted_predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = (sorted_predictions.len() as f64 * 0.025) as usize;
            let upper_idx = (sorted_predictions.len() as f64 * 0.975) as usize;

            upper_bounds.push(sorted_predictions[upper_idx.min(sorted_predictions.len() - 1)]);
            lower_bounds.push(sorted_predictions[lower_idx]);
        }
    }

    let data_x: Vec<f64> = data_points.iter().map(|(x, _)| *x).collect();
    let data_y: Vec<f64> = data_points.iter().map(|(_, y)| *y).collect();

    let y_min = data_points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let y_max = data_points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);

    let plot_data = json!({
        "data": [
            {
                "x": data_x,
                "y": data_y,
                "mode": "markers",
                "type": "scatter",
                "name": "Data Points",
                "marker": {
                    "color": "blue",
                    "size": 8
                }
            },
            {
                "x": x_points,
                "y": curve_points,
                "mode": "lines",
                "type": "scatter",
                "name": "Best Fit",
                "line": {
                    "color": "red",
                    "width": 3
                }
            },
            {
                "x": x_points.clone(),
                "y": upper_bounds,
                "mode": "lines",
                "type": "scatter",
                "name": "95% CI Upper",
                "line": {
                    "color": "rgba(255,0,0,0.3)",
                    "width": 1
                },
                "showlegend": false
            },
            {
                "x": x_points.clone(),
                "y": lower_bounds,
                "mode": "lines",
                "type": "scatter",
                "name": "95% Confidence",
                "line": {
                    "color": "rgba(255,0,0,0.3)",
                    "width": 1
                },
                "fill": "tonexty",
                "fillcolor": "rgba(255,0,0,0.1)"
            },
            {
                "x": [summary.ec50.mean, summary.ec50.mean],
                "y": [y_min - 0.1, y_max + 0.1],
                "mode": "lines",
                "type": "scatter",
                "name": &format!("EC50: {:.2}", summary.ec50.mean),
                "line": {
                    "color": "green",
                    "width": 2,
                    "dash": "dash"
                }
            }
        ],
        "layout": {
            "xaxis": {
                "title": "log10(Concentration)",
                "showgrid": true
            },
            "yaxis": {
                "title": "Response",
                "showgrid": true
            },
            "showlegend": true,
            "hovermode": "closest"
        },
        "config": {
            "displayModeBar": true,
            "responsive": true
        }
    });

    Ok(plot_data.to_string())
}

fn ll4_model(concentration: f64, params: &LL4Parameters) -> f64 {
    params.emin
        + (params.emax - params.emin)
            / (1.0 + 10.0_f64.powf(params.hillslope * (params.ec50 - concentration.log10())))
}

fn format_4_sig_figs(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }

    let abs_value = value.abs();
    let log10_abs = abs_value.log10();
    let magnitude = log10_abs.floor() as i32;
    let normalized = abs_value / 10.0_f64.powi(magnitude);

    // Round to 4 significant figures
    let rounded = (normalized * 1000.0).round() / 1000.0;
    let result = rounded * 10.0_f64.powi(magnitude);

    // Choose appropriate format based on magnitude
    if magnitude >= -2 && magnitude <= 4 {
        // Use decimal notation for reasonable ranges
        let decimal_places = (3 - magnitude).max(0) as usize;
        if value < 0.0 {
            format!("-{:.prec$}", result.abs(), prec = decimal_places)
        } else {
            format!("{:.prec$}", result, prec = decimal_places)
        }
    } else {
        // Use scientific notation for very large or very small numbers
        if value < 0.0 {
            format!("-{:.3e}", result.abs())
        } else {
            format!("{:.3e}", result)
        }
    }
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
            results.emin.mean,
            results.emin.std,
            results.emin.ci_lower,
            results.emin.ci_upper,
            results.emax.mean,
            results.emax.std,
            results.emax.ci_lower,
            results.emax.ci_upper,
            results.ec50_log.mean,
            results.ec50_log.std,
            results.ec50_log.ci_lower,
            results.ec50_log.ci_upper,
            results.ec50_linear.mean,
            results.ec50_linear.std,
            results.ec50_linear.ci_lower,
            results.ec50_linear.ci_upper,
            results.hillslope.mean,
            results.hillslope.std,
            results.hillslope.ci_lower,
            results.hillslope.ci_upper,
            results.acceptance_rate
        );

        let filename = format!(
            "ec50_results_{}.csv",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        );

        Response::builder()
            .header(header::CONTENT_TYPE, "text/csv")
            .header(
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{}\"", filename),
            )
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
        .route("/upload-files", post(upload_files))
        .route("/upload-data", post(upload_data))
        .route("/upload-config", post(upload_config))
        .route("/parameter-form", get(parameter_form))
        .route("/update-parameters", post(update_parameters))
        .route("/download-csv", get(download_csv))
        .nest_service("/static", ServeDir::new("static"))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
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

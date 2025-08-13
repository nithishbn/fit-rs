use crate::{BayesianEC50Fitter, MCMCResult, ParameterSummary};
use anyhow::Error;
use plotters::prelude::*;

pub struct EC50Visualizer<'a> {
    fitter: &'a BayesianEC50Fitter,
}

impl<'a> EC50Visualizer<'a> {
    pub fn new(fitter: &'a BayesianEC50Fitter) -> Self {
        Self { fitter }
    }

    /// Generate dose-response curve with uncertainty bands
    pub fn plot_dose_response_curve(
        &self,
        result: &MCMCResult,
        output_path: &str,
        title: Option<&str>,
        bounds: Option<(f64, f64, f64, f64)>,
    ) -> Result<(), anyhow::Error> {
        // Debug information
        self.print_debug_info(result);

        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Calculate concentration range for smooth curve
        let conc_range = self.get_concentration_range();
        let log_conc_min = conc_range.0.log10() - 1.0;
        let log_conc_max = conc_range.1.log10() + 1.0;

        // Generate prediction points
        let n_points = 1000;
        let mut pred_concentrations = Vec::new();
        let mut pred_responses = Vec::new();
        let mut pred_ci_lower = Vec::new();
        let mut pred_ci_upper = Vec::new();

        for i in 0..n_points {
            let log_conc =
                log_conc_min + (log_conc_max - log_conc_min) * i as f64 / (n_points - 1) as f64;
            let conc = 10.0_f64.powf(log_conc);
            pred_concentrations.push(conc);

            // Calculate predictions for all MCMC samples
            let mut predictions = Vec::new();
            for params in &result.samples {
                let pred = self.fitter.ll4_model(conc, params);
                predictions.push(pred);
            }

            // Calculate statistics
            if predictions.is_empty() {
                eprintln!(
                    "Warning: No predictions generated for concentration {:.2e}",
                    conc
                );
                continue;
            }

            predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;

            let n = predictions.len();
            let ci_lower_idx = ((0.025 * n as f64) as usize).min(n - 1);
            let ci_upper_idx = ((0.975 * n as f64) as usize).min(n - 1);

            let ci_lower = predictions[ci_lower_idx];
            let ci_upper = predictions[ci_upper_idx];

            pred_responses.push(mean);
            pred_ci_lower.push(ci_lower);
            pred_ci_upper.push(ci_upper);
        }
        let (x_min, x_max, y_min, y_max) =
            if let Some((custom_xmin, custom_xmax, custom_ymin, custom_ymax)) = bounds {
                (custom_xmin, custom_xmax, custom_ymin, custom_ymax)
            } else {
                let y_min = self
                    .fitter
                    .data
                    .iter()
                    .map(|d| d.response)
                    .chain(pred_ci_lower.iter().cloned())
                    .fold(f64::INFINITY, f64::min)
                    - 10.0;
                let y_max = self
                    .fitter
                    .data
                    .iter()
                    .map(|d| d.response)
                    .chain(pred_ci_upper.iter().cloned())
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 10.0;
                let x_min = 10.0_f64.powf(log_conc_min);
                let x_max = 10.0_f64.powf(log_conc_max);
                (x_min, x_max, y_min, y_max)
            };
        // Find y-axis range

        let mut chart = ChartBuilder::on(&root)
            .caption(
                title.unwrap_or("EC50 Dose-Response Curve"),
                ("sans-serif", 30),
            )
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d((x_min..x_max).log_scale(), y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc("Concentration (M)")
            .y_desc("Response")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        // Plot confidence interval as filled area
        let ci_points: Vec<(f64, f64)> = pred_concentrations
            .iter()
            .zip(pred_ci_lower.iter())
            .map(|(&x, &y)| (x, y))
            .chain(
                pred_concentrations
                    .iter()
                    .rev()
                    .zip(pred_ci_upper.iter().rev())
                    .map(|(&x, &y)| (x, y)),
            )
            .collect();

        chart
            .draw_series(std::iter::once(Polygon::new(ci_points, &BLUE.mix(0.3))))?
            .label("95% Credible Interval")
            .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 10)], BLUE.mix(0.3)));

        // Plot mean prediction curve
        chart
            .draw_series(LineSeries::new(
                pred_concentrations
                    .iter()
                    .zip(pred_responses.iter())
                    .map(|(&x, &y)| (x, y)),
                BLACK.stroke_width(3),
            ))?
            .label("Mean Prediction")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLACK));

        // Plot observed data points
        chart
            .draw_series(
                self.fitter.data.iter().map(|point| {
                    Circle::new((point.concentration, point.response), 5, RED.filled())
                }),
            )?
            .label("Observed Data")
            .legend(|(x, y)| Circle::new((x + 5, y), 5, RED.filled()));

        // Add EC50 vertical line
        if let Some(params) = result.samples.first() {
            let ec50_conc = 10.0_f64.powf(params.ec50);
            let ec50_line_points = vec![(ec50_conc, y_min), (ec50_conc, y_max)];
            chart
                .draw_series(LineSeries::new(
                    ec50_line_points,
                    MAGENTA.stroke_width(2)
                ))?
                .label("EC50")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &MAGENTA));
        }

        chart.configure_series_labels().draw()?;
        root.present()?;

        println!("Dose-response curve saved to: {}", output_path);
        Ok(())
    }

    /// Plot MCMC trace plots for diagnostics
    pub fn plot_mcmc_traces(
        &self,
        result: &MCMCResult,
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        // Split into 4 subplots (2x2 grid)
        let areas = root.split_evenly((2, 2));
        let emin_area = &areas[0];
        let emax_area = &areas[1];
        let ec50_area = &areas[2];
        let hill_area = &areas[3];

        // Extract parameter traces
        let iterations: Vec<usize> = (0..result.samples.len()).collect();
        let emin_trace: Vec<f64> = result.samples.iter().map(|p| p.emin).collect();
        let emax_trace: Vec<f64> = result.samples.iter().map(|p| p.emax).collect();
        let ec50_trace: Vec<f64> = result.samples.iter().map(|p| p.ec50).collect();
        let hill_trace: Vec<f64> = result.samples.iter().map(|p| p.hillslope).collect();

        // Plot Emin trace
        self.plot_single_trace(emin_area, &iterations, &emin_trace, "Emin", &RED)?;

        // Plot Emax trace
        self.plot_single_trace(emax_area, &iterations, &emax_trace, "Emax", &BLUE)?;

        // Plot EC50 trace
        self.plot_single_trace(ec50_area, &iterations, &ec50_trace, "EC50 (log10)", &GREEN)?;

        // Plot Hill slope trace
        self.plot_single_trace(hill_area, &iterations, &hill_trace, "Hill Slope", &MAGENTA)?;

        root.present()?;
        println!("MCMC traces saved to: {}", output_path);
        Ok(())
    }

    fn plot_single_trace<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, plotters::coord::Shift>,
        iterations: &[usize],
        values: &[f64],
        title: &str,
        color: &RGBColor,
    ) -> Result<(), anyhow::Error>
    where
        DB::ErrorType: 'static,
    {
        let y_min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range = y_max - y_min;
        let y_margin = y_range * 0.1;

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0usize..iterations.len(),
                (y_min - y_margin)..(y_max + y_margin),
            )?;

        chart
            .configure_mesh()
            .x_desc("Iteration")
            .y_desc("Value")
            .axis_desc_style(("sans-serif", 12))
            .draw()?;

        chart.draw_series(LineSeries::new(
            iterations.iter().zip(values.iter()).map(|(&x, &y)| (x, y)),
            color,
        ))?;

        Ok(())
    }

    /// Plot parameter posterior distributions
    pub fn plot_posterior_distributions(
        &self,
        result: &MCMCResult,
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        // Split into 4 subplots
        let areas = root.split_evenly((2, 2));
        let emin_area = &areas[0];
        let emax_area = &areas[1];
        let ec50_area = &areas[2];
        let hill_area = &areas[3];

        // Extract parameter samples
        let emin_samples: Vec<f64> = result.samples.iter().map(|p| p.emin).collect();
        let emax_samples: Vec<f64> = result.samples.iter().map(|p| p.emax).collect();
        let ec50_samples: Vec<f64> = result.samples.iter().map(|p| p.ec50).collect();
        let hill_samples: Vec<f64> = result.samples.iter().map(|p| p.hillslope).collect();

        // Plot histograms
        self.plot_histogram(emin_area, &emin_samples, "Emin Posterior", &RED)?;
        self.plot_histogram(emax_area, &emax_samples, "Emax Posterior", &BLUE)?;
        self.plot_histogram(ec50_area, &ec50_samples, "EC50 Posterior", &GREEN)?;
        self.plot_histogram(hill_area, &hill_samples, "Hill Slope Posterior", &MAGENTA)?;

        root.present()?;
        println!("Posterior distributions saved to: {}", output_path);
        Ok(())
    }

    fn plot_histogram<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, plotters::coord::Shift>,
        samples: &[f64],
        title: &str,
        color: &RGBColor,
    ) -> Result<(), anyhow::Error>
    where
        DB::ErrorType: 'static,
    {
        // Calculate histogram
        let mut sorted_samples = samples.to_vec();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_val = sorted_samples[0];
        let max_val = sorted_samples[sorted_samples.len() - 1];
        let range = max_val - min_val;

        let n_bins = 30;
        let bin_width = range / n_bins as f64;
        let mut histogram = vec![0; n_bins];

        for &sample in samples {
            let bin_idx = ((sample - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            histogram[bin_idx] += 1;
        }

        let max_count = *histogram.iter().max().unwrap();

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(min_val..max_val, 0..max_count)?;

        chart
            .configure_mesh()
            .x_desc("Value")
            .y_desc("Count")
            .axis_desc_style(("sans-serif", 12))
            .draw()?;

        chart.draw_series(histogram.iter().enumerate().map(|(i, &count)| {
            let x1 = min_val + i as f64 * bin_width;
            let x2 = x1 + bin_width;
            Rectangle::new([(x1, 0), (x2, count)], color.mix(0.7).filled())
        }))?;

        Ok(())
    }

    /// Plot log-likelihood trace for convergence diagnostics
    pub fn plot_log_likelihood_trace(
        &self,
        result: &MCMCResult,
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        let root = BitMapBackend::new(output_path, (800, 400)).into_drawing_area();
        root.fill(&WHITE)?;

        let iterations: Vec<usize> = (0..result.log_likelihood.len()).collect();

        let y_min = result
            .log_likelihood
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = result
            .log_likelihood
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range = y_max - y_min;
        let y_margin = y_range * 0.1;

        let mut chart = ChartBuilder::on(&root)
            .caption("Log-Likelihood Trace", ("sans-serif", 25))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                0usize..iterations.len(),
                (y_min - y_margin)..(y_max + y_margin),
            )?;

        chart
            .configure_mesh()
            .x_desc("Iteration")
            .y_desc("Log-Likelihood")
            .axis_desc_style(("sans-serif", 15))
            .draw()?;

        chart.draw_series(LineSeries::new(
            iterations
                .iter()
                .zip(result.log_likelihood.iter())
                .map(|(&x, &y)| (x, y)),
            BLACK.stroke_width(1),
        ))?;

        root.present()?;
        println!("Log-likelihood trace saved to: {}", output_path);
        Ok(())
    }

    /// Generate all diagnostic plots at once
    pub fn generate_all_plots(
        &self,
        result: &MCMCResult,
        summary: &ParameterSummary,
        output_dir: &str,
        bounds: Option<(f64, f64, f64, f64)>,
    ) -> Result<(), anyhow::Error> {
        std::fs::create_dir_all(output_dir)?;

        // Main dose-response curve
        let title = format!(
            "EC50 Fit (Acceptance Rate: {:.1}%, N samples: {})",
            summary.acceptance_rate * 100.0,
            summary.n_samples
        );
        self.plot_dose_response_curve(
            result,
            &format!("{}/dose_response_curve.png", output_dir),
            Some(&title),
            bounds,
        )?;

        // MCMC diagnostics
        self.plot_mcmc_traces(result, &format!("{}/mcmc_traces.png", output_dir))?;
        self.plot_posterior_distributions(
            result,
            &format!("{}/posterior_distributions.png", output_dir),
        )?;
        self.plot_log_likelihood_trace(
            result,
            &format!("{}/log_likelihood_trace.png", output_dir),
        )?;

        println!("All plots generated in directory: {}", output_dir);
        Ok(())
    }

    fn get_concentration_range(&self) -> (f64, f64) {
        let concentrations: Vec<f64> = self.fitter.data.iter().map(|d| d.concentration).collect();
        let min_conc = concentrations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_conc = concentrations
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min_conc, max_conc)
    }

    fn print_debug_info(&self, result: &MCMCResult) {
        let (min_conc, max_conc) = self.get_concentration_range();
        println!(
            "Debug - Data concentration range: {:.2e} to {:.2e} M",
            min_conc, max_conc
        );

        if let Some(sample) = result.samples.first() {
            println!(
                "Debug - Sample parameters: Emin={:.1}, Emax={:.1}, EC50={:.2}, Hill={:.2}",
                sample.emin, sample.emax, sample.ec50, sample.hillslope
            );
        }

        // Test curve calculation at a few points
        if let Some(sample) = result.samples.first() {
            for &conc in &[min_conc, max_conc, 10.0_f64.powf(sample.ec50)] {
                let pred = self.fitter.ll4_model(conc, sample);
                println!(
                    "Debug - At {:.2e} M: predicted response = {:.1}",
                    conc, pred
                );
            }
        }
    }
}

// Extend the main fitter with convenience methods
impl BayesianEC50Fitter {
    /// Fit and generate plots in one step
    pub fn fit_and_plot(
        &self,
        n_samples: usize,
        burnin: usize,
        output_dir: &str,
        bounds: Option<(f64, f64, f64, f64)>,
    ) -> Result<(MCMCResult, ParameterSummary), Error> {
        println!("Running MCMC sampling...");
        let result = self.fit(n_samples, burnin);
        let summary = self.summarize_results(&result);

        println!("Generating diagnostic plots...");
        let visualizer = EC50Visualizer::new(self);
        visualizer.generate_all_plots(&result, &summary, output_dir, bounds)?;

        // Print summary to console
        self.print_summary(&summary);
        self.save_coefficients_csv(&summary, &format!("{}/coefficients.csv", output_dir))?;
        self.save_coefficients_json(&summary, &format!("{}/coefficients.json", output_dir))?;
        Ok((result, summary))
    }

    fn print_summary(&self, summary: &ParameterSummary) {
        println!("\n=== EC50 Fitting Results ===");
        println!(
            "Samples: {}, Acceptance Rate: {:.1}%",
            summary.n_samples,
            summary.acceptance_rate * 100.0
        );
        println!();
        println!("Parameter Estimates (Mean ± SD [95% CI]):");
        println!(
            "  Emin:      {:.3} ± {:.3} [{:.3}, {:.3}]",
            summary.emin.mean, summary.emin.std, summary.emin.ci_lower, summary.emin.ci_upper
        );
        println!(
            "  Emax:      {:.3} ± {:.3} [{:.3}, {:.3}]",
            summary.emax.mean, summary.emax.std, summary.emax.ci_lower, summary.emax.ci_upper
        );
        println!(
            "  EC50:      {:.3} ± {:.3} [{:.3}, {:.3}] (log10 M)",
            summary.ec50.mean, summary.ec50.std, summary.ec50.ci_lower, summary.ec50.ci_upper
        );
        println!(
            "  EC50:      {:.2e} M ({:.2e}, {:.2e})",
            10.0_f64.powf(summary.ec50.mean),
            10.0_f64.powf(summary.ec50.ci_lower),
            10.0_f64.powf(summary.ec50.ci_upper)
        );
        println!(
            "  Hill:      {:.3} ± {:.3} [{:.3}, {:.3}]",
            summary.hillslope.mean,
            summary.hillslope.std,
            summary.hillslope.ci_lower,
            summary.hillslope.ci_upper
        );
    }
    fn save_coefficients_csv(
        &self,
        summary: &ParameterSummary,
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(output_path)?;

        // Write header
        writeln!(
            file,
            "parameter,mean,median,std,ci_lower,ci_upper,description"
        )?;

        // Write parameter estimates
        writeln!(
            file,
            "emin,{:.6},{:.6},{:.6},{:.6},{:.6},\"Lower asymptote (baseline response)\"",
            summary.emin.mean,
            summary.emin.median,
            summary.emin.std,
            summary.emin.ci_lower,
            summary.emin.ci_upper
        )?;

        writeln!(
            file,
            "emax,{:.6},{:.6},{:.6},{:.6},{:.6},\"Upper asymptote (maximum response)\"",
            summary.emax.mean,
            summary.emax.median,
            summary.emax.std,
            summary.emax.ci_lower,
            summary.emax.ci_upper
        )?;

        writeln!(
            file,
            "ec50_log10,{:.6},{:.6},{:.6},{:.6},{:.6},\"EC50 on log10 scale\"",
            summary.ec50.mean,
            summary.ec50.median,
            summary.ec50.std,
            summary.ec50.ci_lower,
            summary.ec50.ci_upper
        )?;

        // Calculate linear EC50 values
        let ec50_linear_mean = 10.0_f64.powf(summary.ec50.mean);
        let ec50_linear_median = 10.0_f64.powf(summary.ec50.median);
        let ec50_linear_ci_lower = 10.0_f64.powf(summary.ec50.ci_lower);
        let ec50_linear_ci_upper = 10.0_f64.powf(summary.ec50.ci_upper);

        writeln!(file, "ec50_linear,{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},\"EC50 on linear scale (concentration units)\"",
                 ec50_linear_mean, ec50_linear_median,
                 ec50_linear_ci_upper - ec50_linear_ci_lower, // Approximate std in linear space
                 ec50_linear_ci_lower, ec50_linear_ci_upper)?;

        writeln!(
            file,
            "hillslope,{:.6},{:.6},{:.6},{:.6},{:.6},\"Hill slope (steepness of curve)\"",
            summary.hillslope.mean,
            summary.hillslope.median,
            summary.hillslope.std,
            summary.hillslope.ci_lower,
            summary.hillslope.ci_upper
        )?;

        // Add model diagnostics
        writeln!(
            file,
            "acceptance_rate,{:.6},{:.6},{:.6},{:.6},{:.6},\"MCMC acceptance rate\"",
            summary.acceptance_rate,
            summary.acceptance_rate,
            0.0,
            summary.acceptance_rate,
            summary.acceptance_rate
        )?;

        let eff_sample_size = summary.n_samples as f64 * summary.acceptance_rate;
        writeln!(file, "effective_sample_size,{:.0},{:.0},{:.0},{:.0},{:.0},\"Effective number of independent samples\"",
                 eff_sample_size, eff_sample_size, 0.0, eff_sample_size, eff_sample_size)?;

        println!("Coefficients saved to: {}", output_path);
        Ok(())
    }
    fn save_coefficients_json(
        &self,
        summary: &ParameterSummary,
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        use serde_json::json;
        use std::fs::File;
        use std::io::Write;

        let coefficients = json!({
            "model_info": {
                "formula": "response ~ (emin + (emax - emin) / (1 + 10**(hillslope * (ec50 - log_concentration))))",
                "n_samples": summary.n_samples,
                "acceptance_rate": summary.acceptance_rate,
                "effective_sample_size": summary.n_samples as f64 * summary.acceptance_rate
            },
            "parameters": {
                "emin": {
                    "mean": summary.emin.mean,
                    "median": summary.emin.median,
                    "std": summary.emin.std,
                    "ci_lower": summary.emin.ci_lower,
                    "ci_upper": summary.emin.ci_upper,
                    "description": "Lower asymptote (baseline response)"
                },
                "emax": {
                    "mean": summary.emax.mean,
                    "median": summary.emax.median,
                    "std": summary.emax.std,
                    "ci_lower": summary.emax.ci_lower,
                    "ci_upper": summary.emax.ci_upper,
                    "description": "Upper asymptote (maximum response)"
                },
                "ec50_log10": {
                    "mean": summary.ec50.mean,
                    "median": summary.ec50.median,
                    "std": summary.ec50.std,
                    "ci_lower": summary.ec50.ci_lower,
                    "ci_upper": summary.ec50.ci_upper,
                    "description": "EC50 on log10 scale"
                },
                "ec50_linear": {
                    "mean": 10.0_f64.powf(summary.ec50.mean),
                    "median": 10.0_f64.powf(summary.ec50.median),
                    "ci_lower": 10.0_f64.powf(summary.ec50.ci_lower),
                    "ci_upper": 10.0_f64.powf(summary.ec50.ci_upper),
                    "description": "EC50 on linear scale (concentration units)"
                },
                "hillslope": {
                    "mean": summary.hillslope.mean,
                    "median": summary.hillslope.median,
                    "std": summary.hillslope.std,
                    "ci_lower": summary.hillslope.ci_lower,
                    "ci_upper": summary.hillslope.ci_upper,
                    "description": "Hill slope (steepness of curve)"
                }
            },
            "diagnostics": {
                "convergence_warnings": self.get_convergence_warnings(summary),
                "model_quality": self.assess_model_quality(summary)
            }
        });

        let mut file = File::create(output_path)?;
        let json_string = serde_json::to_string_pretty(&coefficients)?;
        file.write_all(json_string.as_bytes())?;

        println!("Detailed coefficients saved to: {}", output_path);
        Ok(())
    }

    /// Get convergence warnings for the model
    fn get_convergence_warnings(&self, summary: &ParameterSummary) -> Vec<String> {
        let mut warnings = Vec::new();

        if summary.acceptance_rate < 0.2 {
            warnings.push(format!(
                "Low acceptance rate ({:.1}%) - consider adjusting priors or proposal distribution",
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

        if summary.hillslope.mean < 0.3 || summary.hillslope.mean > 3.0 {
            warnings.push(format!(
                "Unusual Hill slope estimate ({:.2}) - check model fit",
                summary.hillslope.mean
            ));
        }

        warnings
    }

    /// Assess overall model quality
    fn assess_model_quality(&self, summary: &ParameterSummary) -> String {
        let warnings = self.get_convergence_warnings(summary);

        if warnings.is_empty() {
            "Good - No major issues detected".to_string()
        } else if warnings.len() <= 2 {
            "Fair - Minor issues detected".to_string()
        } else {
            "Poor - Multiple issues detected, consider model adjustments".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DoseResponse, LL4Parameters};
    use rand::Rng;

    #[test]
    fn test_visualization() {
        // Generate test data
        let true_params = LL4Parameters {
            emin: 10.0,
            emax: 90.0,
            ec50: -6.0,
            hillslope: 1.2,
        };

        let concentrations: Vec<f64> = vec![1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5];
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();

        for &conc in &concentrations {
            let log_conc = conc.log10();
            let response = true_params.emin
                + (true_params.emax - true_params.emin)
                    / (1.0 + 10.0_f64.powf((true_params.ec50 - log_conc) * true_params.hillslope));
            // Add noise
            let response = response + rng.gen_range(-3.0..3.0);

            data.push(DoseResponse {
                concentration: conc,
                response,
            });
        }

        // Fit and plot
        let fitter = BayesianEC50Fitter::new(data);
        let result = fitter.fit_and_plot(500, 200, "test_output", None);

        assert!(result.is_ok());
    }
}

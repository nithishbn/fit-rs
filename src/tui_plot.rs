use crate::{BayesianEC50Fitter, DoseResponse, LL4Parameters, MCMCResult, ParameterSummary};
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    symbols::Marker,
    widgets::{
        Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap, Clear,
    },
};
use std::{
    io,
    time::{Duration, Instant},
};

pub struct TuiPlotter {
    fitter: BayesianEC50Fitter,
    data: Vec<DoseResponse>,
    results: Option<MCMCResult>,
    summary: Option<ParameterSummary>,
}

#[derive(Clone, Copy, PartialEq)]
enum PlotMode {
    DoseResponse,
    MCMCTraces,
    Posteriors,
    Diagnostics,
}

struct AppState {
    mode: PlotMode,
    selected_param: usize, // For MCMC traces and posteriors
    show_help: bool,
    zoom_level: f64,
    offset_x: f64,
    offset_y: f64,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            mode: PlotMode::DoseResponse,
            selected_param: 0,
            show_help: false,
            zoom_level: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

impl TuiPlotter {
    pub fn new(fitter: BayesianEC50Fitter, data: Vec<DoseResponse>) -> Self {
        Self {
            fitter,
            data,
            results: None,
            summary: None,
        }
    }

    pub fn with_results(mut self, results: MCMCResult, summary: ParameterSummary) -> Self {
        self.results = Some(results);
        self.summary = Some(summary);
        self
    }

    pub fn run_interactive_plot(&self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut app_state = AppState::default();
        let mut last_tick = Instant::now();
        let tick_rate = Duration::from_millis(250);

        let result = loop {
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                        KeyCode::Char('h') | KeyCode::F(1) => {
                            app_state.show_help = !app_state.show_help;
                        }
                        KeyCode::Char('1') => app_state.mode = PlotMode::DoseResponse,
                        KeyCode::Char('2') => app_state.mode = PlotMode::MCMCTraces,
                        KeyCode::Char('3') => app_state.mode = PlotMode::Posteriors,
                        KeyCode::Char('4') => app_state.mode = PlotMode::Diagnostics,
                        KeyCode::Up => {
                            match app_state.mode {
                                PlotMode::MCMCTraces | PlotMode::Posteriors => {
                                    if app_state.selected_param > 0 {
                                        app_state.selected_param -= 1;
                                    }
                                }
                                _ => {
                                    app_state.offset_y += 0.1;
                                }
                            }
                        }
                        KeyCode::Down => {
                            match app_state.mode {
                                PlotMode::MCMCTraces | PlotMode::Posteriors => {
                                    if app_state.selected_param < 3 {
                                        app_state.selected_param += 1;
                                    }
                                }
                                _ => {
                                    app_state.offset_y -= 0.1;
                                }
                            }
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            app_state.zoom_level *= 1.2;
                        }
                        KeyCode::Char('-') => {
                            app_state.zoom_level /= 1.2;
                        }
                        KeyCode::Left => {
                            app_state.offset_x -= 0.1;
                        }
                        KeyCode::Right => {
                            app_state.offset_x += 0.1;
                        }
                        KeyCode::Char('r') => {
                            // Reset zoom and offset
                            app_state.zoom_level = 1.0;
                            app_state.offset_x = 0.0;
                            app_state.offset_y = 0.0;
                        }
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }

            terminal.draw(|f| self.draw_ui(f, &app_state))?;
        };

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    fn draw_ui(&self, frame: &mut Frame, app_state: &AppState) {
        let size = frame.size();

        if app_state.show_help {
            self.draw_help(frame, size);
            return;
        }

        // Create layout
        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Main content
                Constraint::Length(2), // Status bar
            ])
            .split(size);

        // Header
        self.draw_header(frame, main_layout[0], app_state);

        // Main content based on mode
        match app_state.mode {
            PlotMode::DoseResponse => self.draw_dose_response_plot(frame, main_layout[1], app_state),
            PlotMode::MCMCTraces => self.draw_mcmc_traces(frame, main_layout[1], app_state),
            PlotMode::Posteriors => self.draw_posteriors(frame, main_layout[1], app_state),
            PlotMode::Diagnostics => self.draw_diagnostics(frame, main_layout[1], app_state),
        }

        // Status bar
        self.draw_status_bar(frame, main_layout[2], app_state);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        let title = match app_state.mode {
            PlotMode::DoseResponse => "EC50 Dose-Response Curve",
            PlotMode::MCMCTraces => "MCMC Parameter Traces",
            PlotMode::Posteriors => "Posterior Distributions",
            PlotMode::Diagnostics => "MCMC Diagnostics",
        };

        let header = Paragraph::new(title)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);

        frame.render_widget(header, area);
    }

    fn draw_status_bar(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        let mode_indicator = match app_state.mode {
            PlotMode::DoseResponse => "[1] Dose-Response",
            PlotMode::MCMCTraces => "[2] MCMC Traces",
            PlotMode::Posteriors => "[3] Posteriors",
            PlotMode::Diagnostics => "[4] Diagnostics",
        };

        let status_text = format!(
            "{} | Zoom: {:.1}x | [h] Help | [q] Quit",
            mode_indicator, app_state.zoom_level
        );

        let status_bar = Paragraph::new(status_text)
            .style(Style::default().fg(Color::White).bg(Color::Blue))
            .alignment(Alignment::Center);

        frame.render_widget(status_bar, area);
    }

    fn draw_help(&self, frame: &mut Frame, area: Rect) {
        let help_text = vec![
            "EC50 Curve Fitting - Interactive Terminal Plots".into(),
            "".into(),
            "Navigation:".into(),
            "  [1-4]    Switch between plot modes".into(),
            "  [↑/↓]    Select parameter (traces/posteriors) or pan up/down".into(),
            "  [←/→]    Pan left/right".into(),
            "  [+/-]    Zoom in/out".into(),
            "  [r]      Reset zoom and pan".into(),
            "".into(),
            "Plot Modes:".into(),
            "  [1] Dose-Response Curve - Fitted curve with 95% confidence intervals".into(),
            "  [2] MCMC Traces - Parameter convergence over iterations".into(),
            "  [3] Posteriors - Parameter uncertainty distributions".into(),
            "  [4] Diagnostics - Acceptance rate and R-hat statistics".into(),
            "".into(),
            "Controls:".into(),
            "  [h/F1]   Toggle this help".into(),
            "  [q/Esc]  Quit".into(),
            "".into(),
            "Press [h] or [F1] to close help".into(),
        ];

        let help_paragraph = Paragraph::new(help_text)
            .block(
                Block::default()
                    .title("Help")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
            )
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: true })
            .alignment(Alignment::Left);

        // Center the help in the screen
        let help_area = centered_rect(80, 90, area);
        frame.render_widget(Clear, help_area);
        frame.render_widget(help_paragraph, help_area);
    }

    fn draw_dose_response_plot(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        // Prepare data points
        let data_points: Vec<(f64, f64)> = self.data
            .iter()
            .map(|d| (d.concentration.log10(), d.response))
            .collect();

        // Generate fitted curve points and confidence bands
        let (fitted_points, confidence_bands) = if let (Some(summary), Some(results)) = (&self.summary, &self.results) {
            let params = LL4Parameters {
                emin: summary.emin.mean,
                emax: summary.emax.mean,
                ec50: summary.ec50.mean,
                hillslope: summary.hillslope.mean,
            };

            let log_conc_min = data_points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min) - 2.0;
            let log_conc_max = data_points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max) + 2.0;
            
            let x_points: Vec<f64> = (0..100)
                .map(|i| log_conc_min + (log_conc_max - log_conc_min) * i as f64 / 99.0)
                .collect();

            // Calculate fitted curve
            let fitted: Vec<(f64, f64)> = x_points
                .iter()
                .map(|&log_conc| {
                    let conc = 10.0_f64.powf(log_conc);
                    let response = self.fitter.ll4_model(conc, &params);
                    (log_conc, response)
                })
                .collect();

            // Calculate confidence bands using MCMC samples
            let mut confidence_upper = Vec::new();
            let mut confidence_lower = Vec::new();

            for &log_conc in &x_points {
                let conc = 10.0_f64.powf(log_conc);
                let mut responses: Vec<f64> = results.samples
                    .iter()
                    .map(|sample_params| self.fitter.ll4_model(conc, sample_params))
                    .collect();
                
                responses.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                // 95% confidence interval
                let lower_idx = (0.025 * responses.len() as f64) as usize;
                let upper_idx = (0.975 * responses.len() as f64) as usize;
                
                let lower_bound = responses[lower_idx.min(responses.len() - 1)];
                let upper_bound = responses[upper_idx.min(responses.len() - 1)];
                
                confidence_lower.push((log_conc, lower_bound));
                confidence_upper.push((log_conc, upper_bound));
            }

            (fitted, Some((confidence_lower, confidence_upper)))
        } else {
            (vec![], None)
        };

        // Calculate plot bounds with zoom and offset
        let x_min = data_points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min) - 2.0;
        let x_max = data_points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max) + 2.0;
        
        // Get y bounds from all curves including confidence bands
        let mut all_y_values = data_points.iter().map(|(_, y)| *y).collect::<Vec<_>>();
        if !fitted_points.is_empty() {
            all_y_values.extend(fitted_points.iter().map(|(_, y)| *y));
        }
        if let Some((ref lower, ref upper)) = confidence_bands {
            all_y_values.extend(lower.iter().map(|(_, y)| *y));
            all_y_values.extend(upper.iter().map(|(_, y)| *y));
        }
        
        let y_min_raw = all_y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max_raw = all_y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range_raw = y_max_raw - y_min_raw;
        let y_padding = y_range_raw * 0.1; // 10% padding
        let y_min = y_min_raw - y_padding;
        let y_max = y_max_raw + y_padding;

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let zoom_x_range = x_range / app_state.zoom_level;
        let zoom_y_range = y_range / app_state.zoom_level;

        let plot_x_min = x_min + x_range * 0.5 - zoom_x_range * 0.5 + app_state.offset_x;
        let plot_x_max = x_min + x_range * 0.5 + zoom_x_range * 0.5 + app_state.offset_x;
        let plot_y_min = y_min + y_range * 0.5 - zoom_y_range * 0.5 + app_state.offset_y;
        let plot_y_max = y_min + y_range * 0.5 + zoom_y_range * 0.5 + app_state.offset_y;

        let mut datasets = vec![];

        // Add confidence bands first (so they appear behind the curves)
        if let Some((ref lower, ref upper)) = confidence_bands {
            datasets.push(
                Dataset::default()
                    .name("95% CI Lower")
                    .marker(Marker::Dot)
                    .style(Style::default().fg(Color::DarkGray))
                    .graph_type(GraphType::Line)
                    .data(lower)
            );
            datasets.push(
                Dataset::default()
                    .name("95% CI Upper")
                    .marker(Marker::Dot)
                    .style(Style::default().fg(Color::DarkGray))
                    .graph_type(GraphType::Line)
                    .data(upper)
            );
        }

        // Add fitted curve
        if !fitted_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .name("Fitted Curve")
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::Green))
                    .graph_type(GraphType::Line)
                    .data(&fitted_points)
            );
        }

        // Add data points on top
        datasets.push(
            Dataset::default()
                .name("Data Points")
                .marker(Marker::Block)
                .style(Style::default().fg(Color::Red))
                .graph_type(GraphType::Scatter)
                .data(&data_points)
        );

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title("Dose-Response Curve with 95% CI"))
            .x_axis(
                Axis::default()
                    .title("Log10(Concentration)")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([plot_x_min, plot_x_max])
                    .labels(vec![
                        format!("{:.1}", plot_x_min).into(),
                        format!("{:.1}", (plot_x_min + plot_x_max) / 2.0).into(),
                        format!("{:.1}", plot_x_max).into(),
                    ])
            )
            .y_axis(
                Axis::default()
                    .title("Response")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([plot_y_min, plot_y_max])
                    .labels(vec![
                        format!("{:.1}", plot_y_min).into(),
                        format!("{:.1}", (plot_y_min + plot_y_max) / 2.0).into(),
                        format!("{:.1}", plot_y_max).into(),
                    ])
            );

        frame.render_widget(chart, area);
    }

    fn draw_mcmc_traces(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        if let Some(results) = &self.results {
            let param_names = ["Emin", "Emax", "EC50", "Hill Slope"];
            let param_name = param_names[app_state.selected_param];

            let trace_data: Vec<(f64, f64)> = results.samples
                .iter()
                .enumerate()
                .map(|(i, params)| {
                    let value = match app_state.selected_param {
                        0 => params.emin,
                        1 => params.emax,
                        2 => params.ec50,
                        3 => params.hillslope,
                        _ => params.emin,
                    };
                    (i as f64, value)
                })
                .collect();

            if !trace_data.is_empty() {
                let y_min_raw = trace_data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
                let y_max_raw = trace_data.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
                let y_range = y_max_raw - y_min_raw;
                let y_padding = if y_range > 0.0 { y_range * 0.05 } else { 1.0 }; // 5% padding or minimum
                let y_min = y_min_raw - y_padding;
                let y_max = y_max_raw + y_padding;
                let x_max = trace_data.len() as f64;

                let dataset = Dataset::default()
                    .name(format!("{} Trace", param_name))
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::Yellow))
                    .graph_type(GraphType::Line)
                    .data(&trace_data);

                let chart = Chart::new(vec![dataset])
                    .block(Block::default()
                        .borders(Borders::ALL)
                        .title(format!("MCMC Trace: {} (Use ↑/↓ to select parameter)", param_name))
                    )
                    .x_axis(
                        Axis::default()
                            .title("Iteration")
                            .style(Style::default().fg(Color::Gray))
                            .bounds([0.0, x_max])
                    )
                    .y_axis(
                        Axis::default()
                            .title("Parameter Value")
                            .style(Style::default().fg(Color::Gray))
                            .bounds([y_min, y_max])
                    );

                frame.render_widget(chart, area);
            }
        } else {
            let no_data = Paragraph::new("No MCMC results available")
                .block(Block::default().borders(Borders::ALL).title("MCMC Traces"))
                .alignment(Alignment::Center);
            frame.render_widget(no_data, area);
        }
    }

    fn draw_posteriors(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        if let Some(results) = &self.results {
            let param_names = ["Emin", "Emax", "EC50", "Hill Slope"];
            let param_name = param_names[app_state.selected_param];

            let param_values: Vec<f64> = results.samples
                .iter()
                .map(|params| match app_state.selected_param {
                    0 => params.emin,
                    1 => params.emax,
                    2 => params.ec50,
                    3 => params.hillslope,
                    _ => params.emin,
                })
                .collect();

            if !param_values.is_empty() {
                // Create histogram
                let min_val = param_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = param_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let n_bins = 50;
                let bin_width = (max_val - min_val) / n_bins as f64;

                let mut histogram = vec![0; n_bins];
                for &value in &param_values {
                    let bin_idx = ((value - min_val) / bin_width).floor() as usize;
                    let bin_idx = bin_idx.min(n_bins - 1);
                    histogram[bin_idx] += 1;
                }

                let max_count = histogram.iter().max().copied().unwrap_or(1) as f64;
                let hist_data: Vec<(f64, f64)> = histogram
                    .into_iter()
                    .enumerate()
                    .map(|(i, count)| {
                        let x = min_val + (i as f64 + 0.5) * bin_width;
                        let y = count as f64 / max_count;
                        (x, y)
                    })
                    .collect();

                let dataset = Dataset::default()
                    .name(format!("{} Posterior", param_name))
                    .marker(Marker::Block)
                    .style(Style::default().fg(Color::Magenta))
                    .graph_type(GraphType::Scatter)
                    .data(&hist_data);

                let chart = Chart::new(vec![dataset])
                    .block(Block::default()
                        .borders(Borders::ALL)
                        .title(format!("Posterior Distribution: {} (Use ↑/↓ to select)", param_name))
                    )
                    .x_axis(
                        Axis::default()
                            .title("Parameter Value")
                            .style(Style::default().fg(Color::Gray))
                            .bounds([min_val, max_val])
                    )
                    .y_axis(
                        Axis::default()
                            .title("Normalized Frequency")
                            .style(Style::default().fg(Color::Gray))
                            .bounds([0.0, 1.0])
                    );

                frame.render_widget(chart, area);
            }
        } else {
            let no_data = Paragraph::new("No MCMC results available")
                .block(Block::default().borders(Borders::ALL).title("Posterior Distributions"))
                .alignment(Alignment::Center);
            frame.render_widget(no_data, area);
        }
    }

    fn draw_diagnostics(&self, frame: &mut Frame, area: Rect, _app_state: &AppState) {
        if let (Some(results), Some(summary)) = (&self.results, &self.summary) {
            let diagnostics_text = vec![
                "MCMC Diagnostics".into(),
                "".into(),
                format!("Number of samples: {}", results.samples.len()).into(),
                format!("Acceptance rate: {:.1}%", summary.acceptance_rate * 100.0).into(),
                "".into(),
                "Parameter Estimates (Mean ± SD):".into(),
                format!("  Emin:      {:.4} ± {:.4}", summary.emin.mean, summary.emin.std).into(),
                format!("  Emax:      {:.4} ± {:.4}", summary.emax.mean, summary.emax.std).into(),
                format!("  EC50:      {:.4} ± {:.4} (log10)", summary.ec50.mean, summary.ec50.std).into(),
                format!("  EC50:      {:.2e} (linear)", 10.0_f64.powf(summary.ec50.mean)).into(),
                format!("  Hill:      {:.4} ± {:.4}", summary.hillslope.mean, summary.hillslope.std).into(),
                "".into(),
                "95% Credible Intervals:".into(),
                format!("  Emin:  [{:.4}, {:.4}]", summary.emin.ci_lower, summary.emin.ci_upper).into(),
                format!("  Emax:  [{:.4}, {:.4}]", summary.emax.ci_lower, summary.emax.ci_upper).into(),
                format!("  EC50:  [{:.4}, {:.4}]", summary.ec50.ci_lower, summary.ec50.ci_upper).into(),
                format!("  Hill:  [{:.4}, {:.4}]", summary.hillslope.ci_lower, summary.hillslope.ci_upper).into(),
            ];

            let diagnostics_paragraph = Paragraph::new(diagnostics_text)
                .block(Block::default().borders(Borders::ALL).title("MCMC Diagnostics"))
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true });

            frame.render_widget(diagnostics_paragraph, area);
        } else {
            let no_data = Paragraph::new("No MCMC results available")
                .block(Block::default().borders(Borders::ALL).title("Diagnostics"))
                .alignment(Alignment::Center);
            frame.render_widget(no_data, area);
        }
    }
}

// Helper function to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}


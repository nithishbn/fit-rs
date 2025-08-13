use crate::{BayesianEC50Fitter, DoseResponse, LL4Parameters, MCMCResult, ParameterSummary, Prior, PriorType};
use crate::config::MCMCConfig;
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
    original_prior: Prior, // Store original priors from parameters.json
    mcmc_config: MCMCConfig, // Store original MCMC configuration
    data: Vec<DoseResponse>,
    results: Option<MCMCResult>,
    summary: Option<ParameterSummary>,
    is_refitting: bool,
    refit_progress: String,
}

#[derive(Clone, Copy, PartialEq)]
enum PlotMode {
    DoseResponse,
    MCMCTraces,
    Posteriors,
    Diagnostics,
    ParameterEdit,
}

struct AppState {
    mode: PlotMode,
    selected_param: usize, // For MCMC traces and posteriors
    show_help: bool,
    zoom_level: f64,
    offset_x: f64,
    offset_y: f64,
    // Parameter editing state
    editing_param: usize, // 0=ec50_mean, 1=ec50_std, 2=emin_mean, 3=emin_std, 4=emax_mean, 5=emax_std, 6=hillslope_mean, 7=hillslope_std
    input_mode: bool,
    input_buffer: String,
    custom_values: [Option<f64>; 8], // [ec50_mean, ec50_std, emin_mean, emin_std, emax_mean, emax_std, hill_mean, hill_std]
    trigger_refit: bool,
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
            editing_param: 0,
            input_mode: false,
            input_buffer: String::new(),
            custom_values: [None; 8],
            trigger_refit: false,
        }
    }
}

impl TuiPlotter {
    pub fn new(fitter: BayesianEC50Fitter, data: Vec<DoseResponse>, mcmc_config: MCMCConfig) -> Self {
        let original_prior = fitter.prior.clone(); // Store original priors
        Self {
            fitter,
            original_prior,
            mcmc_config,
            data,
            results: None,
            summary: None,
            is_refitting: false,
            refit_progress: String::new(),
        }
    }

    pub fn with_results(mut self, results: MCMCResult, summary: ParameterSummary) -> Self {
        self.results = Some(results);
        self.summary = Some(summary);
        self
    }

    fn refit_with_custom_params(&mut self, custom_params: LL4Parameters, custom_stds: Option<[f64; 4]>) {
        // Use custom standard deviations if provided, otherwise use reasonable defaults for meaningful uncertainty
        let default_stds = [1.0, 0.5, 1.0, 0.5]; // [ec50, emin, emax, hillslope] - looser for visible confidence bands
        let stds = custom_stds.unwrap_or(default_stds);
        
        // Create new priors centered around the custom parameters
        let new_prior = Prior {
            emin: PriorType::Normal { 
                mean: custom_params.emin, 
                std: stds[1] 
            },
            emax: PriorType::Normal { 
                mean: custom_params.emax, 
                std: stds[2] 
            },
            ec50: PriorType::Normal { 
                mean: custom_params.ec50, 
                std: stds[0] 
            },
            hillslope: PriorType::Normal { 
                mean: custom_params.hillslope, 
                std: stds[3] 
            },
        };

        // Create new fitter with updated priors
        let new_fitter = BayesianEC50Fitter::new(self.data.clone())
            .with_prior(new_prior)
            .with_sigma(self.fitter.sigma);

        // Run MCMC fit using the same sample counts as initially passed in
        let new_result = new_fitter.fit(self.mcmc_config.samples, self.mcmc_config.burnin);
        let new_summary = new_fitter.summarize_results(&new_result);

        // Update results only, keep original fitter with original priors
        self.results = Some(new_result);
        self.summary = Some(new_summary);
        // Don't update self.fitter - keep the original priors
    }

    pub fn run_interactive_plot(&mut self) -> Result<()> {
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
                        KeyCode::Char('h') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                // In parameter edit input mode, 'h' is for text input
                                app_state.input_buffer.push('h');
                            } else {
                                // Otherwise, 'h' toggles help
                                app_state.show_help = !app_state.show_help;
                            }
                        }
                        KeyCode::F(1) => {
                            app_state.show_help = !app_state.show_help;
                        }
                        KeyCode::Char('d') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('d');
                            } else {
                                app_state.mode = PlotMode::DoseResponse;
                            }
                        }
                        KeyCode::Char('t') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('t');
                            } else {
                                app_state.mode = PlotMode::MCMCTraces;
                            }
                        }
                        KeyCode::Char('p') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('p');
                            } else {
                                app_state.mode = PlotMode::Posteriors;
                            }
                        }
                        KeyCode::Char('i') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('i');
                            } else {
                                app_state.mode = PlotMode::Diagnostics;
                            }
                        }
                        KeyCode::Char('e') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('e');
                            } else {
                                app_state.mode = PlotMode::ParameterEdit;
                            }
                        }
                        KeyCode::Char('k') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('k');
                            } else {
                                match app_state.mode {
                                    PlotMode::MCMCTraces | PlotMode::Posteriors => {
                                        if app_state.selected_param > 0 {
                                            app_state.selected_param -= 1;
                                        }
                                    }
                                    PlotMode::ParameterEdit => {
                                        if app_state.editing_param > 0 {
                                            app_state.editing_param -= 1;
                                        }
                                    }
                                    _ => {
                                        app_state.offset_y += 0.1;
                                    }
                                }
                            }
                        }
                        KeyCode::Up => {
                            match app_state.mode {
                                PlotMode::MCMCTraces | PlotMode::Posteriors => {
                                    if app_state.selected_param > 0 {
                                        app_state.selected_param -= 1;
                                    }
                                }
                                PlotMode::ParameterEdit => {
                                    if !app_state.input_mode && app_state.editing_param > 0 {
                                        app_state.editing_param -= 1;
                                    }
                                }
                                _ => {
                                    app_state.offset_y += 0.1;
                                }
                            }
                        }
                        KeyCode::Char('j') => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('j');
                            } else {
                                match app_state.mode {
                                    PlotMode::MCMCTraces | PlotMode::Posteriors => {
                                        if app_state.selected_param < 3 {
                                            app_state.selected_param += 1;
                                        }
                                    }
                                    PlotMode::ParameterEdit => {
                                        if app_state.editing_param < 7 {
                                            app_state.editing_param += 1;
                                        }
                                    }
                                    _ => {
                                        app_state.offset_y -= 0.1;
                                    }
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
                                PlotMode::ParameterEdit => {
                                    if !app_state.input_mode && app_state.editing_param < 7 {
                                        app_state.editing_param += 1;
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
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.push('-');
                            } else {
                                app_state.zoom_level /= 1.2;
                            }
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
                        KeyCode::Char('f') => {
                            // Fast refit - trigger immediate MCMC refitting in parameter edit mode
                            if app_state.mode == PlotMode::ParameterEdit && !app_state.input_mode {
                                // Check if any custom values have been set
                                if app_state.custom_values.iter().any(|v| v.is_some()) {
                                    app_state.trigger_refit = true;
                                }
                            }
                        }
                        KeyCode::Enter => {
                            if app_state.mode == PlotMode::ParameterEdit {
                                if app_state.input_mode {
                                    // Confirm input - just store the individual value
                                    if let Ok(value) = app_state.input_buffer.parse::<f64>() {
                                        // Store only the individual parameter that was edited
                                        app_state.custom_values[app_state.editing_param] = Some(value);
                                        // Don't auto-refit, user must press 'f' to refit
                                    }
                                    app_state.input_mode = false;
                                    app_state.input_buffer.clear();
                                } else {
                                    // Start input
                                    app_state.input_mode = true;
                                    // Pre-fill with current value (custom if set, otherwise original prior)
                                    let current_value = app_state.custom_values[app_state.editing_param].unwrap_or_else(|| {
                                        match app_state.editing_param {
                                            0 => match &self.original_prior.ec50 {
                                                PriorType::Normal { mean, .. } => *mean,
                                                _ => -6.0,
                                            },
                                            1 => match &self.original_prior.ec50 {
                                                PriorType::Normal { std, .. } => *std,
                                                _ => 1.0,
                                            },
                                            2 => match &self.original_prior.emin {
                                                PriorType::Normal { mean, .. } => *mean,
                                                _ => 0.0,
                                            },
                                            3 => match &self.original_prior.emin {
                                                PriorType::Normal { std, .. } => *std,
                                                _ => 0.5,
                                            },
                                            4 => match &self.original_prior.emax {
                                                PriorType::Normal { mean, .. } => *mean,
                                                _ => 100.0,
                                            },
                                            5 => match &self.original_prior.emax {
                                                PriorType::Normal { std, .. } => *std,
                                                _ => 1.0,
                                            },
                                            6 => match &self.original_prior.hillslope {
                                                PriorType::Normal { mean, .. } => *mean,
                                                _ => 1.0,
                                            },
                                            7 => match &self.original_prior.hillslope {
                                                PriorType::Normal { std, .. } => *std,
                                                _ => 0.5,
                                            },
                                            _ => 0.0,
                                        }
                                    });
                                    app_state.input_buffer = format!("{:.4}", current_value);
                                }
                            }
                        }
                        KeyCode::Backspace => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                app_state.input_buffer.pop();
                            }
                        }
                        KeyCode::Char(c) => {
                            if app_state.mode == PlotMode::ParameterEdit && app_state.input_mode {
                                // Allow all numeric input and scientific notation
                                if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+' {
                                    app_state.input_buffer.push(c);
                                }
                            }
                            // Note: Letter keys for mode switching are handled above
                        }
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }

            // Handle refitting trigger
            if app_state.trigger_refit {
                // Build parameters from custom values or original priors
                let custom_params = LL4Parameters {
                    ec50: app_state.custom_values[0].unwrap_or_else(|| {
                        match &self.original_prior.ec50 {
                            PriorType::Normal { mean, .. } => *mean,
                            _ => -6.0,
                        }
                    }),
                    emin: app_state.custom_values[2].unwrap_or_else(|| {
                        match &self.original_prior.emin {
                            PriorType::Normal { mean, .. } => *mean,
                            _ => 0.0,
                        }
                    }),
                    emax: app_state.custom_values[4].unwrap_or_else(|| {
                        match &self.original_prior.emax {
                            PriorType::Normal { mean, .. } => *mean,
                            _ => 100.0,
                        }
                    }),
                    hillslope: app_state.custom_values[6].unwrap_or_else(|| {
                        match &self.original_prior.hillslope {
                            PriorType::Normal { mean, .. } => *mean,
                            _ => 1.0,
                        }
                    }),
                };
                
                let custom_stds = [
                    app_state.custom_values[1].unwrap_or_else(|| {
                        match &self.original_prior.ec50 {
                            PriorType::Normal { std, .. } => *std,
                            _ => 1.0,
                        }
                    }),
                    app_state.custom_values[3].unwrap_or_else(|| {
                        match &self.original_prior.emin {
                            PriorType::Normal { std, .. } => *std,
                            _ => 0.5,
                        }
                    }),
                    app_state.custom_values[5].unwrap_or_else(|| {
                        match &self.original_prior.emax {
                            PriorType::Normal { std, .. } => *std,
                            _ => 1.0,
                        }
                    }),
                    app_state.custom_values[7].unwrap_or_else(|| {
                        match &self.original_prior.hillslope {
                            PriorType::Normal { std, .. } => *std,
                            _ => 0.5,
                        }
                    }),
                ];

                self.is_refitting = true;
                self.refit_progress = "Refitting curve with new parameters...".to_string();
                
                // Draw progress immediately
                terminal.draw(|f| self.draw_ui(f, &app_state))?;
                
                // Perform the refit
                self.refit_with_custom_params(custom_params, Some(custom_stds));
                
                self.is_refitting = false;
                self.refit_progress.clear();
                // Don't clear custom values - keep them for further editing
                
                app_state.trigger_refit = false;
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

        // Show refitting overlay if in progress
        if self.is_refitting {
            self.draw_refitting_overlay(frame, size);
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
            PlotMode::ParameterEdit => self.draw_parameter_edit(frame, main_layout[1], app_state),
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
            PlotMode::ParameterEdit => "Parameter Editor",
        };

        let header = Paragraph::new(title)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);

        frame.render_widget(header, area);
    }

    fn draw_status_bar(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        let mode_indicator = match app_state.mode {
            PlotMode::DoseResponse => "[d] Dose-Response",
            PlotMode::MCMCTraces => "[t] MCMC Traces",
            PlotMode::Posteriors => "[p] Posteriors",
            PlotMode::Diagnostics => "[i] Diagnostics",
            PlotMode::ParameterEdit => "[e] Parameter Edit",
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
            "  [d/t/p/i/e] Switch between plot modes".into(),
            "  [k/j] or [↑/↓] Select parameter or pan up/down".into(),
            "  [←/→]    Pan left/right".into(),
            "  [+/-]    Zoom in/out".into(),
            "  [r]      Reset zoom and pan".into(),
            "".into(),
            "Plot Modes:".into(),
            "  [d] Dose-Response Curve - Fitted curve with 95% confidence intervals".into(),
            "  [t] MCMC Traces - Parameter convergence over iterations".into(),
            "  [p] Posteriors - Parameter uncertainty distributions".into(),
            "  [i] Diagnostics - Acceptance rate and R-hat statistics".into(),
            "  [e] Parameter Editor - Manually adjust curve parameters".into(),
            "".into(),
            "Parameter Editor (in [e] mode):".into(),
            "  [k/j] or [↑/↓] Select prior parameter to edit".into(),
            "  [Enter]  Start editing / Confirm and REFIT curve".into(),
            "  [f]      Quick REFIT with current parameter values".into(),
            "  [0-9.-+eE] Type numeric values when editing".into(),
            "  [Backspace] Delete characters when editing".into(),
            "  Note: Edit PRIOR means/stds, not fitted values".into(),
            "  Note: Refitting runs MCMC with new priors + confidence bands".into(),
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
            // Clean confidence band boundaries without fill
            datasets.push(
                Dataset::default()
                    .name("95% CI Lower")
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::LightBlue))
                    .graph_type(GraphType::Line)
                    .data(lower)
            );
            datasets.push(
                Dataset::default()
                    .name("95% CI Upper")
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::LightBlue))
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

        // Add EC50 vertical line
        let ec50_line_data = if let Some(ref summary) = self.summary {
            // Use fitted EC50
            let ec50_log = summary.ec50.mean;
            vec![(ec50_log, plot_y_min), (ec50_log, plot_y_max)]
        } else {
            vec![]
        };

        if !ec50_line_data.is_empty() {
            datasets.push(
                Dataset::default()
                    .name("EC50")
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::Magenta))
                    .graph_type(GraphType::Line)
                    .data(&ec50_line_data)
            );
        }

        let title = if confidence_bands.is_some() {
            "Dose-Response Curve with 95% Confidence Intervals"
        } else {
            "Dose-Response Curve (Preview)"
        };
        
        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(title))
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

    fn draw_parameter_edit(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        // Split area into parameter controls and preview plot
        let layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(40), // Parameter controls
                Constraint::Min(0),     // Preview plot
            ])
            .split(area);

        self.draw_parameter_controls(frame, layout[0], app_state);
        self.draw_parameter_preview(frame, layout[1], app_state);
    }

    fn draw_parameter_controls(&self, frame: &mut Frame, area: Rect, app_state: &AppState) {
        let param_labels = [
            "EC50 (log10) Prior Mean", "EC50 (log10) Prior Std",
            "Emin Prior Mean", "Emin Prior Std", 
            "Emax Prior Mean", "Emax Prior Std",
            "Hill Slope Prior Mean", "Hill Slope Prior Std"
        ];
        
        // Get current prior values from the original priors (from parameters.json)
        let prior_means = [
            match &self.original_prior.ec50 {
                PriorType::Normal { mean, .. } => *mean,
                _ => -6.0,
            },
            match &self.original_prior.emin {
                PriorType::Normal { mean, .. } => *mean,
                _ => 0.0,
            },
            match &self.original_prior.emax {
                PriorType::Normal { mean, .. } => *mean,
                _ => 100.0,
            },
            match &self.original_prior.hillslope {
                PriorType::Normal { mean, .. } => *mean,
                _ => 1.0,
            },
        ];

        let prior_stds = [
            match &self.original_prior.ec50 {
                PriorType::Normal { std, .. } => *std,
                _ => 1.0,
            },
            match &self.original_prior.emin {
                PriorType::Normal { std, .. } => *std,
                _ => 0.5,
            },
            match &self.original_prior.emax {
                PriorType::Normal { std, .. } => *std,
                _ => 1.0,
            },
            match &self.original_prior.hillslope {
                PriorType::Normal { std, .. } => *std,
                _ => 0.5,
            },
        ];

        // Build param_values using individual custom values or original priors
        let param_values = [
            // EC50 mean, std
            app_state.custom_values[0].unwrap_or(prior_means[0]), 
            app_state.custom_values[1].unwrap_or(prior_stds[0]),
            // Emin mean, std  
            app_state.custom_values[2].unwrap_or(prior_means[1]), 
            app_state.custom_values[3].unwrap_or(prior_stds[1]),
            // Emax mean, std
            app_state.custom_values[4].unwrap_or(prior_means[2]), 
            app_state.custom_values[5].unwrap_or(prior_stds[2]),
            // Hill mean, std
            app_state.custom_values[6].unwrap_or(prior_means[3]), 
            app_state.custom_values[7].unwrap_or(prior_stds[3]),
        ];

        let mut text_lines = vec![
            "Prior Parameter Editor (from parameters.json)".into(),
            "".into(),
            "Use k/j or ↑/↓ to select prior".into(),
            "Press Enter to edit prior value".into(),
            "Type numbers (0-9, ., -, e, E, +)".into(),
            "Enter confirms edit (no refit)".into(),
            "Press [f] to REFIT curve with new priors".into(),
            "Press [d] to see refitted results".into(),
            "".into(),
            "Editing PRIORS (independent of fit results):".into(),
            "".into(),
        ];

        for (i, (label, value)) in param_labels.iter().zip(param_values.iter()).enumerate() {
            let line = if i == app_state.editing_param {
                if app_state.input_mode {
                    format!("▶ {}: {}", label, app_state.input_buffer)
                } else {
                    format!("▶ {}: {:.4}", label, value)
                }
            } else {
                format!("  {}: {:.4}", label, value)
            };
            
            text_lines.push(line.into());
        }

        text_lines.push("".into());
        if app_state.input_mode {
            text_lines.push("Enter to confirm, Backspace to edit".into());
        } else {
            text_lines.push("Enter to edit selected prior".into());
        }

        // Show current fitted values for reference only
        if let Some(ref summary) = self.summary {
            text_lines.push("".into());
            text_lines.push("Latest fitted values (reference only):".into());
            text_lines.push(format!("  EC50: {:.4} ± {:.4}", summary.ec50.mean, summary.ec50.std).into());
            text_lines.push(format!("  Emin: {:.4} ± {:.4}", summary.emin.mean, summary.emin.std).into());
            text_lines.push(format!("  Emax: {:.4} ± {:.4}", summary.emax.mean, summary.emax.std).into());
            text_lines.push(format!("  Hill: {:.4} ± {:.4}", summary.hillslope.mean, summary.hillslope.std).into());
            text_lines.push("Note: Priors stay from parameters.json".into());
        }

        let controls = Paragraph::new(text_lines)
            .block(Block::default().borders(Borders::ALL).title("Controls"))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: true });

        frame.render_widget(controls, area);
    }

    fn draw_parameter_preview(&self, frame: &mut Frame, area: Rect, _app_state: &AppState) {
        // Draw a dose-response plot with current parameters AND confidence bands if available
        let data_points: Vec<(f64, f64)> = self.data
            .iter()
            .map(|d| (d.concentration.log10(), d.response))
            .collect();

        // Get current parameters for preview - only use fitted results, not custom params
        let params = if let Some(ref summary) = self.summary {
            LL4Parameters {
                emin: summary.emin.mean,
                emax: summary.emax.mean,
                ec50: summary.ec50.mean,
                hillslope: summary.hillslope.mean,
            }
        } else {
            LL4Parameters {
                emin: 0.0,
                emax: 100.0,
                ec50: -6.0,
                hillslope: 1.0,
            }
        };

        // Generate fitted curve and confidence bands
        let log_conc_min = data_points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min) - 2.0;
        let log_conc_max = data_points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max) + 2.0;
        
        let x_points: Vec<f64> = (0..100)
            .map(|i| log_conc_min + (log_conc_max - log_conc_min) * i as f64 / 99.0)
            .collect();

        // Main fitted curve with current parameters
        let fitted_points: Vec<(f64, f64)> = x_points
            .iter()
            .map(|&log_conc| {
                let conc = 10.0_f64.powf(log_conc);
                let response = self.fitter.ll4_model(conc, &params);
                (log_conc, response)
            })
            .collect();

        // Calculate confidence bands using existing MCMC samples if available
        let confidence_bands = if let Some(ref results) = self.results {
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

            Some((confidence_lower, confidence_upper))
        } else {
            None
        };

        // Calculate plot bounds including confidence bands
        let mut all_y_values = data_points.iter().map(|(_, y)| *y).collect::<Vec<_>>();
        all_y_values.extend(fitted_points.iter().map(|(_, y)| *y));
        if let Some((ref lower, ref upper)) = confidence_bands {
            all_y_values.extend(lower.iter().map(|(_, y)| *y));
            all_y_values.extend(upper.iter().map(|(_, y)| *y));
        }
        
        let y_min_raw = all_y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max_raw = all_y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_range_raw = y_max_raw - y_min_raw;
        let y_padding = y_range_raw * 0.1;
        let y_min = y_min_raw - y_padding;
        let y_max = y_max_raw + y_padding;

        let mut datasets = vec![];

        // Add confidence bands first (behind the curve)
        if let Some((ref lower, ref upper)) = confidence_bands {
            datasets.push(
                Dataset::default()
                    .name("95% CI Lower")
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::LightBlue))
                    .graph_type(GraphType::Line)
                    .data(lower)
            );
            datasets.push(
                Dataset::default()
                    .name("95% CI Upper")
                    .marker(Marker::Braille)
                    .style(Style::default().fg(Color::LightBlue))
                    .graph_type(GraphType::Line)
                    .data(upper)
            );
        }

        // Add fitted curve
        datasets.push(
            Dataset::default()
                .name("Fitted Curve")
                .marker(Marker::Braille)
                .style(Style::default().fg(Color::Green))
                .graph_type(GraphType::Line)
                .data(&fitted_points)
        );

        // Add data points on top
        datasets.push(
            Dataset::default()
                .name("Data Points")
                .marker(Marker::Block)
                .style(Style::default().fg(Color::Red))
                .graph_type(GraphType::Scatter)
                .data(&data_points)
        );

        // Add EC50 vertical line
        let ec50_line_data = vec![(params.ec50, y_min), (params.ec50, y_max)];
        datasets.push(
            Dataset::default()
                .name("EC50")
                .marker(Marker::Braille)
                .style(Style::default().fg(Color::Magenta))
                .graph_type(GraphType::Line)
                .data(&ec50_line_data)
        );

        let title = if confidence_bands.is_some() {
            "Preview with 95% CI"
        } else {
            "Preview"
        };

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(title))
            .x_axis(
                Axis::default()
                    .title("Log10(Concentration)")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([log_conc_min, log_conc_max])
                    .labels(vec![
                        format!("{:.1}", log_conc_min).into(),
                        format!("{:.1}", (log_conc_min + log_conc_max) / 2.0).into(),
                        format!("{:.1}", log_conc_max).into(),
                    ])
            )
            .y_axis(
                Axis::default()
                    .title("Response")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        format!("{:.1}", y_min).into(),
                        format!("{:.1}", (y_min + y_max) / 2.0).into(),
                        format!("{:.1}", y_max).into(),
                    ])
            );

        frame.render_widget(chart, area);
    }

    fn draw_refitting_overlay(&self, frame: &mut Frame, area: Rect) {
        let progress_text = vec![
            "🔄 Interactive Refitting".into(),
            "".into(),
            "Running MCMC (800 samples)...".into(),
            "Updating curve & confidence bands".into(),
            "".into(),
            "⚡ Interactive mode - optimized for speed".into(),
        ];

        let progress_paragraph = Paragraph::new(progress_text)
            .block(
                Block::default()
                    .title("Refitting")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
            )
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: true })
            .alignment(Alignment::Center);

        // Center the progress overlay
        let progress_area = centered_rect(50, 30, area);
        frame.render_widget(Clear, progress_area);
        frame.render_widget(progress_paragraph, progress_area);
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


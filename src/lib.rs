use rand::prelude::*;
use rand_distr::{LogNormal, Normal, Uniform};
use std::f64::consts::PI;
pub mod io;
pub mod visualization;
pub mod stan_backend;
pub mod config;

use anyhow::Result;
pub use stan_backend::{MCMCSampler, StanSampler, precompile_stan_model, force_recompile_stan_model};
pub use config::Config;
#[derive(Debug, Clone)]
pub struct DoseResponse {
    pub concentration: f64,
    pub response: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LL4Parameters {
    pub emin: f64,      // Lower asymptote
    pub emax: f64,      // Upper asymptote
    pub ec50: f64,      // Inflection point (log10 scale)
    pub hillslope: f64, // Hill slope
}

#[derive(Debug, Clone)]
pub struct Prior {
    pub emin: PriorType,
    pub emax: PriorType,
    pub ec50: PriorType,
    pub hillslope: PriorType,
}

#[derive(Debug, Clone)]
pub enum PriorType {
    Normal { mean: f64, std: f64 },
    LogNormal { mean: f64, std: f64 },
    Uniform { min: f64, max: f64 },
}

#[derive(Debug)]
pub struct MCMCResult {
    pub samples: Vec<LL4Parameters>,
    pub acceptance_rate: f64,
    pub log_likelihood: Vec<f64>,
}

pub struct BayesianEC50Fitter {
    pub data: Vec<DoseResponse>,
    pub prior: Prior,
    pub sigma: f64, // Error standard deviation
}

impl BayesianEC50Fitter {
    pub fn new(data: Vec<DoseResponse>) -> Self {
        // Default priors similar to brms defaults
        let prior = Prior {
            emin: PriorType::Normal {
                mean: 0.0,
                std: 10.0,
            },
            emax: PriorType::Normal {
                mean: 100.0,
                std: 20.0,
            },
            ec50: PriorType::Normal {
                mean: -6.0,
                std: 2.0,
            }, // log10 scale
            hillslope: PriorType::Normal {
                mean: 1.0,
                std: 1.0,
            },
        };

        Self {
            data,
            prior,
            sigma: 1.0,
        }
    }

    pub fn with_prior(mut self, prior: Prior) -> Self {
        self.prior = prior;
        self
    }

    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// 4-parameter logistic function
    fn ll4_model(&self, concentration: f64, params: &LL4Parameters) -> f64 {
        let log_conc = concentration.log10();
        params.emin
            + (params.emax - params.emin)
                / (1.0 + 10.0_f64.powf((params.ec50 - log_conc) * params.hillslope))
    }

    /// Calculate log-likelihood of data given parameters
    fn log_likelihood(&self, params: &LL4Parameters) -> f64 {
        let mut log_lik = 0.0;

        for point in &self.data {
            let predicted = self.ll4_model(point.concentration, params);
            let residual = point.response - predicted;
            log_lik += -0.5 * (residual * residual) / (self.sigma * self.sigma);
            log_lik += -0.5 * (2.0 * PI * self.sigma * self.sigma).ln();
        }

        log_lik
    }

    /// Calculate log-prior probability
    fn log_prior(&self, params: &LL4Parameters) -> f64 {
        let mut log_prior = 0.0;

        log_prior += self.evaluate_prior(&self.prior.emin, params.emin);
        log_prior += self.evaluate_prior(&self.prior.emax, params.emax);
        log_prior += self.evaluate_prior(&self.prior.ec50, params.ec50);
        log_prior += self.evaluate_prior(&self.prior.hillslope, params.hillslope);

        log_prior
    }

    fn evaluate_prior(&self, prior: &PriorType, value: f64) -> f64 {
        match prior {
            PriorType::Normal { mean, std } => {
                let z = (value - mean) / std;
                -0.5 * z * z - 0.5 * (2.0 * PI * std * std).ln()
            }
            PriorType::LogNormal { mean, std } => {
                if value <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    let log_val = value.ln();
                    let z = (log_val - mean) / std;
                    -0.5 * z * z - 0.5 * (2.0 * PI * std * std).ln() - log_val
                }
            }
            PriorType::Uniform { min, max } => {
                if value >= *min && value <= *max {
                    -(max - min).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Sample from prior distribution
    fn sample_from_prior(&self, rng: &mut ThreadRng, prior: &PriorType) -> f64 {
        match prior {
            PriorType::Normal { mean, std } => {
                let normal = Normal::new(*mean, *std).unwrap();
                normal.sample(rng)
            }
            PriorType::LogNormal { mean, std } => {
                let lognormal = LogNormal::new(*mean, *std).unwrap();
                lognormal.sample(rng)
            }
            PriorType::Uniform { min, max } => {
                let uniform = Uniform::new(*min, *max);
                uniform.sample(rng)
            }
        }
    }

    /// Initialize parameters from prior
    fn initialize_params(&self, rng: &mut ThreadRng) -> LL4Parameters {
        LL4Parameters {
            emin: self.sample_from_prior(rng, &self.prior.emin),
            emax: self.sample_from_prior(rng, &self.prior.emax),
            ec50: self.sample_from_prior(rng, &self.prior.ec50),
            hillslope: self.sample_from_prior(rng, &self.prior.hillslope),
        }
    }

    /// Propose new parameters (random walk)
    fn propose_params(&self, current: &LL4Parameters, rng: &mut ThreadRng) -> LL4Parameters {
        let proposal_std = 0.1; // Tuning parameter
        let normal = Normal::new(0.0, proposal_std).unwrap();

        LL4Parameters {
            emin: current.emin + normal.sample(rng),
            emax: current.emax + normal.sample(rng),
            ec50: current.ec50 + normal.sample(rng),
            hillslope: current.hillslope + normal.sample(rng),
        }
    }

    /// Run Metropolis-Hastings MCMC
    pub fn fit(&self, n_samples: usize, burnin: usize) -> MCMCResult {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(n_samples);
        let mut log_likelihoods = Vec::with_capacity(n_samples);
        let mut n_accepted = 0;

        // Initialize
        let mut current_params = self.initialize_params(&mut rng);
        let mut current_log_posterior =
            self.log_likelihood(&current_params) + self.log_prior(&current_params);

        for i in 0..(n_samples + burnin) {
            // Propose new parameters
            let proposed_params = self.propose_params(&current_params, &mut rng);
            let proposed_log_posterior =
                self.log_likelihood(&proposed_params) + self.log_prior(&proposed_params);

            // Accept/reject
            let log_ratio = proposed_log_posterior - current_log_posterior;
            let accept_prob = log_ratio.exp().min(1.0);

            if rng.gen::<f64>() < accept_prob {
                current_params = proposed_params;
                current_log_posterior = proposed_log_posterior;
                n_accepted += 1;
            }

            // Store samples after burnin
            if i >= burnin {
                samples.push(current_params);
                log_likelihoods.push(self.log_likelihood(&current_params));
            }
        }

        let acceptance_rate = n_accepted as f64 / (n_samples + burnin) as f64;

        MCMCResult {
            samples,
            acceptance_rate,
            log_likelihood: log_likelihoods,
        }
    }

    /// Calculate summary statistics from MCMC samples
    pub fn summarize_results(&self, result: &MCMCResult) -> ParameterSummary {
        let n = result.samples.len();

        // Extract parameter vectors
        let emins: Vec<f64> = result.samples.iter().map(|p| p.emin).collect();
        let emaxs: Vec<f64> = result.samples.iter().map(|p| p.emax).collect();
        let ec50s: Vec<f64> = result.samples.iter().map(|p| p.ec50).collect();
        let hillslopes: Vec<f64> = result.samples.iter().map(|p| p.hillslope).collect();

        ParameterSummary {
            emin: self.calculate_param_stats(&emins),
            emax: self.calculate_param_stats(&emaxs),
            ec50: self.calculate_param_stats(&ec50s),
            hillslope: self.calculate_param_stats(&hillslopes),
            n_samples: n,
            acceptance_rate: result.acceptance_rate,
        }
    }

    fn calculate_param_stats(&self, values: &[f64]) -> ParamStats {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std = variance.sqrt();

        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // 95% credible interval
        let ci_lower = sorted[(0.025 * n as f64) as usize];
        let ci_upper = sorted[(0.975 * n as f64) as usize];

        ParamStats {
            mean,
            median,
            std,
            ci_lower,
            ci_upper,
        }
    }
}

impl MCMCSampler for BayesianEC50Fitter {
    fn fit(&self, n_samples: usize, burnin: usize, _chains: Option<usize>) -> Result<MCMCResult> {
        // Use the existing Metropolis-Hastings implementation
        Ok(self.fit(n_samples, burnin))
    }
    
    fn get_name(&self) -> &'static str {
        "Metropolis-Hastings"
    }
}

#[derive(Debug)]
pub struct ParameterSummary {
    pub emin: ParamStats,
    pub emax: ParamStats,
    pub ec50: ParamStats,
    pub hillslope: ParamStats,
    pub n_samples: usize,
    pub acceptance_rate: f64,
}

#[derive(Debug)]
pub struct ParamStats {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

// Example usage and test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stan_backend() {
        // Create test data
        let test_data = vec![
            DoseResponse { concentration: 1e-9, response: 10.0 },
            DoseResponse { concentration: 1e-8, response: 15.0 },
            DoseResponse { concentration: 1e-7, response: 25.0 },
            DoseResponse { concentration: 1e-6, response: 45.0 },
            DoseResponse { concentration: 1e-5, response: 70.0 },
        ];

        // Set up priors
        let priors = Prior {
            emin: PriorType::Normal { mean: 0.0, std: 10.0 },
            emax: PriorType::Normal { mean: 100.0, std: 20.0 },
            ec50: PriorType::Normal { mean: -6.0, std: 2.0 },
            hillslope: PriorType::Normal { mean: 1.0, std: 1.0 },
        };

        // Test Stan backend (note: this will only work if BridgeStan and model are available)
        if let Ok(stan_sampler) = StanSampler::new(test_data.clone(), priors.clone()) {
            if let Ok(result) = stan_sampler.fit(50, 25, Some(1)) {
                assert!(result.samples.len() > 0);
                assert!(result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0);
                println!("Stan backend test successful - {} samples", result.samples.len());
            } else {
                println!("Stan backend not available for testing (requires BridgeStan setup)");
            }
        } else {
            println!("Stan backend creation failed (requires Stan model file and BridgeStan)");
        }
    }

    #[test]
    fn test_ec50_fitting() {
        // Generate synthetic data
        let true_params = LL4Parameters {
            emin: 5.0,
            emax: 80.0,
            ec50: -5.0, // 1 µM
            hillslope: 1.0,
        };

        let concentrations: Vec<f64> = vec![1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3];
        let mut data = Vec::new();

        for &conc in &concentrations {
            let log_conc = conc.log10();
            let response = true_params.emin
                + (true_params.emax - true_params.emin)
                    / (1.0 + 10.0_f64.powf((true_params.ec50 - log_conc) * true_params.hillslope));
            // Add some noise
            let response = response + rand::thread_rng().gen_range(-5.0..5.0);

            data.push(DoseResponse {
                concentration: conc,
                response,
            });
        }

        // Fit the model
        let mut fitter = BayesianEC50Fitter::new(data);
        let priors = Prior {
            ec50: PriorType::Normal {
                mean: (-6.0),
                std: (1.0),
            },
            emin: PriorType::Normal {
                mean: (10.0),
                std: (1.0),
            },
            emax: PriorType::Normal {
                mean: (90.0),
                std: (1.0),
            },
            hillslope: PriorType::Normal {
                mean: (1.0),
                std: (1.0),
            },
        };
        fitter.prior = priors;
        let result = fitter.fit(4000, 500);
        let summary = fitter.summarize_results(&result);

        println!("Fitting Results:");
        println!("Acceptance rate: {:.3}", summary.acceptance_rate);
        println!(
            "Emin: {:.2} ± {:.2} [CI: {:.2}, {:.2}]",
            summary.emin.mean, summary.emin.std, summary.emin.ci_lower, summary.emin.ci_upper
        );
        println!(
            "Emax: {:.2} ± {:.2} [CI: {:.2}, {:.2}]",
            summary.emax.mean, summary.emax.std, summary.emax.ci_lower, summary.emax.ci_upper
        );
        println!(
            "EC50: {:.2} ± {:.2} [CI: {:.2}, {:.2}]",
            summary.ec50.mean, summary.ec50.std, summary.ec50.ci_lower, summary.ec50.ci_upper
        );
        println!(
            "Hill slope: {:.2} ± {:.2} [CI: {:.2}, {:.2}]",
            summary.hillslope.mean,
            summary.hillslope.std,
            summary.hillslope.ci_lower,
            summary.hillslope.ci_upper
        );

        // Basic sanity checks
        assert!(summary.acceptance_rate > 0.1 && summary.acceptance_rate < 0.9);
        assert!((summary.ec50.mean - true_params.ec50).abs() < 1.0);
    }
}

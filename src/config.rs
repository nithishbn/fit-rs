use crate::{Prior, PriorType};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub input: InputConfig,
    pub mcmc: MCMCConfig,
    pub priors: PriorConfig,
    pub plotting: PlottingConfig,
    #[serde(default)]
    pub metadata: MetadataConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub file: PathBuf,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCMCConfig {
    #[serde(default = "default_samples")]
    pub samples: usize,
    #[serde(default = "default_burnin")]
    pub burnin: usize,
    #[serde(default = "default_chains")]
    pub chains: usize,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    #[serde(default = "default_backend")]
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorConfig {
    pub emin: PriorSpec,
    pub emax: PriorSpec,
    pub ec50: PriorSpec,
    pub hillslope: PriorSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorSpec {
    #[serde(rename = "type")]
    pub prior_type: String,
    pub mean: f64,
    pub std: f64,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlottingConfig {
    #[serde(default)]
    pub bounds: BoundsConfig,
    #[serde(default)]
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BoundsConfig {
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataConfig {
    pub description: Option<String>,
    pub version: Option<String>,
    pub created: Option<String>,
}

// Default values
fn default_output_dir() -> PathBuf {
    PathBuf::from("ec50_output")
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

impl Config {
    /// Load configuration from a JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| anyhow!("Failed to read config file '{}': {}", path.as_ref().display(), e))?;
        
        let config: Config = serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse JSON config: {}", e))?;
        
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize config: {}", e))?;
        
        std::fs::write(&path, content)
            .map_err(|e| anyhow!("Failed to write config file '{}': {}", path.as_ref().display(), e))?;
        
        Ok(())
    }

    /// Create a default configuration
    pub fn default() -> Self {
        Self {
            input: InputConfig {
                file: PathBuf::from("data/data.csv"),
                output_dir: default_output_dir(),
            },
            mcmc: MCMCConfig {
                samples: default_samples(),
                burnin: default_burnin(),
                chains: default_chains(),
                sigma: default_sigma(),
                backend: default_backend(),
            },
            priors: PriorConfig {
                emin: PriorSpec {
                    prior_type: "normal".to_string(),
                    mean: 0.0,
                    std: 10.0,
                    min: None,
                    max: None,
                    description: Some("Lower asymptote (baseline response)".to_string()),
                },
                emax: PriorSpec {
                    prior_type: "normal".to_string(),
                    mean: 100.0,
                    std: 20.0,
                    min: None,
                    max: None,
                    description: Some("Upper asymptote (maximum response)".to_string()),
                },
                ec50: PriorSpec {
                    prior_type: "normal".to_string(),
                    mean: -6.0,
                    std: 2.0,
                    min: None,
                    max: None,
                    description: Some("EC50 on log10 scale".to_string()),
                },
                hillslope: PriorSpec {
                    prior_type: "normal".to_string(),
                    mean: 1.0,
                    std: 1.0,
                    min: None,
                    max: None,
                    description: Some("Hill slope (steepness of curve)".to_string()),
                },
            },
            plotting: PlottingConfig {
                bounds: BoundsConfig::default(),
                verbose: false,
            },
            metadata: MetadataConfig {
                description: Some("EC50 curve fitting configuration".to_string()),
                version: Some("1.0".to_string()),
                created: Some(chrono::Utc::now().to_rfc3339()),
            },
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate input file exists (if not relative path)
        if self.input.file.is_absolute() && !self.input.file.exists() {
            return Err(anyhow!("Input file does not exist: {}", self.input.file.display()));
        }

        // Validate MCMC parameters
        if self.mcmc.samples == 0 {
            return Err(anyhow!("MCMC samples must be > 0"));
        }
        if self.mcmc.chains == 0 {
            return Err(anyhow!("MCMC chains must be > 0"));
        }
        if self.mcmc.sigma <= 0.0 {
            return Err(anyhow!("MCMC sigma must be > 0"));
        }

        // Validate backend
        match self.mcmc.backend.as_str() {
            "mh" | "stan" => {},
            _ => return Err(anyhow!("Invalid backend '{}'. Must be 'mh' or 'stan'", self.mcmc.backend)),
        }

        // Validate priors
        self.validate_prior_spec(&self.priors.emin, "emin")?;
        self.validate_prior_spec(&self.priors.emax, "emax")?;
        self.validate_prior_spec(&self.priors.ec50, "ec50")?;
        self.validate_prior_spec(&self.priors.hillslope, "hillslope")?;

        Ok(())
    }

    fn validate_prior_spec(&self, spec: &PriorSpec, name: &str) -> Result<()> {
        match spec.prior_type.as_str() {
            "normal" => {
                if spec.std <= 0.0 {
                    return Err(anyhow!("Prior '{}': normal distribution std must be > 0", name));
                }
            },
            "lognormal" => {
                if spec.std <= 0.0 {
                    return Err(anyhow!("Prior '{}': log-normal distribution std must be > 0", name));
                }
            },
            "uniform" => {
                let min = spec.min.ok_or_else(|| anyhow!("Prior '{}': uniform distribution requires 'min'", name))?;
                let max = spec.max.ok_or_else(|| anyhow!("Prior '{}': uniform distribution requires 'max'", name))?;
                if min >= max {
                    return Err(anyhow!("Prior '{}': uniform distribution min must be < max", name));
                }
            },
            _ => return Err(anyhow!("Prior '{}': unsupported type '{}'. Must be 'normal', 'lognormal', or 'uniform'", 
                                  name, spec.prior_type)),
        }
        Ok(())
    }

    /// Convert to the internal Prior struct
    pub fn to_prior(&self) -> Result<Prior> {
        Ok(Prior {
            emin: self.convert_prior_spec(&self.priors.emin, "emin")?,
            emax: self.convert_prior_spec(&self.priors.emax, "emax")?,
            ec50: self.convert_prior_spec(&self.priors.ec50, "ec50")?,
            hillslope: self.convert_prior_spec(&self.priors.hillslope, "hillslope")?,
        })
    }

    fn convert_prior_spec(&self, spec: &PriorSpec, name: &str) -> Result<PriorType> {
        match spec.prior_type.as_str() {
            "normal" => Ok(PriorType::Normal {
                mean: spec.mean,
                std: spec.std,
            }),
            "lognormal" => Ok(PriorType::LogNormal {
                mean: spec.mean,
                std: spec.std,
            }),
            "uniform" => {
                let min = spec.min.ok_or_else(|| anyhow!("Prior '{}': uniform distribution requires 'min'", name))?;
                let max = spec.max.ok_or_else(|| anyhow!("Prior '{}': uniform distribution requires 'max'", name))?;
                Ok(PriorType::Uniform { min, max })
            },
            _ => Err(anyhow!("Unsupported prior type: {}", spec.prior_type)),
        }
    }

    /// Get plotting bounds as a tuple, if all are specified
    pub fn get_plotting_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        match (
            self.plotting.bounds.x_min,
            self.plotting.bounds.x_max,
            self.plotting.bounds.y_min,
            self.plotting.bounds.y_max,
        ) {
            (Some(x_min), Some(x_max), Some(y_min), Some(y_max)) => Some((x_min, x_max, y_min, y_max)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        
        let prior = config.to_prior().unwrap();
        assert!(matches!(prior.emin, PriorType::Normal { .. }));
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn test_invalid_backend() {
        let mut config = Config::default();
        config.mcmc.backend = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_prior() {
        let mut config = Config::default();
        config.priors.emin.std = -1.0; // Invalid negative std
        assert!(config.validate().is_err());
    }
}
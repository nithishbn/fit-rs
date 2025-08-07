use crate::{DoseResponse, LL4Parameters, MCMCResult, Prior, PriorType};
use anyhow::{anyhow, Result};
use bridgestan::{compile_model, open_library, Model, StanLibrary};
use serde_json::json;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::OnceLock;

pub trait MCMCSampler {
    fn fit(&self, n_samples: usize, burnin: usize, chains: Option<usize>) -> Result<MCMCResult>;
    fn get_name(&self) -> &'static str;
}

// Global cache for compiled Stan model
static COMPILED_MODEL_CACHE: OnceLock<PathBuf> = OnceLock::new();
static BRIDGESTAN_PATH_CACHE: OnceLock<PathBuf> = OnceLock::new();

fn get_or_compile_stan_model() -> Result<PathBuf> {
    if let Some(cached_path) = COMPILED_MODEL_CACHE.get() {
        if cached_path.exists() {
            return Ok(cached_path.clone());
        }
    }
    
    // First, check if we have a build-time compiled model
    if let Some(build_time_path) = option_env!("COMPILED_STAN_MODEL_PATH") {
        let build_time_model = PathBuf::from(build_time_path);
        if build_time_model.exists() {
            println!("Using build-time compiled Stan model: {}", build_time_model.display());
            COMPILED_MODEL_CACHE.set(build_time_model.clone()).ok();
            return Ok(build_time_model);
        }
    }
    
    // Fall back to runtime compilation
    // Get or cache BridgeStan path
    let bridgestan_path = if let Some(cached_bs_path) = BRIDGESTAN_PATH_CACHE.get() {
        cached_bs_path.clone()
    } else {
        let bs_path = match bridgestan::download_bridgestan_src() {
            Ok(downloaded_path) => downloaded_path,
            Err(_) => {
                if let Ok(env_path) = std::env::var("BRIDGESTAN_PATH") {
                    PathBuf::from(env_path)
                } else {
                    return Err(anyhow!(
                        "BridgeStan not found. Please either:\n\
                        1. Use the 'download-bridgestan-src' feature (already enabled), or\n\
                        2. Set BRIDGESTAN_PATH environment variable, or\n\
                        3. Install BridgeStan manually, or\n\
                        4. Use --features build-time-compile for build-time compilation"
                    ));
                }
            }
        };
        
        // Cache the BridgeStan path
        BRIDGESTAN_PATH_CACHE.set(bs_path.clone()).ok();
        bs_path
    };
    
    let stan_file_path = std::env::current_dir()?.join("ec50_model.stan");
    
    // Check if we have a cached compiled model in target directory
    let target_dir = std::env::current_dir()?.join("target").join("stan_models");
    std::fs::create_dir_all(&target_dir)?;
    
    let model_name = stan_file_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let compiled_path = target_dir.join(format!("{}_model.so", model_name));
    
    // Check if compiled model exists and is newer than source
    let should_recompile = if compiled_path.exists() {
        let source_modified = std::fs::metadata(&stan_file_path)?.modified()?;
        let compiled_modified = std::fs::metadata(&compiled_path)?.modified()?;
        source_modified > compiled_modified
    } else {
        true
    };
    
    if should_recompile {
        println!("Compiling Stan model at runtime (consider using --features build-time-compile)...");
        let temp_compiled = compile_model(
            &bridgestan_path,
            &stan_file_path,
            &[],
            &[]
        ).map_err(|e| anyhow!("Failed to compile Stan model: {}", e))?;
        
        // Move compiled model to our cache directory
        std::fs::copy(&temp_compiled, &compiled_path)?;
        println!("Stan model compiled and cached at: {}", compiled_path.display());
    } else {
        println!("Using cached Stan model: {}", compiled_path.display());
    }
    
    // Cache the compiled model path
    COMPILED_MODEL_CACHE.set(compiled_path.clone()).ok();
    
    Ok(compiled_path)
}

pub struct StanSampler {
    data: Vec<DoseResponse>,
    prior: Prior,
    model_path: PathBuf,
}

impl StanSampler {
    pub fn new(data: Vec<DoseResponse>, prior: Prior) -> Result<Self> {
        let model_path = PathBuf::from("ec50_model.stan");
        if !model_path.exists() {
            return Err(anyhow!("Stan model file not found: {}", model_path.display()));
        }
        
        Ok(Self {
            data,
            prior,
            model_path,
        })
    }

    fn prepare_stan_data(&self) -> Result<String> {
        let log_conc: Vec<f64> = self.data.iter()
            .map(|d| d.concentration.log10())
            .collect();
        
        let response: Vec<f64> = self.data.iter()
            .map(|d| d.response)
            .collect();

        let (emin_mean, emin_sd) = match &self.prior.emin {
            PriorType::Normal { mean, std } => (*mean, *std),
            _ => return Err(anyhow!("Only Normal priors supported for Stan backend")),
        };

        let (emax_mean, emax_sd) = match &self.prior.emax {
            PriorType::Normal { mean, std } => (*mean, *std),
            _ => return Err(anyhow!("Only Normal priors supported for Stan backend")),
        };

        let (ec50_mean, ec50_sd) = match &self.prior.ec50 {
            PriorType::Normal { mean, std } => (*mean, *std),
            _ => return Err(anyhow!("Only Normal priors supported for Stan backend")),
        };

        let (hillslope_mean, hillslope_sd) = match &self.prior.hillslope {
            PriorType::Normal { mean, std } => (*mean, *std),
            _ => return Err(anyhow!("Only Normal priors supported for Stan backend")),
        };

        let data_json = json!({
            "N": self.data.len(),
            "log_conc": log_conc,
            "response": response,
            "emin_mean": emin_mean,
            "emin_sd": emin_sd,
            "emax_mean": emax_mean,
            "emax_sd": emax_sd,
            "ec50_mean": ec50_mean,
            "ec50_sd": ec50_sd,
            "hillslope_mean": hillslope_mean,
            "hillslope_sd": hillslope_sd,
        });

        Ok(data_json.to_string())
    }

    fn compile_stan_model(&self) -> Result<PathBuf> {
        // Use the cached compilation function
        get_or_compile_stan_model()
    }

    fn extract_samples_from_stan(&self, model: &Model<&StanLibrary>, n_samples: usize, chains: usize) -> Result<MCMCResult> {
        // This is a simplified approach - in practice, you would run Stan's sampler
        // For now, we'll use BridgeStan's interface to evaluate the model
        // and implement a basic sampling scheme
        
        let mut all_samples = Vec::new();
        let mut all_log_likelihoods = Vec::new();
        let mut total_accepted = 0;
        
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        for _chain in 0..chains {
            let mut chain_samples = Vec::new();
            let mut chain_log_liks = Vec::new();
            let mut chain_accepted = 0;
            
            // Initialize parameters from prior
            let mut current_params = vec![0.0; 5]; // emin, emax, ec50, hillslope, sigma
            current_params[0] = rng.gen_range(-5.0..5.0);   // emin
            current_params[1] = rng.gen_range(50.0..150.0); // emax  
            current_params[2] = rng.gen_range(-8.0..-4.0);  // ec50
            current_params[3] = rng.gen_range(0.5..2.0);    // hillslope
            current_params[4] = rng.gen_range(0.1..2.0);    // sigma
            
            let mut current_log_prob = model.log_density(&current_params, true, true)
                .map_err(|e| anyhow!("Failed to evaluate log density: {}", e))?;
            
            // Simple random walk Metropolis (Stan would use HMC)
            for _ in 0..n_samples {
                let mut proposed_params = current_params.clone();
                
                // Random walk proposal
                for i in 0..4 { // Don't propose for sigma
                    proposed_params[i] += rng.gen_range(-0.1..0.1);
                }
                proposed_params[4] = (proposed_params[4] + rng.gen_range(-0.05..0.05)).max(0.01f64);
                
                match model.log_density(&proposed_params, true, true) {
                    Ok(proposed_log_prob) => {
                        let log_ratio = proposed_log_prob - current_log_prob;
                        if rng.gen::<f64>().ln() < log_ratio {
                            current_params = proposed_params;
                            current_log_prob = proposed_log_prob;
                            chain_accepted += 1;
                        }
                    },
                    Err(_) => {
                        // Reject proposal if it's invalid
                    }
                }
                
                chain_samples.push(LL4Parameters {
                    emin: current_params[0],
                    emax: current_params[1],
                    ec50: current_params[2],
                    hillslope: current_params[3],
                });
                
                chain_log_liks.push(current_log_prob);
            }
            
            all_samples.extend(chain_samples);
            all_log_likelihoods.extend(chain_log_liks);
            total_accepted += chain_accepted;
        }
        
        let acceptance_rate = total_accepted as f64 / (n_samples * chains) as f64;
        
        Ok(MCMCResult {
            samples: all_samples,
            acceptance_rate,
            log_likelihood: all_log_likelihoods,
        })
    }
}

impl MCMCSampler for StanSampler {
    fn fit(&self, n_samples: usize, burnin: usize, chains: Option<usize>) -> Result<MCMCResult> {
        let chains = chains.unwrap_or(4);
        
        // Compile the Stan model
        let compiled_path = self.compile_stan_model()?;
        
        // Load the compiled model
        let lib = open_library(compiled_path)
            .map_err(|e| anyhow!("Failed to load compiled Stan model: {}", e))?;
        
        // Prepare data
        let data_json = self.prepare_stan_data()?;
        
        // Create model instance
        let data_cstring = CString::new(data_json)?;
        let model = Model::new(&lib, Some(&data_cstring), 12345) // seed
            .map_err(|e| anyhow!("Failed to create Stan model: {}", e))?;
        
        // Extract samples (this is simplified - real Stan would use NUTS/HMC)
        let mut result = self.extract_samples_from_stan(&model, n_samples + burnin, chains)?;
        
        // Remove burnin samples
        let burnin_total = burnin * chains;
        if result.samples.len() > burnin_total {
            result.samples.drain(0..burnin_total);
            result.log_likelihood.drain(0..burnin_total);
        }
        
        Ok(result)
    }
    
    fn get_name(&self) -> &'static str {
        "Stan (BridgeStan)"
    }
}

/// Force recompilation of the Stan model (useful for development)
pub fn force_recompile_stan_model() -> Result<PathBuf> {
    // Clear the cache
    if let Some(cached_path) = COMPILED_MODEL_CACHE.get() {
        if cached_path.exists() {
            std::fs::remove_file(cached_path)?;
        }
    }
    
    // Get fresh compilation
    get_or_compile_stan_model()
}

/// Precompile the Stan model (useful for CI/CD or deployment)
pub fn precompile_stan_model() -> Result<()> {
    let compiled_path = get_or_compile_stan_model()?;
    println!("Stan model precompiled successfully: {}", compiled_path.display());
    Ok(())
}
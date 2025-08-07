use fit_rs::{DoseResponse, MCMCSampler, Prior, PriorType, StanSampler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Stan backend implementation...");

    // Create test data
    let test_data = vec![
        DoseResponse { concentration: 1e-9, response: 10.0 },
        DoseResponse { concentration: 1e-8, response: 15.0 },
        DoseResponse { concentration: 1e-7, response: 25.0 },
        DoseResponse { concentration: 1e-6, response: 45.0 },
        DoseResponse { concentration: 1e-5, response: 70.0 },
        DoseResponse { concentration: 1e-4, response: 85.0 },
        DoseResponse { concentration: 1e-3, response: 90.0 },
    ];

    // Set up priors
    let priors = Prior {
        emin: PriorType::Normal { mean: 0.0, std: 10.0 },
        emax: PriorType::Normal { mean: 100.0, std: 20.0 },
        ec50: PriorType::Normal { mean: -6.0, std: 2.0 },
        hillslope: PriorType::Normal { mean: 1.0, std: 1.0 },
    };

    // Test Stan backend
    println!("Creating Stan sampler...");
    let stan_sampler = StanSampler::new(test_data, priors)?;
    
    println!("Running Stan fit (reduced samples for testing)...");
    let result = stan_sampler.fit(100, 50, Some(1))?;
    
    println!("Stan backend: {}", stan_sampler.get_name());
    println!("Number of samples: {}", result.samples.len());
    println!("Acceptance rate: {:.1}%", result.acceptance_rate * 100.0);
    
    if !result.samples.is_empty() {
        let last_sample = &result.samples[result.samples.len() - 1];
        println!("Final sample - EC50: {:.3}, Hill slope: {:.3}", 
                 last_sample.ec50, last_sample.hillslope);
    }
    
    println!("Stan backend test completed successfully!");
    
    Ok(())
}
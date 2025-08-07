// Build script to compile Stan model at build time
use std::env;
use std::path::PathBuf;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=ec50_model.stan");
    println!("cargo:rerun-if-env-changed=BRIDGESTAN_PATH");
    
    // Check if build-time compilation is enabled
    let build_time_compile = cfg!(feature = "build-time-compile") 
        || env::var("BUILD_TIME_COMPILE_STAN").is_ok();
    
    if !build_time_compile {
        println!("cargo:warning=Stan model will be compiled at runtime. Use --features build-time-compile for build-time compilation.");
        return Ok(());
    }
    
    println!("cargo:warning=Compiling Stan model at build time...");
    
    // Get output directory from Cargo
    let out_dir = env::var("OUT_DIR")?;
    let out_path = PathBuf::from(&out_dir);
    
    // Create stan_models directory in OUT_DIR
    let stan_models_dir = out_path.join("stan_models");
    fs::create_dir_all(&stan_models_dir)?;
    
    // Define model paths
    let stan_model_src = PathBuf::from("ec50_model.stan");
    let compiled_model_path = stan_models_dir.join("ec50_model.so");
    
    if !stan_model_src.exists() {
        println!("cargo:warning=Stan model file not found: {}", stan_model_src.display());
        return Ok(());
    }
    
    // Check if we need to compile (model doesn't exist or source is newer)
    let should_compile = if compiled_model_path.exists() {
        let source_time = fs::metadata(&stan_model_src)?.modified()?;
        let compiled_time = fs::metadata(&compiled_model_path)?.modified()?;
        source_time > compiled_time
    } else {
        true
    };
    
    if should_compile {
        // Try to compile the Stan model
        match compile_stan_model(&stan_model_src, &compiled_model_path) {
            Ok(_) => {
                println!("cargo:warning=Stan model compiled successfully at build time: {}", compiled_model_path.display());
                
                // Set environment variable so runtime knows where to find the compiled model
                println!("cargo:rustc-env=COMPILED_STAN_MODEL_PATH={}", compiled_model_path.display());
            }
            Err(e) => {
                println!("cargo:warning=Failed to compile Stan model at build time: {}. Will fall back to runtime compilation.", e);
            }
        }
    } else {
        println!("cargo:warning=Using cached Stan model from build time: {}", compiled_model_path.display());
        println!("cargo:rustc-env=COMPILED_STAN_MODEL_PATH={}", compiled_model_path.display());
    }
    
    Ok(())
}

fn compile_stan_model(source: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // We need to use BridgeStan to compile, but we can't depend on our own crate here
    // So we'll use the bridgestan crate directly
    
    // Try to get BridgeStan path
    let bridgestan_path = if let Ok(path) = env::var("BRIDGESTAN_PATH") {
        PathBuf::from(path)
    } else {
        // Try to download BridgeStan
        match try_download_bridgestan() {
            Some(path) => path,
            None => return Err("Could not find or download BridgeStan".into()),
        }
    };
    
    println!("cargo:warning=Using BridgeStan at: {}", bridgestan_path.display());
    
    // Compile using bridgestan crate
    let temp_compiled = bridgestan::compile_model(
        &bridgestan_path,
        source,
        &[],
        &[]
    )?;
    
    // Copy to our output location
    fs::copy(&temp_compiled, output)?;
    
    Ok(())
}

fn try_download_bridgestan() -> Option<PathBuf> {
    match bridgestan::download_bridgestan_src() {
        Ok(path) => Some(path),
        Err(_) => None,
    }
}
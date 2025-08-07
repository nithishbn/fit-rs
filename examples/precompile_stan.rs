use fit_rs::precompile_stan_model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Precompiling Stan model...");
    precompile_stan_model()?;
    println!("Stan model precompilation complete!");
    Ok(())
}
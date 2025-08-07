data {
  int<lower=1> N;                    // Number of data points
  vector[N] log_conc;                // Log10 concentrations
  vector[N] response;                // Response values
  
  // Prior specifications
  real emin_mean;
  real emin_sd;
  real emax_mean;
  real emax_sd;
  real ec50_mean;
  real ec50_sd;
  real hillslope_mean;
  real hillslope_sd;
}

parameters {
  real emin;                         // Lower asymptote
  real emax;                         // Upper asymptote  
  real ec50;                         // EC50 (log10 scale)
  real hillslope;                    // Hill slope
  real<lower=0> sigma;               // Error standard deviation
}

model {
  // Priors
  emin ~ normal(emin_mean, emin_sd);
  emax ~ normal(emax_mean, emax_sd);
  ec50 ~ normal(ec50_mean, ec50_sd);
  hillslope ~ normal(hillslope_mean, hillslope_sd);
  sigma ~ exponential(0.1);          // Weakly informative prior for error
  
  // 4-parameter logistic model
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = emin + (emax - emin) / (1 + pow(10, (ec50 - log_conc[i]) * hillslope));
  }
  
  // Likelihood
  response ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive samples for model checking
  vector[N] response_pred;
  vector[N] log_lik;
  
  for (i in 1:N) {
    real mu_i = emin + (emax - emin) / (1 + pow(10, (ec50 - log_conc[i]) * hillslope));
    response_pred[i] = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(response[i] | mu_i, sigma);
  }
}
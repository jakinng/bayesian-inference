
data {
  int<lower=1> K;                    // number of polls
  int<lower=1> J;                    // number of campaign days
  int<lower=1> S;                    // number of states
  array[K] int<lower=1, upper=S> state; // state for each poll
  array[K] int<lower=1, upper=J> day;   // day for each poll
  array[K] int<lower=0> n;               // sample size of poll
  array[K] int<lower=0> y;               // number of dem supporters
  vector[S] h;                        // historical forecasts (h_i)
  vector[S] tau;                      // prior precision for each state (tau_i)
}
parameters {
  matrix[S, J] beta;                 // state-level support over time
  vector[J - 1] delta_raw;           // free parameters for national swing (J-1 values)
  real<lower=0, upper=1> sigma_beta;          // variance of beta
  real<lower=0, upper=1> sigma_delta;         // variance of delta
}
transformed parameters {
  vector[J] delta;
  for (j in 1:(J - 1))
    delta[j] = delta_raw[j];
  delta[J] = 0;  // anchor on Election Day
}
model {
  // Priors on beta and delta (reverse random walk from election day J)
  for (s in 1:S)
    beta[s, J] ~ normal(logit(h[s]), inv(sqrt(tau[s])));

  for (j in 1:(J - 1)) {
    for (s in 1:S)
      beta[s, j] ~ normal(beta[s, j + 1], sigma_beta);
    delta_raw[j] ~ normal(delta[j + 1], sigma_delta);
  }

  // Likelihood: poll observations
  for (k in 1:K) {
    real logit_pi = beta[state[k], day[k]] + delta[day[k]];
    y[k] ~ binomial_logit(n[k], logit_pi);
  }

  // Hyperpriors
  sigma_beta ~ uniform(0, 1);
  sigma_delta ~ uniform(0, 1);
}
generated quantities {
  matrix[S, J] pi;
  for (s in 1:S)
    for (j in 1:J)
      pi[s, j] = inv_logit(beta[s, j] + delta[j]);
}

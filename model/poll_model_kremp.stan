
data {
  int<lower=1> K;
  int<lower=1> T;
  int<lower=1> S;
  int<lower=1> P; //Number of polls
  array[K] int<lower=1, upper=S> state;
  array[K] int<lower=1, upper=T> day;
  array[K] int<lower=1, upper=P> pollster;
  array[K] int<lower=0> n;
  array[K] int<lower=0> y;
  vector[S] h;
  matrix[S, S] Sigma_tau;
  matrix[S, S] Sigma_e;
  real<lower=0> sigma_u_upper_bound;
}
parameters {
  vector[T - 1] mu_a_raw;
  matrix[T, S] mu_b;
  vector[S] e;
  vector[K] u;
  vector[P] mu_c;
  real<lower=1e-6> sigma_a;
  real<lower=1e-6> sigma_b;
  real<lower=1e-6> sigma_c;
  real<lower=0> sigma_u;

}
transformed parameters {
  vector[T] mu_a;
  vector[S] mu_e;
  matrix[S, S] Sigma_mu_b;
  mu_a[T] = 0;
  for (s in 1:S)
    mu_e[s] = 0;
  for (t in 1:(T - 1))
    mu_a[t] = mu_a_raw[t];
}

model {
  for (p in 1:P){
    mu_c[p] ~ normal(0, sigma_c);
  }
  e ~ multi_normal(mu_e, Sigma_e);

  // Priors on beta (reverse random walk from election day J), using multinormal distribution
  mu_b[T] ~ multi_normal(logit(h), Sigma_tau);

  for (t in 1:(T - 1))
    mu_a[t] ~ normal(mu_a[t + 1], sigma_a);
  for (s in 1:S)
    for (t in 1:(T - 1))
      mu_b[t, s] ~ normal(mu_b[t + 1, s], sigma_b);
    
  // measurement error
  for (k in 1:K)
    u[k] ~ normal(0, sigma_u);

  // Likelihood: poll observations
  for (k in 1:K) {
    // add up components
    real logit_pi = mu_a[day[k]] + mu_b[day[k], state[k]] + mu_c[pollster[k]] + e[state[k]] + u[k];
    // Define the vote share as the logit of the probability of Dem winning
    y[k] ~ binomial_logit(n[k], logit_pi);
  }

  // Hyper-priors
  sigma_a ~ normal(0,0.5);   // half-normal
  sigma_b ~ normal(0,0.5);   // half-normal
  sigma_c ~ normal(0,0.5);   // half-normal
  sigma_u ~ normal(0,sigma_u_upper_bound);

}

generated quantities {
  matrix[S, T] pi;
  for (s in 1:S)
    for (t in 1:T)
      pi[s, t] = inv_logit(mu_b[t, s]+mu_a[t]);
}

data{
 int <lower=0> N;
 int <lower=0> J;
 int<lower=1,upper=J> county[N];
 vector[N] x;
 vector[N] y;
}

parameters{
real mu_a;
real<lower=0,upper=100> sigma_a;
real<lower=0,upper=100> sigma_y;
real beta;
vector[J] alpha;
}

transformed parameters{
vector[N] y_mean;

for (i in 1:N) {
    y_mean[i] = x[i]*beta + alpha[county[i]];    
}

}
model{
sigma_a ~ gamma(1,1);
sigma_y ~gamma(1,1);
beta ~ normal(0,1);
alpha ~ normal(mu_a, sigma_a);
y ~ normal(y_mean, sigma_y);

}
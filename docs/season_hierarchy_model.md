# Bayesian Hierarchical Model: Season-Level Structure

## Model Overview

We aim to estimate the latent (true) quality of each episode in a TV series, using observed ratings and vote counts. Episodes are grouped by season, allowing episodes within the same season to share information and stabilize estimates, especially for episodes with few votes.

## Variables

### Observed Variables
- $y_i$: Observed average rating for episode $i$ (bounded, continuous, e.g., from TMDB)
- $n_i$: Number of votes for episode $i$ (proxy for confidence)
- $s_i$: Season index for episode $i$

### Latent Variable Being Modeled
- $\theta_i$: Latent “true” quality of episode $i$ (unobserved, continuous, bounded)

## Likelihood

The observed rating for each episode is modeled as a noisy measurement of its latent quality, using a truncated normal distribution to respect rating bounds:
$$
y_i \sim \text{TruncatedNormal}(\theta_i, \sigma^2_i; a, b)
$$
where $a$ and $b$ are the lower and upper bounds of the rating scale ( $a=1$, $b=10$), and $\sigma^2_i$ is the observation noise, typically set as $\sigma^2 / n_i$ (more votes $\Rightarrow$ less noise).

**What is $\theta_i$?**

In this likelihood, $\theta_i$ represents the latent (unobserved) “true” quality of episode $i$. It is the underlying quality or appeal of episode $i$ that we want to estimate. We do not observe $\theta_i$ directly; instead, we see the average rating ($y_i$) given by viewers, which is a noisy reflection of $\theta_i$. The model assumes that the observed rating for each episode is generated from its true quality, plus some random noise due to limited votes and individual differences.

## Hierarchy (Season-Level)

Episodes are grouped by season. Each episode’s latent quality is drawn from a truncated normal distribution specific to its season:
$$
\theta_i \sim \text{TruncatedNormal}(\mu_{s_i}, \tau_{s_i}; a, b)
$$
where:
- $\mu_{s_i}$: Mean quality for season $s_i$
- $\tau_{s_i}$: Variance of episode qualities within season $s_i$
- $a, b$: Lower and upper bounds of the rating scale

## Priors and Hyperpriors

- Season means:
  $$
  \mu_s \sim \text{TruncatedNormal}(\mu_0, \sigma^2_\mu; a, b)
  $$
- Season variances:
  $$
  \tau_s \sim \text{Half-Cauchy}(0, \tau_0)
  $$
- Global mean:
  $$
  \mu_0 \sim \text{TruncatedNormal}(m, v; a, b)
  $$
- Hyperpriors for variances ($\sigma^2_\mu$, $\tau_0$) are set to weakly informative values to encourage shrinkage and avoid overfitting.

## Posterior

The posterior distribution combines the likelihood and all priors:
$$
P(\theta, \mu, \tau, \mu_0 \mid y, n, s) \propto \prod_i P(y_i \mid \theta_i, n_i) P(\theta_i \mid \mu_{s_i}, \tau_{s_i}) P(\mu_{s_i} \mid \mu_0, \sigma^2_\mu) P(\tau_{s_i} \mid \tau_0) P(\mu_0)
$$
This yields the joint posterior for all episode qualities, season means, season variances, and the global mean, given the observed ratings, vote counts, and season assignments.


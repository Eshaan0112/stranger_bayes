import pymc as pm
import numpy as np
import pandas as pd

class SeasonHierarchicalModel:
    def __init__(self, df, rating_col='vote_average', season_col='season_number', n_col='vote_count', lower=-0.5, upper=10.5):
        """
        Initialize the SeasonHierarchicalModel.

        Parameters:
        - df: pandas DataFrame containing episode data with columns for ratings, season, and vote count.
        - rating_col: Name of the column with observed ratings (default: 'vote_average').
        - season_col: Name of the column with season indices (default: 'season_number').
        - n_col: Name of the column with vote counts (default: 'vote_count').
        - lower, upper: Lower and upper bounds for ratings (default: -0.5, 10.5).
        """
        self.df = df.copy()
        self.rating_col = rating_col
        self.n_col = n_col

        # Clean missing values in observed ratings: set to 1 if missing or 0
        self.df[self.rating_col] = self.df[self.rating_col].fillna(1)
        # Ensure no zero or negative vote counts for model stability
        self.df[self.n_col] = self.df[self.n_col].clip(lower=1)

        self.season_col = season_col
        self.lower = lower
        self.upper = upper
        self.seasons = self.df[season_col].unique()
        self.season_idx = pd.Categorical(self.df[season_col], categories=self.seasons).codes
        self.model = None
        self.trace = None

    def build_model(self):
        """
        Build the PyMC hierarchical Bayesian model for episode ratings.
        Defines priors, hyperpriors, season-level parameters, latent episode qualities,
        and the likelihood using truncated normal distributions to respect rating bounds.
        """
        with pm.Model() as model:
            # Hyperpriors (global priors for group-level parameters) 
            mu_0 = pm.Uniform('mu_0', lower=self.lower, upper=self.upper)  # Global mean for season means (prior)
            sigma_mu = pm.HalfCauchy('sigma_mu', beta=2)  # Prior for std dev of season means
            tau_0 = pm.HalfCauchy('tau_0', beta=2)  # Prior for std dev of season-level variances

            # Season-level parameters (hierarchical group parameters)
            mu_s = pm.TruncatedNormal('mu_s', mu=mu_0, sigma=sigma_mu, lower=self.lower, upper=self.upper, shape=len(self.seasons))  # Mean quality for each season
            tau_s = pm.HalfCauchy('tau_s', beta=tau_0, shape=len(self.seasons))  # Std dev for each season

            # Episode-level latent variables 
            theta = pm.TruncatedNormal('theta', mu=mu_s[self.season_idx], sigma=tau_s[self.season_idx], lower=self.lower, upper=self.upper, shape=len(self.df))  # Latent true quality for each episode

            # Observation model (likelihood) 
            sigma_obs = 1.0 / np.sqrt(self.df[self.n_col].values)  # Observation noise, scaled by vote count
            y_obs = pm.TruncatedNormal('y_obs', mu=theta, sigma=sigma_obs, lower=self.lower, upper=self.upper, observed=self.df[self.rating_col].values)  # Observed ratings as noisy measurements of latent quality

            self.model = model
        return model

    def fit(self, draws=2000, tune=1000, target_accept=0.9, random_seed=42, chains=4):
        """
        Fit the hierarchical Bayesian model using MCMC sampling.

        Parameters:
        - draws: Number of posterior samples to draw (default: 2000).
        - tune: Number of tuning (burn-in) steps (default: 1000).
        - target_accept: Target acceptance probability for NUTS sampler (default: 0.9).
        - random_seed: Random seed for reproducibility (default: 42).
        - chains: Number of MCMC chains to run in parallel (default: 4).

        Returns:
        - trace: InferenceData object containing posterior samples.
        """
        if self.model is None:
            self.build_model()
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, random_seed=random_seed, chains=chains, return_inferencedata=True)
        return self.trace

    def summary(self):
        """
        Return a summary of the posterior samples using ArviZ.
        Returns a DataFrame with posterior means, credible intervals, and diagnostics.
        """
        if self.trace is not None:
            import arviz as az
            return az.summary(self.trace)
        else:
            print("Model has not been fit yet.")
            return None

    def infer_episode_quality(self, season, episode_idx=None):
        """
        Return the posterior samples and summary for the latent quality (theta) of a specific episode.

        Parameters:
        - season: The season number of the episode to infer.
        - episode_idx: The index (0-based, within the DataFrame) of the episode in that season. If None, returns all episodes in the season.

        Returns:
        - samples: Posterior samples for theta for the specified episode(s).
        - summary: Posterior mean, median, and credible interval for the episode(s).
        """
        if self.trace is None:
            print("Model has not been fit yet.")
            return None
        # Find indices for the specified season
        season_mask = (self.df[self.season_col] == season)
        if episode_idx is not None:
            # Get the DataFrame index for the requested episode in the season
            season_indices = self.df[season_mask].index.tolist()
            if episode_idx >= len(season_indices):
                print("Invalid episode index for this season.")
                return None
            idx = season_indices[episode_idx]
            theta_samples = self.trace.posterior['theta'].values[..., idx].flatten()
            summary = {
                'mean': np.mean(theta_samples),
                'median': np.median(theta_samples),
                'hdi_3%': np.percentile(theta_samples, 3),
                'hdi_97%': np.percentile(theta_samples, 97)
            }
            return theta_samples, summary
        else:
            # Return all episodes in the season
            season_indices = self.df[season_mask].index.tolist()
            theta_samples = [self.trace.posterior['theta'].values[..., idx].flatten() for idx in season_indices]
            summaries = []
            for samples in theta_samples:
                summaries.append({
                    'mean': np.mean(samples),
                    'median': np.median(samples),
                    'hdi_3%': np.percentile(samples, 3),
                    'hdi_97%': np.percentile(samples, 97)
                })
            return theta_samples, summaries

    def add_episode(self, season, episode_number, **kwargs):
        """
        Add a new episode (an unaired or unrated episode) to the DataFrame for inference.

        Parameters:
        - season: The season number for the new episode.
        - episode_number: The episode number (or any unique identifier).
        - kwargs: Additional columns/values (vote_average, vote_count). If not provided, vote_average will be set to NaN and vote_count to 1.

        The new episode will be included in the next model fit and can be used for inference.
        """
        new_row = {self.season_col: season, 'episode_number': episode_number}
        # Set default values for required columns if not provided
        new_row[self.rating_col] = kwargs.get(self.rating_col, np.nan)
        new_row[self.n_col] = kwargs.get(self.n_col, 1)
        # Add any other columns provided
        for k, v in kwargs.items():
            if k not in new_row:
                new_row[k] = v
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        # Update season indices
        self.seasons = self.df[self.season_col].unique()
        self.season_idx = pd.Categorical(self.df[self.season_col], categories=self.seasons).codes



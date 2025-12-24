## Bayesian Hierarchical Model for predicting TV show popularity

The aim of this project is to build a Bayesian hierarchical model to predict TV episode ratings. The model uses episode metadata and ratings fetched from the TMDB API, allowing for uncertainty quantification and probabilistic predictions. This approach enables more robust analysis of TV show ratings, including credible intervals and season/episode effects. **For theoretical decisions, please refer to docs/**

### Recommended Environment


### Install Miniconda and PyMC
Make sure you have Python 3.11 installed. Follow these steps to set up Miniconda and install PyMC in your environment:

#### 1. Download and Install Miniconda
- Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system (Windows, macOS, or Linux).
- Run the installer and follow the prompts to complete the installation.
- After installation, open a new terminal (Anaconda Prompt or your preferred shell).


#### 2. Install PyMC and Dependencies
```
conda install -c conda-forge pymc arviz numpy pandas 
matplotlib
```

```
conda activate *environment_name*
```
- This project uses a yaml file to maintain dependencies: environment.yml
	- These dependencies can be updated as:
	```
	conda env update -f environment.yml --prune
	```
- This will install PyMC and common scientific libraries from the conda-forge channel.

#### 4. Verify Installation
```
python -c "import pymc; print(pymc.__version__)"
```

### Fetch Data from TMDB API

After setting up your environment and installing dependencies, fetch the episode-level data from the TMDB API:

1. Ensure your TMDB API key is set in a `.env` file as required by the script.
2. Run the data fetch script:
   ```bash
    python data_collection/fetch_tmdb_episodes.py
   ```
   - The output CSV will be saved in the `data/` folder as `stranger_things_episodes.csv`.

**Notes:**
- If you encounter SSL errors, the script disables certificate verification for API requests (for debugging purposes).
- The script collects episode-level metadata, ratings, and vote counts for Stranger Things.


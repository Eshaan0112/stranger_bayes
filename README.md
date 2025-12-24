## Bayesian Hierarchical Model for Predicting TV Show Popularity

This project builds a Bayesian hierarchical model to predict TV episode ratings using metadata and ratings from the TMDB API. It provides uncertainty quantification, credible intervals, and season/episode effects. For theoretical details, see the `docs/` folder.

---

### Usage

#### 1. Local Development (CLI)
To run the model locally and experiment with code, use the main script:
```bash
python src/main.py
```
- This runs the main function for local development, testing, and debugging.

#### 2. Web UI (FastAPI)
To use the interactive web interface, run:
```bash
uvicorn src.api:app --reload
```
- Access the UI at [http://127.0.0.1:8000/predict_quality/](http://127.0.0.1:8000/predict_quality/)
- Enter season and episode number in the form to get predictions.

##### Kill Process on Port 8000 (WSL/Linux)
If you get "address already in use" errors:
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

---

### Environment Setup
Make sure you have Python 3.11 installed.

#### Install Miniconda and PyMC
1. Download Miniconda: [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html)
2. Install and open a new terminal.
3. Install dependencies:
```bash
conda install -c conda-forge pymc arviz numpy pandas matplotlib
```
4. Activate your environment:
```bash
conda activate <environment_name>
```
5. Update dependencies from yaml:
```bash
conda env update -f environment.yml --prune
```
6. Verify installation:
```bash
python -c "import pymc; print(pymc.__version__)"
```

---

### TMDB API - if needed
1. Set your TMDB API key in a `.env` file.
2. Run:
```bash
python data_collection/fetch_tmdb_episodes.py
```
- Output CSV: `data/stranger_things_episodes.csv`

**Notes:**
- SSL errors: The script disables certificate verification for debugging.
- Data includes episode-level metadata, ratings, and vote counts for Stranger Things.

---

### Project Structure
- `src/main.py`: Main script for local dev/testing
- `src/api.py`: FastAPI app for web UI
- `data/`: Episode-level data
- `docs/`: Theory and documentation



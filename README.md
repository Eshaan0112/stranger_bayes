
## Project Aim

The aim of this project is to build a Bayesian hierarchical model to predict TV episode ratings. The model uses episode metadata and ratings fetched from the TMDB API, allowing for uncertainty quantification and probabilistic predictions. This approach enables more robust analysis of TV show ratings, including credible intervals and season/episode effects. **For theoretical decisions, please refer to docs/**



### How to Run the Data Fetch Script

1. Make sure you have Python 3.11 installed.
2. Install Poetry if you don't have it:
	```bash
	pip install poetry
	```
3. Install dependencies and set up the environment:
	```bash
	poetry install
	```
4. Run the data fetch script:
	```bash
	poetry run python data_collection/fetch_tmdb_episodes.py
	```
	The output CSV will be saved in the `data/` folder.

### Notes
- Ensure your TMDB API key is set in the .env as required by the script.
- If you encounter SSL errors, the script disables certificate verification for API requests (for debugging purposes).

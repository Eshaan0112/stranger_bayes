import pandas as pd
from data_collection.fetch_tmdb_episodes import fetch_all_episodes
from src.season_hierarchical_model import SeasonHierarchicalModel

def main():
    
    # Load TV show data
    df = fetch_all_episodes("Stranger Things")

    # Clean unreleased episodes: set vote_average to NaN where vote_count == 0
    df.loc[df['vote_count'] == 0, 'vote_average'] = float('nan')

    # Initialize the model
    model = SeasonHierarchicalModel(df)

    # Build and fit the model
    print("Fitting the model. This may take a while...")
    model.fit(draws=1000, tune=500)

    # Infer the latent quality for Season 5, Episode 5 (first episode in season 5, index 0)
    samples, summary = model.infer_episode_quality(season=5, episode_idx=5)
    print('Posterior summary for S5E5:', summary)

    # To infer all episodes in a season:
    # all_samples, all_summaries = model.infer_episode_quality(season=5)
    # print('All S5 episode summaries:', all_summaries)
    
if __name__ == "__main__":
    main()
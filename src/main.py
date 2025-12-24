import pandas as pd
from data_collection.fetch_tmdb_episodes import fetch_all_episodes
from src.season_hierarchical_model import SeasonHierarchicalModel

def main():
    
    # Load TV show data
    df = fetch_all_episodes("Stranger Things")

    # Initialize the model
    model = SeasonHierarchicalModel(df)

    # Add a new episode for inference 
    model.add_episode(season=5, episode_number=5)

    # Build and fit the model
    print("Fitting the model. This may take a while...")
    model.fit(draws=1000, tune=500)

    # Infer the latent quality for Season 5, Episode 5 (first episode in season 5, index 0)
    samples, summary = model.infer_episode_quality(season=5, episode_idx=0)
    print('Posterior summary for S5E5:', summary)

    # To infer all episodes in a season:
    # all_samples, all_summaries = model.infer_episode_quality(season=5)
    # print('All S5 episode summaries:', all_summaries)
    
if __name__ == "__main__":
    main()
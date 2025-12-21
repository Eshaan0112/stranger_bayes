"""
Script to fetch episode metadata and ratings from the TMDB API.
"""
import os
import requests
import pandas as pd
from typing import List, Dict
from utils.config import Config

TMDB_API_KEY = Config.TMDB_API_KEY
TMDB_API_URL = 'https://api.themoviedb.org/3'


def get_show_id(show_name: str) -> int:
    """Search for a TV show and return its TMDB ID."""
    url = f"{TMDB_API_URL}/search/tv"
    params = {"api_key": TMDB_API_KEY, "query": show_name}
    resp = requests.get(url, params=params, verify=False)
    resp.raise_for_status()
    results = resp.json().get('results', [])
    if not results:
        raise ValueError(f"Show '{show_name}' not found.")
    return results[0]['id']


def get_seasons(show_id: int) -> List[int]:
    """Get all season numbers for a show."""
    url = f"{TMDB_API_URL}/tv/{show_id}"
    params = {"api_key": TMDB_API_KEY}
    resp = requests.get(url, params=params, verify=False)
    resp.raise_for_status()
    data = resp.json()
    return [season['season_number'] for season in data['seasons'] if season['season_number'] > 0]


def get_episodes(show_id: int, season_number: int) -> List[Dict]:
    """Get all episodes for a given season."""
    url = f"{TMDB_API_URL}/tv/{show_id}/season/{season_number}"
    params = {"api_key": TMDB_API_KEY}
    resp = requests.get(url, params=params, verify=False)
    resp.raise_for_status()
    data = resp.json()
    return data['episodes']


def fetch_all_episodes(show_name: str) -> pd.DataFrame:
    show_id = get_show_id(show_name)
    seasons = get_seasons(show_id)
    all_episodes = []
    for season in seasons:
        episodes = get_episodes(show_id, season)
        for ep in episodes:
            all_episodes.append({
                'show_id': show_id,
                'show_name': show_name,
                'season_number': season,
                'episode_number': ep.get('episode_number'),
                'title': ep.get('name'),
                'overview': ep.get('overview'),
                'air_date': ep.get('air_date'),
                'vote_count': ep.get('vote_count'),
                'vote_average': ep.get('vote_average'),
                'runtime': ep.get('runtime'),
            })
    return pd.DataFrame(all_episodes)


def main():
    
    # Fetch episode data for "Stranger Things"
    show_name = "Stranger Things"
    df = fetch_all_episodes(show_name)
    
    # Store data
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{show_name.replace(' ', '_').lower()}_episodes.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved episode data to {out_path}")


if __name__ == "__main__":
    main()

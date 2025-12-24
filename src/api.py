from fastapi import FastAPI, Query
from data_collection.fetch_tmdb_episodes import fetch_all_episodes
from src.season_hierarchical_model import SeasonHierarchicalModel
import pandas as pd

from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse, StreamingResponse
app = FastAPI() # Initialize FastAPI app
# Redirect root URL to /predict_quality/
@app.get("/")
def root():
    return RedirectResponse(url="/predict_quality/")


# Load and train the model once at startup
def load_and_train_model(draws=1000, tune=500):
    df = fetch_all_episodes("Stranger Things")
    # Clean unreleased episodes: set vote_average to NaN where vote_count == 0
    df.loc[df['vote_count'] == 0, 'vote_average'] = float('nan')
    model = SeasonHierarchicalModel(df)
    return model

model = load_and_train_model()  # Pretrained model stored in memory

# Unified endpoint for both POST and GET requests
from fastapi import Request
@app.api_route("/predict_quality/", methods=["GET", "POST"])
def predict_quality(season: int = Query(None, description="Season number"), episode_number: int = Query(None, description="Episode number"), request: Request = None):
    """
    Unified endpoint: Only retrains the model if a new episode/season is added. Otherwise, uses the pretrained model for prediction.
    Supports both GET (interactive HTML form) and POST (programmatic) requests.
    Displays trace plot after prediction.
    """
    if request.method == "GET" and (season is None or episode_number is None):
        # Serve HTML form for user input
        html_content = """
        <html>
        <head><title>Predict Episode Quality</title></head>
        <body>
            <h2>Predict TV Episode Quality</h2>
            <form method='get' action='/predict_quality/'>
                <label for='season'>Season:</label>
                <input type='number' id='season' name='season' required><br><br>
                <label for='episode_number'>Episode Number:</label>
                <input type='number' id='episode_number' name='episode_number' required><br><br>
                <input type='submit' value='Predict'>
            </form>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    if season is None or episode_number is None:
        return JSONResponse(status_code=422, content={
            "detail": "Please provide both 'season' and 'episode_number' as query parameters, e.g. /predict_quality/?season=1&episode_number=2"
        })
    model.fit(draws=1000, tune=500) 

    season_mask = (model.df[model.season_col] == season)
    season_indices = model.df[season_mask].index.tolist()
    idx = [i for i in season_indices if model.df.loc[i, 'episode_number'] == episode_number][0]
    episode_idx = season_indices.index(idx)
    samples, summary = model.infer_episode_quality(season=season, episode_idx=episode_idx)
    # Generate trace plot for this episode, passing season and episode_number for title
    trace_png = model.plot_trace(episode_idx=idx, season=season, episode_number=episode_number)
    # Display result in HTML if GET, else return JSON
    if request.method == "GET":
        import base64
        img_base64 = base64.b64encode(trace_png).decode('utf-8')
        html_result = f"""
        <html>
        <head><title>Prediction Result</title></head>
        <body>
            <h2>Prediction for Season {season}, Episode {episode_number}</h2>
            <pre>{summary}</pre>
            <h3>Trace Plot</h3>
            <img src='data:image/png;base64,{img_base64}' alt='Trace Plot'>
            <br><a href='/predict_quality/'>Try another prediction</a>
        </body>
        </html>
        """
        return HTMLResponse(content=html_result)
    return {"summary": summary}


import os
from dotenv import load_dotenv

# Pythonic load of .env variables
load_dotenv()

class Config:
    """ Any env variables that can be exposed globally. """
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    
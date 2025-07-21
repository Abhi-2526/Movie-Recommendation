import ast
import os
import json
import time
import urllib.request
import zipfile
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from security import safe_requests

load_dotenv()

def download_movielens_dataset():
    """Download and extract the MovieLens dataset"""
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = "../../data/raw/ml-25m.zip"
    os.makedirs("../../data/raw", exist_ok=True)

    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("../../data/raw")

    print("Done!")


def download_tmdb_data(movie_ids, api_key):
    """Download additional movie data from TMDB API"""
    os.makedirs("../../data/raw/tmdb", exist_ok=True)
    base_url = "https://api.themoviedb.org/3/movie/"

    for movie_id in tqdm(movie_ids, desc="Downloading TMDB data"):
        output_file = f"../../data/raw/tmdb/{movie_id}.json"
        if os.path.exists(output_file):
            continue

        url = f"{base_url}{movie_id}?api_key={api_key}&append_to_response=credits,keywords,similar"
        response = safe_requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'w') as f:
                json.dump(response.json(), f)
        else:
            print(f"Failed to download data for movie {movie_id}: {response.status_code}")
        time.sleep(0.25)


def process_movielens_data():
    # Load the raw movies CSV
    movies_df = pd.read_csv("../../data/raw/ml-25m/movies.csv")
    if 'genres' in movies_df.columns:
        movies_df['genres'] = movies_df['genres'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
        )

    # Load ratings and calculate average rating for each movie.
    ratings_df = pd.read_csv("../../data/raw/ml-25m/ratings.csv")
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']

    # Merge the average rating with the movies DataFrame.
    movies_df = pd.merge(movies_df, avg_ratings, on='movieId', how='left')

    # Ensure the processed directory exists and save the processed CSV
    os.makedirs("../../data/processed", exist_ok=True)
    movies_df.to_csv("../../data/processed/movies_processed.csv", index=False)

    return movies_df


def map_movielens_to_tmdb():
    """Map MovieLens IDs to TMDB IDs"""
    links_df = pd.read_csv("../../data/raw/ml-25m/links.csv")
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    links_df.to_csv("../../data/processed/movielens_tmdb_mapping.csv", index=False)
    return links_df['tmdbId'].tolist()


def main():
    # Step 1: Download and extract the MovieLens dataset.
    print("Starting MovieLens dataset download...")
    download_movielens_dataset()

    # Step 2: Process the MovieLens data.
    print("Processing MovieLens data...")
    movies_df = process_movielens_data()
    print(f"Processed {len(movies_df)} movies and saved to 'data/processed/movies_processed.csv'.")

    # Step 3: Map MovieLens IDs to TMDB IDs.
    print("Mapping MovieLens IDs to TMDB IDs...")
    movie_ids = map_movielens_to_tmdb()
    print(
        f"Found {len(movie_ids)} valid TMDB IDs in MovieLens mapping. Mapping saved to 'data/processed/movielens_tmdb_mapping.csv'.")

    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if tmdb_api_key:
        subset_ids = movie_ids[57291:]
        print("Downloading TMDB data for a subset of movies...")
        download_tmdb_data(subset_ids, tmdb_api_key)
        print("TMDB data download complete.")
    else:
        print("TMDB_API_KEY not found in environment. Skipping TMDB data download.")


if __name__ == "__main__":
    main()

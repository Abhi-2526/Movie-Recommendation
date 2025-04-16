from neo4j import GraphDatabase
import os
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import random
import numpy as np

load_dotenv()


class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

    def setup_constraints(self):
        constraints = [
            "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.id IS UNIQUE",
            "CREATE INDEX movie_title IF NOT EXISTS FOR (m:Movie) ON (m.title)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)"
        ]
        for constraint in constraints:
            self.run_query(constraint)

    def clear_database(self):
        self.run_query("MATCH (n) DETACH DELETE n")

    def create_movie_nodes(self, movies_df):
        batch_size = 1000
        for i in tqdm(range(0, len(movies_df), batch_size), desc="Creating movie nodes"):
            batch = movies_df.iloc[i:i + batch_size].to_dict('records')
            query = """
            UNWIND $batch AS movie
            MERGE (m:Movie {id: movie.movieId})
            ON CREATE SET 
                m.title = movie.title,
                m.avg_rating = movie.avg_rating
            """
            self.run_query(query, {"batch": batch})

    def create_genre_relationships(self, movies_df):
        all_genres = set()
        for genres in movies_df['genres']:
            if isinstance(genres, list):
                all_genres.update(genres)
        for genre in tqdm(all_genres, desc="Creating genre nodes"):
            if genre != "(no genres listed)":
                query = "MERGE (g:Genre {name: $genre})"
                self.run_query(query, {"genre": genre})
        batch_size = 1000
        for i in tqdm(range(0, len(movies_df), batch_size), desc="Creating genre relationships"):
            batch = movies_df.iloc[i:i + batch_size].to_dict('records')
            query = """
            UNWIND $batch AS movie
            MATCH (m:Movie {id: movie.movieId})
            WITH m, movie
            UNWIND movie.genres AS genre
            MATCH (g:Genre {name: genre})
            MERGE (m)-[:HAS_GENRE]->(g)
            """
            self.run_query(query, {"batch": batch})

    def import_tmdb_data(self, mapping_df, max_movies=500, target_genres=None, min_rating=None):
        """
        Import a subset of TMDB data based on parameters

        Args:
            mapping_df: DataFrame with mapping between MovieLens and TMDB IDs
            max_movies: Maximum number of movies to import
            target_genres: List of genres to prioritize for import
            min_rating: Minimum average rating to include
        """
        # If we have a min_rating, filter the mapping_df
        if min_rating is not None:
            mapping_df = mapping_df[mapping_df['avg_rating'] >= min_rating]

        # If we have target genres, filter for those
        selected_movie_ids = None
        if target_genres:
            # This assumes your mapping_df has genre information
            genre_matches = mapping_df['genres'].apply(
                lambda x: any(genre in x.split('|') for genre in target_genres) if isinstance(x, str) else False
            )
            selected_ids = mapping_df[genre_matches]['tmdbId'].values
            if len(selected_ids) > max_movies:
                selected_movie_ids = np.random.choice(selected_ids, size=max_movies, replace=False)
            else:
                selected_movie_ids = selected_ids

        # Get the list of TMDB files
        tmdb_files = os.listdir("../../data/raw/tmdb")

        # Shuffle the files to get a random sample if we don't have specific selections
        if selected_movie_ids is None or len(selected_movie_ids) == 0:
            random.shuffle(tmdb_files)
            tmdb_files = tmdb_files[:max_movies]  # Take only max_movies
        else:
            # Filter files based on our selection
            tmdb_files = [f for f in tmdb_files if int(f.split('.')[0]) in selected_movie_ids]

        # Now process this subset
        for file in tqdm(tmdb_files, desc="Importing TMDB data"):
            file_path = os.path.join("../../data/raw/tmdb", file)
            with open(file_path, 'r') as f:
                movie_data = json.load(f)
            tmdb_id = int(file.split('.')[0])
            movielens_id = mapping_df[mapping_df['tmdbId'] == tmdb_id]['movieId'].values
            if len(movielens_id) == 0:
                continue
            movielens_id = int(movielens_id[0])
            query = """
            MATCH (m:Movie {id: $movieId})
            SET 
                m.overview = $overview,
                m.release_date = $release_date,
                m.runtime = $runtime,
                m.tmdb_id = $tmdb_id,
                m.poster_path = $poster_path
            """
            self.run_query(query, {
                "movieId": movielens_id,
                "overview": movie_data.get('overview', ''),
                "release_date": movie_data.get('release_date', ''),
                "runtime": movie_data.get('runtime', 0),
                "tmdb_id": tmdb_id,
                "poster_path": movie_data.get('poster_path', '')
            })
            # Process cast - reduce to only 5 actors per movie
            cast = movie_data.get('credits', {}).get('cast', [])[:5]  # Reduced from 10 to 5
            for actor in cast:
                person_query = """
                MERGE (p:Person {id: $person_id})
                ON CREATE SET 
                    p.name = $name,
                    p.profile_path = $profile_path
                """
                self.run_query(person_query, {
                    "person_id": actor['id'],
                    "name": actor['name'],
                    "profile_path": actor.get('profile_path', '')
                })
                acted_query = """
                MATCH (p:Person {id: $person_id}), (m:Movie {id: $movie_id})
                MERGE (p)-[r:ACTED_IN]->(m)
                SET 
                    r.character = $character,
                    r.order = $order
                """
                self.run_query(acted_query, {
                    "person_id": actor['id'],
                    "movie_id": movielens_id,
                    "character": actor.get('character', ''),
                    "order": actor.get('order', 0)
                })
            # Process crew: only the main director
            crew = movie_data.get('credits', {}).get('crew', [])
            directors = [c for c in crew if c['job'] == 'Director'][:1]  # Just the main director
            writers = [c for c in crew if c['department'] == 'Writing'][:2]  # Reduced from 5 to 2
            for director in directors:
                person_query = """
                MERGE (p:Person {id: $person_id})
                ON CREATE SET 
                    p.name = $name,
                    p.profile_path = $profile_path
                """
                self.run_query(person_query, {
                    "person_id": director['id'],
                    "name": director['name'],
                    "profile_path": director.get('profile_path', '')
                })
                directed_query = """
                MATCH (p:Person {id: $person_id}), (m:Movie {id: $movie_id})
                MERGE (p)-[:DIRECTED]->(m)
                """
                self.run_query(directed_query, {
                    "person_id": director['id'],
                    "movie_id": movielens_id
                })
            for writer in writers:
                person_query = """
                MERGE (p:Person {id: $person_id})
                ON CREATE SET 
                    p.name = $name,
                    p.profile_path = $profile_path
                """
                self.run_query(person_query, {
                    "person_id": writer['id'],
                    "name": writer['name'],
                    "profile_path": writer.get('profile_path', '')
                })
                wrote_query = """
                MATCH (p:Person {id: $person_id}), (m:Movie {id: $movie_id})
                MERGE (p)-[:WROTE]->(m)
                """
                self.run_query(wrote_query, {
                    "person_id": writer['id'],
                    "movie_id": movielens_id
                })
            # Process keywords - limit to 5 key keywords
            keywords = movie_data.get('keywords', {}).get('keywords', [])[:5]  # Only top 5 keywords
            for keyword in keywords:
                keyword_query = """
                MERGE (k:Keyword {id: $keyword_id})
                ON CREATE SET k.name = $name
                """
                self.run_query(keyword_query, {
                    "keyword_id": keyword['id'],
                    "name": keyword['name']
                })
                has_keyword_query = """
                MATCH (k:Keyword {id: $keyword_id}), (m:Movie {id: $movie_id})
                MERGE (m)-[:HAS_KEYWORD]->(k)
                """
                self.run_query(has_keyword_query, {
                    "keyword_id": keyword['id'],
                    "movie_id": movielens_id
                })

    def create_similar_relationships(self, max_relationships_per_movie=10, min_similarity_score=8, max_movies=None):
        """
        Create SIMILAR_TO relationships between movies with optimized parameters

        Args:
            max_relationships_per_movie: Maximum number of similar movies to link per movie
            min_similarity_score: Minimum similarity score required (higher = stricter)
            max_movies: Maximum number of movies to process for similarities
        """
        print("Creating SIMILAR_TO relationships...")

        # If max_movies is set, only create relationships for a subset of movies
        if max_movies:
            # Get top-rated movies with the most metadata
            query = """
            MATCH (m:Movie)
            WHERE m.avg_rating IS NOT NULL AND m.overview IS NOT NULL
            RETURN m.id AS id
            ORDER BY m.avg_rating DESC
            LIMIT $max_movies
            """
            result = self.run_query(query, {"max_movies": max_movies})
            movie_ids = [record["id"] for record in result]

            # Process in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(movie_ids), batch_size):
                batch = movie_ids[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(len(movie_ids) + batch_size - 1) // batch_size}")

                # For each movie in the batch, find similar movies
                for movie_id in tqdm(batch):
                    self._create_similar_for_movie(movie_id, max_relationships_per_movie, min_similarity_score)
        else:
            # Original approach but with more memory-efficient parameters
            query = """
            MATCH (m1:Movie)-[:HAS_GENRE]->(g)<-[:HAS_GENRE]-(m2:Movie)
            WITH m1, m2, COUNT(g) AS genre_overlap
            WHERE m1 <> m2 AND genre_overlap >= 3
            MATCH (m1)-[:HAS_KEYWORD]->(k)<-[:HAS_KEYWORD]-(m2)
            WITH m1, m2, genre_overlap, COUNT(k) AS keyword_overlap
            WHERE keyword_overlap >= 2
            WITH m1, m2, genre_overlap, keyword_overlap, 
                 genre_overlap * 2 + keyword_overlap AS similarity_score
            WHERE similarity_score >= $min_score
            WITH m1, m2, similarity_score
            ORDER BY similarity_score DESC
            WITH m1, collect({movie: m2, score: similarity_score})[0..$max_per_movie] as similar_movies
            UNWIND similar_movies as similar
            MERGE (m1)-[r:SIMILAR_TO]->(similar.movie)
            SET r.score = similar.score
            """
            self.run_query(query, {
                "min_score": min_similarity_score,
                "max_per_movie": max_relationships_per_movie
            })

    def _create_similar_for_movie(self, movie_id, max_relationships=10, min_score=8):
        """Helper method to create similar relationships for a single movie"""
        # Fixed query to avoid the aggregation warnings by checking for keyword relationships first
        query = """
        MATCH (m1:Movie {id: $movie_id})-[:HAS_GENRE]->(g)<-[:HAS_GENRE]-(m2:Movie)
        WHERE m1 <> m2
        WITH m1, m2, COUNT(g) AS genre_overlap
        WHERE genre_overlap >= 3

        // First check if both movies have keywords
        MATCH (m1)-[:HAS_KEYWORD]->()
        MATCH (m2)-[:HAS_KEYWORD]->()

        // Now safely find overlapping keywords
        MATCH (m1)-[:HAS_KEYWORD]->(k)<-[:HAS_KEYWORD]-(m2)
        WITH m1, m2, genre_overlap, COUNT(k) AS keyword_overlap

        // Calculate similarity score
        WITH m1, m2, genre_overlap * 2 + keyword_overlap AS similarity_score
        WHERE similarity_score >= $min_score

        WITH m1, m2, similarity_score
        ORDER BY similarity_score DESC
        LIMIT $max_relationships

        // Create the relationship
        MERGE (m1)-[r:SIMILAR_TO]->(m2)
        SET r.score = similarity_score
        """
        self.run_query(query, {
            "movie_id": movie_id,
            "min_score": min_score,
            "max_relationships": max_relationships
        })


def main():
    movies_file = "../../data/processed/movies_processed.csv"
    if not os.path.exists(movies_file):
        print("Processed movies file not found. Please run your ETL process first.")
        return
    movies_df = pd.read_csv(movies_file)

    # Create a subset for movies if the dataset is too large
    if len(movies_df) > 60000:
        print(f"Reducing dataset from {len(movies_df)} to 60000 movies")
        # Get top rated movies
        movies_df = movies_df.sort_values('avg_rating', ascending=False).head(60000)

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    neo4j = Neo4jHandler(uri=uri, user=user, password=password)

    print("Setting up constraints...")
    neo4j.setup_constraints()

    print("Clearing existing database...")
    neo4j.clear_database()

    print("Creating movie nodes...")
    neo4j.create_movie_nodes(movies_df)

    movies_df['genres'] = movies_df['genres'].apply(
        lambda x: x.split('|') if isinstance(x, str) else x
    )

    print("Creating genre relationships...")
    neo4j.create_genre_relationships(movies_df)

    mapping_file = "../../data/processed/movielens_tmdb_mapping.csv"
    if os.path.exists(mapping_file):
        print("Importing TMDB data (subset)...")
        mapping_df = pd.read_csv(mapping_file)

        # Merge with movies_df to get genre information
        mapping_df = mapping_df.merge(movies_df[['movieId', 'genres', 'avg_rating']],
                                      on='movieId', how='left')

        # Import 60000 popular movies with at least a 3.5 average rating
        neo4j.import_tmdb_data(
            mapping_df,
            max_movies=60000,
            min_rating=2.5
        )
    else:
        print("TMDB mapping file not found; skipping TMDB data import.")

    print("Creating similar movie relationships (optimized)...")

    neo4j.create_similar_relationships(
        max_relationships_per_movie=10,
        min_similarity_score=8,
        max_movies=60000
    )

    neo4j.close()
    print("Graph population complete.")


if __name__ == "__main__":
    main()
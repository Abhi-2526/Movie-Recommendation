import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
from src.graph.neo4j_handler import Neo4jHandler
from src.retrieval.graph_rag_retriever import GraphRAGRetriever
from src.generation.recommendation_generation import MovieRecommendationGenerator
import os
import logging

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI(title="Movie GraphRAG API", description="API for movie recommendations with GraphRAG")

load_dotenv()
# Environment variables setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DEEPSEEK_TOKEN = os.getenv("DEEPSEEK_API_KEY")

retriever = GraphRAGRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
generator = MovieRecommendationGenerator(DEEPSEEK_TOKEN)


# Pydantic models for request and response
class Movie(BaseModel):
    movie_id: int
    title: str
    overview: Optional[str] = None
    rating: Optional[float] = None
    release_date: Optional[str] = None
    genres: Optional[List[str]] = None
    directors: Optional[List[str]] = None
    actors: Optional[List[str]] = None
    relevance_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    movies: List[Movie]
    explanation: str


class MovieComparisonRequest(BaseModel):
    source_movie_id: int
    target_movie_id: int


class PathNode(BaseModel):
    type: str
    id: Optional[int] = None
    name: Optional[str] = None
    title: Optional[str] = None


class Path(BaseModel):
    nodes: List[PathNode]
    relationships: List[str]


class MovieComparisonResponse(BaseModel):
    source_movie: Movie
    target_movie: Movie
    paths: List[Path]
    explanation: str


# Endpoint for exploring a movie's neighbors in the graph
@app.get("/graph/neighbors")
def graph_neighbors(movie_id: int = Query(..., description="Movie ID to explore neighbors of")):
    try:
        query = """
        MATCH (m:Movie {id: $movie_id})-[r]-(neighbor)
        RETURN m.id AS movie_id, m.title AS movie_title, 
               neighbor.id AS neighbor_id, neighbor.title AS neighbor_title,
               neighbor.name AS neighbor_name, labels(neighbor) AS neighbor_labels,
               type(r) AS relationship
        """
        results = retriever.run_neo4j_query(query, {"movie_id": movie_id})
        nodes = []
        edges = []
        if results:
            main_movie = results[0]
            main_movie_node = {
                "id": main_movie["movie_id"],
                "label": main_movie["movie_title"],
                "type": "Movie"
            }
            nodes.append(main_movie_node)
            for rec in results:
                neighbor_id = rec["neighbor_id"]
                neighbor_labels = rec["neighbor_labels"]
                neighbor_type = "Unknown"
                if "Movie" in neighbor_labels:
                    neighbor_type = "Movie"
                elif "Person" in neighbor_labels:
                    neighbor_type = "Person"
                elif "Genre" in neighbor_labels:
                    neighbor_type = "Genre"
                elif "Keyword" in neighbor_labels:
                    neighbor_type = "Keyword"
                neighbor_label = rec["neighbor_title"] if rec["neighbor_title"] else rec["neighbor_name"]
                if not any(n["id"] == neighbor_id for n in nodes):
                    nodes.append({
                        "id": neighbor_id,
                        "label": neighbor_label,
                        "type": neighbor_type
                    })
                edges.append({
                    "source": main_movie["movie_id"],
                    "source_label": main_movie["movie_title"],
                    "target": neighbor_id,
                    "target_label": neighbor_label,
                    "relationship": rec["relationship"]
                })
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching neighbors: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie GraphRAG API"}


@app.get("/movies/search", response_model=List[Movie])
def search_movies(query: str = Query(..., description="Search query")):
    try:
        movie_ids, _ = retriever.find_similar_movies_by_embedding(query)
        movie_details = []
        for movie_id in movie_ids:
            neo4j_query = """
            MATCH (m:Movie {id: $movie_id})
            OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
            WITH m, COLLECT(DISTINCT g.name) AS genres
            OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
            WITH m, genres, COLLECT(DISTINCT d.name) AS directors
            OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
            WITH m, genres, directors, COLLECT(DISTINCT a.name) AS actors
            RETURN m.id AS movie_id, 
                   m.title AS title, 
                   m.overview AS overview,
                   m.avg_rating AS rating,
                   m.release_date AS release_date,
                   genres,
                   directors,
                   actors
            """
            result = retriever.run_neo4j_query(neo4j_query, {"movie_id": movie_id})
            if result:
                movie_details.append(result[0])
        return movie_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching movies: {str(e)}")


@app.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(query: str = Query(..., description="Natural language query for movie recommendations")):
    try:
        movie_results = retriever.complex_movie_query(query)
        explanation = generator.generate_recommendation(query, movie_results)
        return {"movies": movie_results, "explanation": explanation}
    except Exception as e:
        logger.exception("Error generating recommendations")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@app.get("/movies/keyword", response_model=List[Movie])
def movies_by_keyword(query: str = Query(..., description="Search query for keyword similarity")):
    try:
        movies = retriever.find_movies_by_keyword_similarity(query)
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding movies by keyword similarity: {str(e)}")


@app.get("/movies/genre", response_model=List[Movie])
def movies_by_genre(query: str = Query(..., description="Search query for genre similarity")):
    try:
        movies = retriever.find_movies_by_genre_similarity(query)
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding movies by genre similarity: {str(e)}")


@app.get("/movies/person", response_model=List[Movie])
def movies_by_person(query: str = Query(..., description="Search query for person similarity")):
    try:
        movies = retriever.find_movies_by_person_similarity(query)
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding movies by person similarity: {str(e)}")


@app.get("/movies/similar", response_model=List[Movie])
def similar_movies(movie_id: int = Query(..., description="Movie ID to find similar movies for"), limit: int = 10):
    try:
        movies = retriever.find_similar_movies(movie_id, limit=limit)
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar movies: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

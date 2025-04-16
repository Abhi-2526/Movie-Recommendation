import numpy as np
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase

class GraphRAGRetriever:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.load_embeddings()
        self.build_indices()

    def close(self):
        self.neo4j_driver.close()

    def load_embeddings(self):
        self.movie_embeddings = np.load("../../models/embeddings/movie_embeddings.npy")
        self.movie_ids = pd.read_pickle("../../models/embeddings/movie_ids.pkl")
        self.keyword_embeddings = np.load("../../models/embeddings/keyword_embeddings.npy")
        self.keyword_ids = pd.read_pickle("../../models/embeddings/keyword_ids.pkl")
        self.genre_embeddings = np.load("../../models/embeddings/genre_embeddings.npy")
        self.genre_ids = pd.read_pickle("../../models/embeddings/genre_ids.pkl")
        self.person_embeddings = np.load("../../models/embeddings/person_embeddings.npy")
        self.person_ids = pd.read_pickle("../../models/embeddings/person_ids.pkl")

    def build_indices(self):
        self.movie_index = faiss.IndexFlatIP(self.movie_embeddings.shape[1])
        self.movie_index.add(self.movie_embeddings.astype('float32'))
        self.keyword_index = faiss.IndexFlatIP(self.keyword_embeddings.shape[1])
        self.keyword_index.add(self.keyword_embeddings.astype('float32'))
        self.genre_index = faiss.IndexFlatIP(self.genre_embeddings.shape[1])
        self.genre_index.add(self.genre_embeddings.astype('float32'))
        self.person_index = faiss.IndexFlatIP(self.person_embeddings.shape[1])
        self.person_index.add(self.person_embeddings.astype('float32'))

    def encode_text(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(
            self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return sentence_embedding.cpu().numpy()

    def run_neo4j_query(self, query, parameters=None):
        with self.neo4j_driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

    def find_similar_movies_by_embedding(self, query_text, n=10):
        query_embedding = self.encode_text(query_text)
        distances, indices = self.movie_index.search(query_embedding.astype('float32'), n)
        movie_ids = [self.movie_ids.iloc[idx]['id'] for idx in indices[0]]
        return movie_ids, distances[0]

    def find_movies_by_keyword_similarity(self, query_text, n=5):
        query_embedding = self.encode_text(query_text)
        distances, indices = self.keyword_index.search(query_embedding.astype('float32'), n)
        keyword_ids = [self.keyword_ids.iloc[idx]['id'] for idx in indices[0]]
        query = """
        MATCH (m:Movie)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.id IN $keyword_ids
        WITH m, COUNT(k) AS keyword_count
        ORDER BY keyword_count DESC
        RETURN m.id AS movie_id, 
               m.title AS title, 
               m.overview AS overview,
               m.avg_rating AS rating,
               m.release_date AS release_date,
               keyword_count
        LIMIT 10
        """
        movies = self.run_neo4j_query(query, {"keyword_ids": keyword_ids})
        return movies

    def find_movies_by_genre_similarity(self, query_text, n=3):
        query_embedding = self.encode_text(query_text)
        distances, indices = self.genre_index.search(query_embedding.astype('float32'), n)
        genre_names = [self.genre_ids.iloc[idx]['name'] for idx in indices[0]]
        query = """
        MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
        WHERE g.name IN $genre_names
        WITH m, COLLECT(g.name) AS genres, m
        WHERE SIZE(genres) >= 2
        RETURN m.id AS movie_id, 
               m.title AS title, 
               m.overview AS overview,
               m.avg_rating AS rating,
               m.release_date AS release_date, 
               genres
        ORDER BY m.avg_rating DESC
        LIMIT 10
        """
        movies = self.run_neo4j_query(query, {"genre_names": genre_names})
        return movies

    def find_movies_by_person_similarity(self, query_text, n=5):
        query_embedding = self.encode_text(query_text)
        distances, indices = self.person_index.search(query_embedding.astype('float32'), n)
        person_ids = [self.person_ids.iloc[idx]['id'] for idx in indices[0]]
        query = """
        MATCH (p:Person)-[r:ACTED_IN|DIRECTED|WROTE]->(m:Movie)
        WHERE p.id IN $person_ids
        RETURN m.id AS movie_id, 
               m.title AS title,
               m.overview AS overview,
               m.avg_rating AS rating,
               m.release_date AS release_date,
               COLLECT(DISTINCT {name: p.name, relationship: TYPE(r)}) AS people
        ORDER BY m.avg_rating DESC
        LIMIT 10
        """
        movies = self.run_neo4j_query(query, {"person_ids": person_ids})
        return movies

    def find_movie_paths(self, source_movie_id, target_movie_id, max_depth=3):
        query = """
        MATCH path = shortestPath((m1:Movie {id: $source_id})-[*1..3]-(m2:Movie {id: $target_id}))
        RETURN [node IN nodes(path) | 
        CASE 
          WHEN node:Movie THEN {type: 'Movie', id: node.id, title: node.title}
          WHEN node:Person THEN {type: 'Person', id: node.id, name: node.name}
          WHEN node:Genre THEN {type: 'Genre', name: node.name}
          WHEN node:Keyword THEN {type: 'Keyword', id: node.id, name: node.name}
          ELSE {type: 'Unknown', id: elementId(node)}
        END] AS nodes,
       [rel IN relationships(path) | type(rel)] AS relationships
        LIMIT 3
        """
        paths = self.run_neo4j_query(query, {
            "source_id": source_movie_id,
            "target_id": target_movie_id
        })
        return paths

    def find_similar_movies(self, movie_id, limit=10):
        query = """
        MATCH (m:Movie {id: $movie_id})-[r:SIMILAR_TO]->(similar:Movie)
        RETURN similar.id AS movie_id, similar.title AS title, r.score AS similarity_score
        ORDER BY r.score DESC
        LIMIT $limit
        """
        similar_movies = self.run_neo4j_query(query, {"movie_id": movie_id, "limit": limit})
        return similar_movies

    def complex_movie_query(self, query_text):
        query_embedding = self.encode_text(query_text)
        movie_distances, movie_indices = self.movie_index.search(query_embedding.astype('float32'), 5)
        semantic_movies = [self.movie_ids.iloc[idx]['id'] for idx in movie_indices[0]]
        keyword_distances, keyword_indices = self.keyword_index.search(query_embedding.astype('float32'), 5)
        keyword_ids = [self.keyword_ids.iloc[idx]['id'] for idx in keyword_indices[0]]
        genre_distances, genre_indices = self.genre_index.search(query_embedding.astype('float32'), 3)
        genre_names = [self.genre_ids.iloc[idx]['name'] for idx in genre_indices[0]]
        person_distances, person_indices = self.person_index.search(query_embedding.astype('float32'), 3)
        person_ids = [self.person_ids.iloc[idx]['id'] for idx in person_indices[0]]
        query = """
        MATCH (m:Movie)
        WHERE (
            m.id IN $semantic_movies
            OR
            EXISTS {
                MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
                WHERE k.id IN $keyword_ids
            }
            OR
            EXISTS {
                MATCH (m)-[:HAS_GENRE]->(g:Genre)
                WHERE g.name IN $genre_names
            }
            OR
            EXISTS {
                MATCH (m)<-[:ACTED_IN|DIRECTED|WROTE]-(p:Person)
                WHERE p.id IN $person_ids
            }
        )
        OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.id IN $keyword_ids
        WITH m, COUNT(DISTINCT k) AS keyword_score
        OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
        WHERE g.name IN $genre_names
        WITH m, keyword_score, COUNT(DISTINCT g) AS genre_score
        OPTIONAL MATCH (m)<-[:ACTED_IN|DIRECTED|WROTE]-(p:Person)
        WHERE p.id IN $person_ids
        WITH m, keyword_score, genre_score, COUNT(DISTINCT p) AS person_score
        WITH m, 
             CASE WHEN m.id IN $semantic_movies THEN 50 ELSE 0 END +
             keyword_score * 2 +
             genre_score * 3 +
             person_score * 4 AS relevance_score
        WHERE relevance_score > 0
        MATCH (m)-[:HAS_GENRE]->(g:Genre)
        WITH m, relevance_score, COLLECT(DISTINCT coalesce(g.name, "Unknown")) AS genres
        OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
        WITH m, relevance_score, genres, COLLECT(DISTINCT d.name) AS directors
        OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
        WITH m, relevance_score, genres, directors, COLLECT(DISTINCT a.name) AS actors
        RETURN m.id AS movie_id, 
               m.title AS title, 
               m.overview AS overview,
               m.avg_rating AS rating,
               m.release_date AS release_date,
               genres,
               directors,
               actors,
               relevance_score
        ORDER BY relevance_score DESC
        LIMIT 10
        """
        movies = self.run_neo4j_query(query, {
            "semantic_movies": semantic_movies,
            "keyword_ids": keyword_ids,
            "genre_names": genre_names,
            "person_ids": person_ids
        })
        return movies

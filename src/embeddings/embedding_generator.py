import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from src.graph.neo4j_handler import Neo4jHandler

load_dotenv()


class MovieEmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_embeddings(self, texts, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True,
                                           max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                attention_mask = encoded_input['attention_mask']
                token_embeddings = model_output.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(embeddings)

    def generate_movie_embeddings(self, neo4j_handler):
        query = """
        MATCH (m:Movie)
        WHERE m.overview IS NOT NULL AND m.overview <> ''
        RETURN m.id AS id, m.title AS title, m.overview AS overview
        """
        result = neo4j_handler.run_query(query)
        movies_df = pd.DataFrame(
            [{'id': record['id'], 'title': record['title'], 'overview': record['overview']} for record in result])
        texts = [f"Movie: {row.title}. Plot: {row.overview}" for _, row in movies_df.iterrows()]
        embeddings = self.generate_embeddings(texts)
        os.makedirs("../../models/embeddings", exist_ok=True)
        np.save("../../models/embeddings/movie_embeddings.npy", embeddings)
        movies_df.to_pickle("../../models/embeddings/movie_ids.pkl")
        return embeddings, movies_df

    def generate_keyword_embeddings(self, neo4j_handler):
        query = "MATCH (k:Keyword) RETURN k.id AS id, k.name AS name"
        result = neo4j_handler.run_query(query)
        keywords_df = pd.DataFrame([{'id': record['id'], 'name': record['name']} for record in result])
        texts = [f"Movie keyword: {row.name}" for _, row in keywords_df.iterrows()]
        embeddings = self.generate_embeddings(texts)
        np.save("../../models/embeddings/keyword_embeddings.npy", embeddings)
        keywords_df.to_pickle("../../models/embeddings/keyword_ids.pkl")
        return embeddings, keywords_df

    def generate_genre_embeddings(self, neo4j_handler):
        query = "MATCH (g:Genre) RETURN g.name AS name"
        result = neo4j_handler.run_query(query)
        genres_df = pd.DataFrame([{'name': record['name']} for record in result])
        texts = [f"Movie genre: {row.name}" for _, row in genres_df.iterrows()]
        embeddings = self.generate_embeddings(texts)
        np.save("../../models/embeddings/genre_embeddings.npy", embeddings)
        genres_df.to_pickle("../../models/embeddings/genre_ids.pkl")
        return embeddings, genres_df

    def generate_person_embeddings(self, neo4j_handler):
        query = "MATCH (p:Person) RETURN p.id AS id, p.name AS name"
        result = neo4j_handler.run_query(query)
        persons_df = pd.DataFrame([{'id': record['id'], 'name': record['name']} for record in result])
        texts = [f"Person in film industry: {row.name}" for _, row in persons_df.iterrows()]
        embeddings = self.generate_embeddings(texts)
        np.save("../../models/embeddings/person_embeddings.npy", embeddings)
        persons_df.to_pickle("../../models/embeddings/person_ids.pkl")
        return embeddings, persons_df


def main():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    # Import the Neo4jHandler from your graph module

    # Initialize Neo4j connection (update password as needed)
    neo4j = Neo4jHandler(uri=uri, user=user, password=password)

    generator = MovieEmbeddingGenerator()

    # Generate movie embeddings
    print("Generating movie embeddings...")
    movie_embeddings, movie_ids_df = generator.generate_movie_embeddings(neo4j)
    print("Movie embeddings generated and saved")

    print("Generating keyword embeddings...")
    keyword_embeddings, keyword_ids_df = generator.generate_keyword_embeddings(neo4j)
    print("Keyword embeddings generated and saved.")

    print("Generating genre embeddings...")
    genre_embeddings, genre_ids_df = generator.generate_genre_embeddings(neo4j)
    print("Genre embeddings generated and saved.")

    print("Generating person embeddings...")
    person_embeddings, person_ids_df = generator.generate_person_embeddings(neo4j)
    print("Person embeddings generated and saved.")

    neo4j.close()
    print("Neo4j connection closed.")


if __name__ == "__main__":
    main()


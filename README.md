

https://github.com/user-attachments/assets/0e779bc9-66fb-43f6-bfa6-64810c94ddbc

# Movie GraphRAG System

Movie GraphRAG is an advanced movie recommendation system that combines graph-based data enrichment with semantic embedding retrieval and language generation for explainable recommendations. By leveraging a Neo4j graph database for structured relationships and transformer-based embeddings for content understanding, the system can answer complex natural language queries and provide transparent recommendations.

## Features

- **Structured Data in a Graph Database:**  
  Movie nodes, along with associated genres, keywords, and person (actor, director, writer) relationships, are stored in Neo4j.
  
- **Deep Semantic Embeddings:**  
  Overviews and other textual attributes are converted into embeddings using the `sentence-transformers/all-mpnet-base-v2` model to enable efficient semantic similarity search.
  
- **Hybrid Retrieval:**  
  The system combines semantic retrieval via FAISS with graph traversal queries to generate recommendations.
  
- **Explainable Recommendations:**  
  Using a DeepSeek (or similar LLM) integration, the system generates detailed, explainable recommendations based only on the retrieved data.
  
- **Multiple Endpoints:**  
  Supports retrieving movies by keyword similarity, genre similarity, person similarity, and precomputed similar relationships.

- **Interactive UI:**  
  A Streamlit user interface provides a user-friendly way to search movies, request contextual recommendations, explore graph connections, and visualize results.

## Prerequisites

Before starting, ensure you have the following tools and services ready:

- **Git**: For cloning the repository.
- **Python 3.8 or higher**: For running scripts and applications.
- **pip**: For installing Python dependencies.
- **Docker**: For running a local Neo4j instance (optional if using a managed service).
- **Neo4j instance**: Either a local instance via Docker or a managed service like [Neo4j Aura](https://neo4j.com/cloud/aura/).
- **DeepSeek or OpenAI API token**: Required for features like embedding generation.
  
## Project Structure (Only src with code given, run the scripts in order given below and create this file structure)

```plaintext
movie-graphrag/
├── data/                         
│   ├── raw/                      # Raw datasets (e.g., MovieLens, TMDB JSON files)
│   └── processed/                # Processed CSV files (movies_processed.csv, mapping files)
├── models/                       
│   └── embeddings/               # Generated embeddings (NumPy arrays) and DataFrames stored as Pickles
├── docker/                       # Dockerfile or docker-compose.yml for running Neo4j (if needed)
├── notebooks/                    # Jupyter notebooks for data exploration (optional)
├── src/
│   ├── etl/
│   │   └── data_download.py      # ETL functions to download, extract, and process data
│   ├── graph/
│   │   └── neo4j_handler.py      # Neo4jHandler class for database interactions
│   ├── embeddings/
│   │   └── embedding_generator.py  # MovieEmbeddingGenerator class for generating embeddings
│   ├── retrieval/
│   │   └── graph_rag_retriever.py   # GraphRAGRetriever class: retrieval using FAISS and Neo4j queries
│   ├── generation/
│   │   └── recommendation_generation.py  # MovieRecommendationGenerator using DeepSeek LLM
│   └── api/
│       └── main.py               # FastAPI backend with endpoints for recommendations and searches
│   └── ui/
│       └── streamlit_app.py      # Streamlit UI for interactive exploration and visualization
├── .env                          # Environment configuration file
├── requirements.txt              # Python dependencies
└── README.md
```
## Installation and Setup

Follow these steps to set up the project. All commands should be run from the project root directory unless specified otherwise.

### 1. Clone the Repository

Clone the project repository and navigate to the project folder:


### 2. Create and Activate a Virtual Environment

Create a Python virtual environment to isolate dependencies:

- **On macOS/Linux**:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

- **On Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

This ensures that dependencies are installed locally without affecting your system’s Python environment.

### 3. Install Python Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: Ensure the `requirements.txt` file exists in the project root. If dependencies fail to install, verify your Python version and pip configuration.

### 4. Configure Environment Variables

Create a `.env` file in the project root to store sensitive configuration details:

```dotenv
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# DeepSeek API Token
DEEPSEEK_API_TOKEN=your_actual_deepseek_api_key_here
```

- **Instructions**:
  - Replace `your_neo4j_password_here` with your Neo4j password.
  - Replace `your_actual_deepseek_api_key_here` with your DeepSeek API token, obtainable from the [DeepSeek Platform](https://platform.deepseek.com/api_keys).
  - Ensure no extra spaces or quotes are included in the `.env` file to avoid configuration errors.

### 5. Run the ETL Process

Download and process the MovieLens dataset:

```bash
python src/etl/data_download.py
```

This script:
- Downloads the MovieLens dataset (likely one of the versions like 25M, 1M, or 100K, depending on the script’s configuration).
- Processes the data, including splitting genres.
- Creates mapping files stored in the `data/processed` directory.

**Note**: The MovieLens dataset is a widely used benchmark for recommendation systems, containing movie ratings and metadata ([MovieLens | GroupLens](https://grouplens.org/datasets/movielens/)).

### 6. Set Up Neo4j

Set up a Neo4j graph database to store the processed data.

- **Option 1: Run Neo4j Locally with Docker**

  Start a local Neo4j instance using Docker:

  ```bash
  docker run -d --name my-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_neo4j_password neo4j:latest
  ```

  - Replace `your_neo4j_password` with the password you set in the `.env` file.
  - This command runs Neo4j in detached mode, mapping ports 7474 (HTTP) and 7687 (Bolt) for access.
  - Access the Neo4j browser at `http://localhost:7474` to verify the instance is running.

- **Option 2: Use a Managed Neo4j Service**

  Use a managed service like [Neo4j Aura Free](https://neo4j.com/cloud/aura/). Update the `.env` file with the provided connection details:

  - `NEO4J_URI`: The Bolt URI for your Aura instance.
  - `NEO4J_USER`: The username (usually `neo4j`).
  - `NEO4J_PASSWORD`: The password for your Aura instance.

**Note**: Ensure Docker is installed and running if using the local option. For managed services, follow the provider’s setup instructions.

### 7. Populate Neo4j Database

Populate the Neo4j database with the processed MovieLens data:

```bash
python src/graph/neo4j_handler.py
```

This script likely creates nodes and relationships in Neo4j (e.g., for movies, users, ratings) based on the processed data.

### 8. Generate Embeddings

Generate embeddings for movies and optionally for keywords, genres, or persons:

```bash
python src/embeddings/embedding_generator.py
```

- This script generates embeddings, likely using the DeepSeek API for advanced machine learning capabilities.
- Embeddings are stored as NumPy arrays and Pickle files in the `models/embeddings` directory.
- Embeddings enhance recommendation systems by representing movies and other entities in a vector space.

**Note**: Ensure your DeepSeek API token is correctly configured in the `.env` file, as it may be required for this step ([DeepSeek API Docs](https://api-docs.deepseek.com/)).

### 9. Start the FastAPI Backend

Launch the FastAPI backend to serve the application’s API:

```bash
uvicorn src/api/main:app --reload
```

- The `--reload` flag enables auto-reloading during development.
- The API will be available at `http://localhost:8000` (update if your configuration differs).

**Note**: FastAPI is a modern Python web framework for building APIs ([FastAPI](https://fastapi.tiangolo.com/)). Verify that the `src/api/main.py` file exists and is correctly configured.

### 10. Launch the Streamlit UI

Launch the Streamlit user interface for interactive features:

```bash
streamlit run src/ui/streamlit_app.py
```

- The UI will be available at `http://localhost:8501` (update if your configuration differs).
- Features include search, movie recommendations, and graph exploration.

**Note**: Streamlit is a Python library for creating interactive web applications ([Streamlit](https://streamlit.io/)). Ensure the `src/ui/streamlit_app.py` file exists.

## Notes

- **Command Execution**: Run all commands from the project root directory unless specified otherwise.
- **Docker**: Ensure Docker is installed and running for the local Neo4j setup. Install Docker from [Docker](https://www.docker.com/get-started).
- **Neo4j Managed Services**: If using Neo4j Aura or similar, update the `.env` file with the correct connection details and skip the Docker step.
- **DeepSeek API**: Obtain your API token from the [DeepSeek Platform](https://platform.deepseek.com/api_keys) and ensure it’s correctly set in the `.env` file.
- **Troubleshooting**:
  - If dependencies fail to install, check your Python version and pip configuration.
  - If Neo4j fails to connect, verify the URI, username, and password in the `.env` file.
  - If the API or UI fails to start, ensure the respective scripts (`main.py`, `streamlit_app.py`) exist and are correctly configured.
- **MovieLens Dataset**: The dataset version (e.g., 25M, 1M) depends on the `data_download.py` script. Refer to [MovieLens | GroupLens](https://grouplens.org/datasets/movielens/) for details on available versions.

## Project Structure

Below is the assumed project structure based on the setup steps:

| Directory/File                | Description                                      |
|-------------------------------|--------------------------------------------------|
| `src/etl/data_download.py`    | Script to download and process MovieLens data    |
| `src/graph/neo4j_handler.py`  | Script to populate Neo4j database                |
| `src/embeddings/embedding_generator.py` | Script to generate embeddings           |
| `src/api/main.py`             | FastAPI application entry point                  |
| `src/ui/streamlit_app.py`     | Streamlit UI application entry point             |
| `data/processed/`             | Directory for processed data files               |
| `models/embeddings/`          | Directory for embedding files                    |
| `requirements.txt`            | List of Python dependencies                      |
| `.env`                        | Environment variables file                       |

This structure helps you locate scripts and understand their roles in the setup process.

## Additional Information

- **MovieLens Dataset**: The MovieLens dataset, maintained by [GroupLens](https://grouplens.org/datasets/movielens/), is a benchmark for recommendation systems. It includes movie ratings, tags, and metadata across versions like 100K, 1M, and 25M. The `data_download.py` script likely selects an appropriate version for this project.
- **Neo4j**: Neo4j is a graph database ideal for storing relationships like user-movie ratings ([Neo4j](https://neo4j.com/)). The local Docker setup is convenient for development, while Neo4j Aura offers a cloud-based alternative.
- **DeepSeek API**: The DeepSeek API, compatible with OpenAI’s format, is likely used for generating embeddings or other AI-driven features ([DeepSeek API Docs](https://api-docs.deepseek.com/)). Ensure your API token is valid to avoid errors in the embedding generation step.
- **FastAPI and Streamlit**: These frameworks enable a robust backend and interactive UI, respectively. FastAPI handles API requests, while Streamlit provides a user-friendly interface for exploring recommendations and graphs.

By following these steps, you should have the movie-graphrag project fully set up and running, ready to explore movie recommendations and graph-based features.

import os
from openai import OpenAI
from langchain.prompts import PromptTemplate
from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable


class DeepSeekLLM:
    """
    A simple wrapper for the DeepSeek Chat API that can be used as a Runnable.
    """

    def __init__(self, api_key):
        # Set up the DeepSeek API client with the given API key and base URL.
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def invoke(self, input_text: str, config: Optional[Dict] = None) -> str:
        """
        Makes a call to DeepSeek's chat completions endpoint.
        Compatible with LangChain's Runnable interface.
        """
        # Get parameters from config or use defaults
        config = config or {}
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1024)

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": input_text},
            ],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Return the generated text from DeepSeek's response.
        return response.choices[0].message.content


class MovieRecommendationGenerator:
    def __init__(self, deepseek_api_key):
        self.llm = DeepSeekLLM(api_key=deepseek_api_key)
        self.setup_templates()

    def setup_templates(self):
        # General recommendation template.
        self.general_recommendation_template = PromptTemplate.from_template(
            """You are a movie recommendation expert. 

The user asked: "{query}"

Based on their query, I've found these movies that might match what they're looking for:

{movie_results}

Please provide a thoughtful explanation of why these movies match the user's request. 
Include relevant details about plot, themes, actors, directors, or genres that connect 
to what the user is looking for. Format your answer in a conversational way.
Don't drift, your answer should only be from the movies I gave you, nothing else.
Specifically address how these recommendations relate to these aspects of the query: {query_keywords}

Also, suggest 1-2 movies from the list that you think would be the best starting points."""
        )

        # Movie comparison template.
        self.movie_comparison_template = PromptTemplate.from_template(
            """You are a movie recommendation expert.

The user wants to understand the connection between these movies: {source_movie} and {target_movie}.

I've found the following paths connecting these movies in our knowledge graph:

{connection_paths}

Please explain these connections in an interesting and informative way. Discuss what these 
connections reveal about both movies and why a fan of one might enjoy the other.

Include relevant details about shared themes, style, creators, or other elements that 
link these films together."""
        )

    def format_movie_results(self, movies):
        formatted_results = []
        for i, movie in enumerate(movies, 1):
            movie_info = f"{i}. {movie['title']} "
            if 'release_date' in movie and movie['release_date']:
                try:
                    year = movie['release_date'].split('-')[0]
                    movie_info += f"({year}) "
                except:
                    pass
            if 'rating' in movie and movie['rating']:
                movie_info += f"- Rating: {movie['rating']:.1f}/5 "
            if 'genres' in movie and movie['genres']:
                if isinstance(movie['genres'], list):
                    movie_info += f"- Genres: {', '.join(movie['genres'])} "
                else:
                    movie_info += f"- Genres: {movie['genres']} "
            if 'directors' in movie and movie['directors'] and len(movie['directors']) > 0:
                if isinstance(movie['directors'], list):
                    movie_info += f"- Director(s): {', '.join(movie['directors'][:2])} "
                else:
                    movie_info += f"- Director(s): {movie['directors']} "
            if 'actors' in movie and movie['actors'] and len(movie['actors']) > 0:
                if isinstance(movie['actors'], list):
                    movie_info += f"- Starring: {', '.join(movie['actors'][:3])}"
                else:
                    movie_info += f"- Starring: {movie['actors']}"
            if 'overview' in movie and movie['overview']:
                movie_info += f"\n   Plot: {movie['overview']}"
            formatted_results.append(movie_info)
        return "\n\n".join(formatted_results)

    def format_connection_paths(self, paths):
        formatted_paths = []
        for i, path in enumerate(paths, 1):
            formatted_path = f"Path {i}:\n"
            nodes = path['nodes']
            relationships = path['relationships']
            for j in range(len(nodes) - 1):
                current = nodes[j]
                next_node = nodes[j + 1]
                relationship = relationships[j]
                if current.get('type') == 'Movie':
                    formatted_path += f"Movie '{current.get('title')}' "
                elif current.get('type') == 'Person':
                    formatted_path += f"Person '{current.get('name')}' "
                elif current.get('type') == 'Genre':
                    formatted_path += f"Genre '{current.get('name')}' "
                elif current.get('type') == 'Keyword':
                    formatted_path += f"Keyword '{current.get('name')}' "
                formatted_path += f"--[{relationship}]--> "
                if j == len(nodes) - 2:
                    if next_node.get('type') == 'Movie':
                        formatted_path += f"Movie '{next_node.get('title')}'"
                    elif next_node.get('type') == 'Person':
                        formatted_path += f"Person '{next_node.get('name')}'"
                    elif next_node.get('type') == 'Genre':
                        formatted_path += f"Genre '{next_node.get('name')}'"
                    elif next_node.get('type') == 'Keyword':
                        formatted_path += f"Keyword '{next_node.get('name')}'"
            formatted_paths.append(formatted_path)
        return "\n\n".join(formatted_paths)

    def extract_query_keywords(self, query):
        import re
        common_words = {"movie", "movies", "film", "films", "like", "similar", "to", "with", "about", "and", "or",
                        "the", "a", "an"}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        seen = set()
        unique_keywords = [k for k in keywords if not (k in seen or seen.add(k))]
        return ", ".join(unique_keywords[:5])

    def generate_recommendation(self, query, movie_results):
        formatted_movies = self.format_movie_results(movie_results)
        query_keywords = self.extract_query_keywords(query)

        # Format prompt and call LLM directly
        prompt = self.general_recommendation_template.format(
            query=query,
            movie_results=formatted_movies,
            query_keywords=query_keywords
        )
        print("Generated Recommendation Prompt:")
        print(prompt)

        return self.llm.invoke(prompt)

    def generate_movie_comparison(self, source_movie, target_movie, connection_paths):
        formatted_paths = self.format_connection_paths(connection_paths)

        # Format prompt and call LLM directly
        prompt = self.movie_comparison_template.format(
            source_movie=source_movie,
            target_movie=target_movie,
            connection_paths=formatted_paths
        )
        print("Generated Movie Comparision Prompt:")
        print(prompt)

        return self.llm.invoke(prompt)
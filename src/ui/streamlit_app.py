import streamlit as st
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from security import safe_requests


# Set up page configuration and API URL
def setup_page():
    st.set_page_config(page_title="Movie GraphRAG Explorer", page_icon="🎬", layout="wide")
    st.title("🎬 Movie GraphRAG Explorer")
    st.markdown("""
    This app uses a knowledge graph of movies combined with semantic search to provide 
    contextualized movie recommendations and insights.
    """)
    st.sidebar.title("Navigation")
    # Updated radio options with new functionalities
    return st.sidebar.radio("Choose a feature:",
                            ["Movie Search",
                             "Contextual Recommendations",
                             "Movies by Keyword Similarity",
                             "Movies by Genre Similarity",
                             "Movies by Person Similarity",
                             "Similar Movies",
                             "Graph Explorer"])


def api_request(endpoint, params=None, json_data=None, method="get"):
    API_URL = "http://localhost:8000"
    try:
        if method == "get":
            response = safe_requests.get(f"{API_URL}/{endpoint}", params=params)
        else:
            response = requests.post(f"{API_URL}/{endpoint}", json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def movie_search_page():
    st.header("Movie Search")
    search_query = st.text_input("Enter your search:", placeholder="e.g., science fiction adventure")
    if search_query:
        with st.spinner("Searching movies..."):
            results = api_request("movies/search", {"query": search_query})
        if results:
            st.subheader(f"Search Results for '{search_query}'")
            for movie in results:
                with st.expander(
                        f"{movie['title']} ({movie['release_date'][:4] if movie.get('release_date') else 'N/A'})"):
                    cols = st.columns([3, 2])
                    with cols[0]:
                        st.markdown(f"**Overview**: {movie.get('overview', 'No overview available')}")
                        st.markdown(f"**Rating**: {movie.get('rating', 'N/A'):.1f}/5" if movie.get(
                            'rating') else "**Rating**: N/A")
                        genres = movie.get('genres', [])
                        if genres:
                            st.markdown(f"**Genres**: {', '.join(genres)}")
                        directors = movie.get('directors', [])
                        if directors:
                            st.markdown(f"**Director(s)**: {', '.join(directors[:3])}")
                        actors = movie.get('actors', [])
                        if actors:
                            st.markdown(f"**Starring**: {', '.join(actors[:5])}")
                    with cols[1]:
                        if movie.get('poster_path'):
                            poster_url = f"https://image.tmdb.org/t/p/w300{movie['poster_path']}"
                            st.image(poster_url, width=200)
                        else:
                            st.markdown("*No poster available*")


def contextual_recommendations_page():
    st.header("Contextual Movie Recommendations")
    st.markdown("""
    Ask for movie recommendations using natural language. Be as specific as possible 
    about genres, themes, actors, directors, or any other criteria.
    """)
    examples = [
        "Movies like Inception but with female protagonists",
        "Sci-fi movies that explore time travel with a philosophical angle",
        "Action comedies with unlikely friendships from the 2010s",
        "Movies directed by Christopher Nolan with themes of identity",
        "Films that connect the worlds of Wes Anderson and Quentin Tarantino"
    ]
    example_query = st.selectbox("Try an example query:", [""] + examples)
    if example_query:
        user_query = example_query
    else:
        user_query = st.text_area("Enter your recommendation request:",
                                  placeholder="e.g., Movies like The Matrix but with more philosophical themes")
    if user_query:
        with st.spinner("Finding recommendations..."):
            recommendations = api_request("recommendations", {"query": user_query})
        if recommendations:
            st.subheader("Recommended Movies")
            with st.expander("Why these recommendations?", expanded=True):
                st.markdown(recommendations["explanation"])
            for i, movie in enumerate(recommendations["movies"]):
                with st.expander(
                        f"{i + 1}. {movie['title']} ({movie['release_date'][:4] if movie.get('release_date') else 'N/A'})"):
                    cols = st.columns([3, 2])
                    with cols[0]:
                        st.markdown(f"**Overview**: {movie.get('overview', 'No overview available')}")
                        st.markdown(f"**Rating**: {movie.get('rating', 'N/A'):.1f}/5" if movie.get(
                            'rating') else "**Rating**: N/A")
                        st.markdown(f"**Relevance Score**: {movie.get('relevance_score', 'N/A'):.1f}" if movie.get(
                            'relevance_score') else "**Relevance**: N/A")
                        genres = movie.get('genres', [])
                        if genres:
                            st.markdown(f"**Genres**: {', '.join(genres)}")
                        directors = movie.get('directors', [])
                        if directors:
                            st.markdown(f"**Director(s)**: {', '.join(directors[:3])}")
                        actors = movie.get('actors', [])
                        if actors:
                            st.markdown(f"**Starring**: {', '.join(actors[:5])}")
                    with cols[1]:
                        if movie.get('poster_path'):
                            poster_url = f"https://image.tmdb.org/t/p/w300{movie['poster_path']}"
                            st.image(poster_url, width=200)
                        else:
                            st.markdown("*No poster available*")


def movies_by_keyword_page():
    st.header("Movies by Keyword Similarity")
    search_query = st.text_input("Enter a keyword-based query:", placeholder="e.g., thriller, mystery")
    if search_query:
        with st.spinner("Searching movies by keyword..."):
            results = api_request("movies/keyword", {"query": search_query})
        if results:
            st.subheader(f"Movies matching keyword similarity for '{search_query}'")
            for movie in results:
                with st.expander(
                        f"{movie['title']} ({movie.get('release_date', 'N/A')})"):
                    st.markdown(f"**Rating**: {movie.get('rating', 'N/A')}")
                    st.markdown(f"**Keyword Count**: {movie.get('keyword_count', 'N/A')}")
        else:
            st.info("No movies found using keyword similarity.")


def movies_by_genre_page():
    st.header("Movies by Genre Similarity")
    search_query = st.text_input("Enter a genre-based query:", placeholder="e.g., action, comedy")
    if search_query:
        genre_list = [g.strip() for g in search_query.split(",") if g.strip()]
        with st.spinner("Searching movies by genre..."):
            results = api_request("movies/genre", {"query": genre_list})
        if results:
            st.subheader(f"Movies matching genre similarity for '{genre_list}'")
            for movie in results:
                with st.expander(
                        f"{movie['title']} ({movie.get('release_date', 'N/A')})"):
                    st.markdown(f"**Rating**: {movie.get('rating', 'N/A')}")
                    st.markdown(f"**Genres**: {', '.join(movie.get('genres', []))}")
        else:
            st.info("No movies found using genre similarity.")


def movies_by_person_page():
    st.header("Movies by Person Similarity")
    search_query = st.text_input("Enter a person-based query:", placeholder="e.g., Nolan, Portman")
    if search_query:
        with st.spinner("Searching movies by person..."):
            results = api_request("movies/person", {"query": search_query})
        if results:
            st.subheader(f"Movies matching person similarity for '{search_query}'")
            for movie in results:
                with st.expander(
                        f"{movie['title']} ({movie.get('release_date', 'N/A')})"):
                    st.markdown(f"**Rating**: {movie.get('rating', 'N/A')}")
                    # Safely join the names of people if present
                    people = movie.get('people', [])
                    if people and isinstance(people, list):
                        try:
                            people_str = ", ".join([p['name'] for p in people if 'name' in p])
                        except Exception as e:
                            people_str = "Error processing people"
                    else:
                        people_str = "N/A"
                    st.markdown(f"**People Involved**: {people_str}")
        else:
            st.info("No movies found using person similarity.")


def similar_movies_page():
    st.header("Similar Movies (Precomputed)")
    movie_id = st.number_input("Enter movie ID to find similar movies:", min_value=0, value=1)
    with st.spinner("Searching similar movies..."):
        results = api_request("movies/similar", {"movie_id": movie_id, "limit": 10})
    if results:
        st.subheader(f"Movies similar to movie ID {movie_id}")
        for movie in results:
            st.markdown(f"- **{movie['title']}** (ID: {movie['movie_id']}, Score: {movie.get('similarity_score', 'N/A')})")
    else:
        st.info("No similar movies found.")


def graph_explorer_page():
    st.header("Graph Explorer")
    st.markdown("""
    Explore the movie knowledge graph by viewing a movie's neighbors (actors, directors, genres, keywords).
    """)
    search_query_explorer = st.text_input("Enter a movie title to explore its connections:")
    if search_query_explorer:
        with st.spinner("Searching..."):
            results_explorer = api_request("movies/search", {"query": search_query_explorer})
        if results_explorer:
            st.subheader("Search Results:")
            selected_title = st.selectbox("Select a movie:", [r["title"] for r in results_explorer])
            selected_movie = next(r for r in results_explorer if r["title"] == selected_title)
            st.subheader(f"Exploring connections for: {selected_movie['title']}")
            neighbors = api_request("graph/neighbors", {"movie_id": selected_movie["movie_id"]})
            if neighbors:
                G = nx.Graph()
                for node in neighbors.get("nodes", []):
                    G.add_node(node.get("label"), type=node.get("type"))
                for edge in neighbors.get("edges", []):
                    G.add_edge(edge.get("source_label"), edge.get("target_label"), label=edge.get("relationship"))
                fig, ax = plt.subplots(figsize=(12, 6))
                pos = nx.spring_layout(G)
                type_colors = {"Movie": "lightblue", "Person": "lightgreen", "Genre": "lightcoral",
                               "Keyword": "lightyellow"}
                for node_type, color in type_colors.items():
                    nodes_of_type = [n for n, attr in G.nodes(data=True) if attr.get("type") == node_type]
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, node_color=color, node_size=500, ax=ax)
                nx.draw_networkx_edges(G, pos, ax=ax)
                nx.draw_networkx_labels(G, pos, ax=ax)
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No neighbor data available for this movie.")
        else:
            st.info("No movie found for the given query.")


def main():
    page = setup_page()

    if page == "Movie Search":
        movie_search_page()
    elif page == "Contextual Recommendations":
        contextual_recommendations_page()
    elif page == "Movies by Keyword Similarity":
        movies_by_keyword_page()
    elif page == "Movies by Genre Similarity":
        movies_by_genre_page()
    elif page == "Movies by Person Similarity":
        movies_by_person_page()
    elif page == "Similar Movies":
        similar_movies_page()
    elif page == "Graph Explorer":
        graph_explorer_page()


if __name__ == "__main__":
    main()

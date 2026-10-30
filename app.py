import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load dataset and similarity matrix
movies = pd.read_csv("data/movies_small.csv")
similarity = joblib.load("model/similarity.pkl")

# Recommendation function
def recommend(movie_name):

    movie_index = movies[
        movies["title"] == movie_name
    ].index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in scores[1:6]:
        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genres": movies.iloc[index]["genres"],
            "overview": movies.iloc[index]["overview"],
            "similarity": round(score * 100, 2)
        })

    return recommendations


# Application UI
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite movies "
    "using Machine Learning."
)

st.divider()

selected_movie = st.selectbox(
    "Select your favourite movie",
    movies["title"].tolist()
)

if st.button("Recommend Movies"):

    results = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for i, movie in enumerate(results, start=1):

        with st.container(border=True):

            st.subheader(
                f"{i}. {movie['title']}"
            )

            st.write("**Genres:**", movie["genres"])

            st.write(movie["overview"])

            st.progress(
                min(movie["similarity"] / 100, 1.0)
            )

            st.caption(
                f"Similarity score: {movie['similarity']}%"
            )

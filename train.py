
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
movies = pd.read_csv("data/movies_small.csv")

# 2. Handle missing values
features = ["genres", "keywords", "overview"]

for feature in features:
    movies[feature] = movies[feature].fillna("")

# 3. Combine movie features
movies["combined_features"] = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["overview"]
)

# 4. Convert text into numerical vectors
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)

# 5. Calculate cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# 6. Save similarity matrix
os.makedirs("model", exist_ok=True)

joblib.dump(similarity, "model/similarity.pkl")

print("Model training completed!")
print("Total movies:", len(movies))
print("Similarity matrix shape:", similarity.shape)

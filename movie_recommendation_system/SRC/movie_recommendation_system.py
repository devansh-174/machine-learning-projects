# A recommendation system sends out suggestions to users through a filtering process based on 
# A content-based recommendation system that suggests movies
# based on a user's own viewing history and genre preferences.

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def recommend_tfidf(user_id, moviedata, top_k=10):
    user_data = moviedata[moviedata["User_Id"] == user_id]
    liked_movies = user_data[user_data["Rating"] >= 4]

    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    genre_matrix = tfidf.fit_transform(moviedata["Genre"])

    # create a user profile by averaging the genre vectors of the movies they liked
    if liked_movies.empty:
        user_profile = np.asarray(genre_matrix.mean(axis=0))
    else:
        liked_indices = liked_movies.index
        user_profile = np.asarray(genre_matrix[liked_indices].mean(axis=0))

    # Ensure correct shape for cosine_similarity (1, n_features)
    user_profile = user_profile.reshape(1, -1)
    
    # to determine how similar each movie's genre vector is to the user's profile
    similarity_scores = cosine_similarity(user_profile, genre_matrix).flatten()

    df = moviedata.copy()
    df["genre_similarity"] = similarity_scores
    df["rating_norm"] = df["Rating"] / 5

    df["final_score"] = (
        0.7 * df["genre_similarity"] +
        0.3 * df["rating_norm"]
    )

    recommendations = (
        df[~df["Movie_Name"].isin(liked_movies["Movie_Name"])]
        .sort_values(by="final_score", ascending=False)
        .head(top_k)
    )

    return recommendations



model = SentenceTransformer("all-MiniLM-L6-v2")
def recommend_transformer(user_id, moviedata, top_k=10):

    user_data = moviedata[moviedata["User_Id"] == user_id]
    liked_movies = user_data[user_data["Rating"] >= 4]

    # to create embeddings for the genres of all movies using a pre-trained sentence transformer model
    genre_embeddings = model.encode(
        moviedata["Genre"].tolist(),
        convert_to_numpy=True
    )

    if liked_movies.empty:
        user_profile = genre_embeddings.mean(axis=0)
    else:
        liked_indices = liked_movies.index
        user_profile = genre_embeddings[liked_indices].mean(axis=0)

    user_profile = user_profile.reshape(1, -1)

    similarity_scores = cosine_similarity(
        user_profile,
        genre_embeddings
    ).flatten()

    df = moviedata.copy()

    df["genre_similarity"] = similarity_scores
    df["rating_norm"] = df["Rating"] / 5

    df["final_score"] = (
        0.7 * df["genre_similarity"] +
        0.3 * df["rating_norm"]
    )

    recommendations = (
        df[~df["Movie_Name"].isin(liked_movies["Movie_Name"])]
        .sort_values(by="final_score", ascending=False)
        .head(top_k)
    )

    return recommendations


def precision_at_k(recommendations, user_data, k=10):

    liked_movies = user_data[user_data["Rating"] >= 4]["Movie_Name"]

    recommended_movies = recommendations["Movie_Name"].head(k)

    relevant = recommended_movies.isin(liked_movies).sum()

    return relevant / k

#reading the data of our file
"""
Dataset: MovieLens-style dataset
Source: Kaggle
Description:
- Movie_Name: title of the movie
- Genre: pipe-separated genres
- Rating: user rating (1–5)
"""

moviedata = pd.read_csv('/Users/devanshbansal/Downloads/movies_dataset.csv')
#using EDA to find out the values where cleaning is required
print(moviedata.isnull().sum())
# Ratings are user-generated; mean imputation used to
# preserve dataset size for EDA and baseline recommendations
moviedata["Genre"] = moviedata["Genre"].fillna("unknown")
print("Filled missing genres with 'unknown'")
moviedata['Rating'] = moviedata['Rating'].fillna(moviedata['Rating'].mean())
print("Filled missing rating with mean value")
moviedata["Genre"] = (
    moviedata["Genre"]
    .str.lower()        # drama, action
    .str.strip()        # remove leading/trailing spaces
    .str.replace(" ", "", regex=False) # remove spaces inside
)
# to remove the movie with duplicate names
moviedata = moviedata.drop_duplicates(subset=["Movie_Name"])
#no cleaning required as there are no null values
# -------- EDA: Number of movies per genre --------

# Split genres and explode into separate rows
genre_df = moviedata.copy()
genre_df["Genre"] = genre_df["Genre"].str.split("|")
genre_df = genre_df.explode("Genre")

# Count movies per genre
genre_counts = genre_df["Genre"].value_counts()
plt.figure(figsize=(10, 5))
genre_counts.plot(kind="bar")
plt.title("Number of Movies per Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- EDA: Number of movies per rating --------
rating_counts = (
    moviedata["Rating"]
    .round(1)
    .value_counts()
    .sort_index()
)
plt.figure(figsize=(8, 5))
rating_counts.plot(kind="bar")
plt.title("Number of Movies per Rating")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# -------- Recommendation System --------

#dropping the unnamed column as it is not required for our analysis
moviedata = moviedata.drop(columns=["Unnamed: 0"], errors="ignore")
#setting our user_id for which we want recommendations
user_id = 1

user_data = moviedata[moviedata["User_Id"] == user_id]

tfidf_rec = recommend_tfidf(user_id, moviedata)
transformer_rec = recommend_transformer(user_id, moviedata)

tfidf_score = precision_at_k(tfidf_rec, user_data)
transformer_score = precision_at_k(transformer_rec, user_data)

print("TF-IDF Precision@10:", tfidf_score)
print("Transformer Precision@10:", transformer_score)


models = ["TF-IDF Recommender", "Sentence Transformer"]
scores = [tfidf_score, transformer_score]

plt.figure()

bars = plt.bar(models, scores)

plt.xlabel("Recommendation Method")
plt.ylabel("Precision@10")
plt.title("TF-IDF vs Transformer Recommendation Performance")

for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

plt.ylim(0, 1)
plt.show()


def recommend_movies_by_genre(moviedata, tfidf, genre_matrix, top_n=10):

    df = moviedata.copy()

    user_genres_input = input(
        "Enter preferred genres (comma-separated, e.g. action,drama,comedy): "
    )

    user_genres = (
        user_genres_input
        .lower()
        .replace(" ", "")
        .split(",")
    )

    user_genre_text = "|".join(user_genres)
    user_vector = tfidf.transform([user_genre_text])

    similarity_scores = cosine_similarity(
        user_vector,
        genre_matrix
    ).flatten()

    df["genre_similarity"] = similarity_scores
    df["rating_norm"] = df["Rating"] / 5

    df["final_score"] = (
        0.7 * df["genre_similarity"] +
        0.3 * df["rating_norm"]
    )

    recommendations = (
        df
        .sort_values(by="final_score", ascending=False)
        .head(top_n)
    )

    print("\n Genre-Based (User Input) Recommendations:\n")
    print(
        recommendations[
            ["Movie_Name", "Genre", "Rating", "final_score"]
        ].reset_index(drop=True)
    )
  
    
# Prepare TF-IDF vectors for genre-based recommendation
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
genre_matrix = tfidf.fit_transform(moviedata["Genre"])

try:
    top_n = int(input("How many movie recommendations do you want? "))
except:
    print("Invalid input. Defaulting to 10 recommendations.")
    top_n = 10

recommend_movies_by_genre(
    moviedata=moviedata,
    tfidf=tfidf,
    genre_matrix=genre_matrix,
    top_n=top_n
)

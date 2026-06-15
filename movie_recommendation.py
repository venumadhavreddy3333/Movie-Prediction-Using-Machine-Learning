import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')

# Features used
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies[feature] = movies[feature].fillna('')

# Combine features
combined_features = (
    movies['genres'] + ' ' +
    movies['keywords'] + ' ' +
    movies['tagline'] + ' ' +
    movies['cast'] + ' ' +
    movies['director']
)

# Convert text to vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Similarity scores
similarity = cosine_similarity(feature_vectors)

# User input
movie_name = input("Enter your favourite movie: ")

movie_titles = movies['title'].tolist()

find_close_match = get_close_matches(movie_name, movie_titles)

close_match = find_close_match[0]

index_of_movie = movies[movies.title == close_match].index[0]

similarity_score = list(enumerate(similarity[index_of_movie]))

sorted_movies = sorted(
    similarity_score,
    key=lambda x: x[1],
    reverse=True
)

print("\nMovies suggested for you:\n")

i = 1

for movie in sorted_movies:
    index = movie[0]
    title = movies.iloc[index]['title']

    if i <= 20:
        print(i, ".", title)
        i += 1

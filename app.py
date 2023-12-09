from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the ratings dataset
dataset_path = 'ratings1.csv'
df = pd.read_csv(dataset_path)

# Pivot the ratings dataset to get a user-movie matrix
user_movie_ratings = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(user_movie_ratings.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# Function to recommend movies based on ratings
def recommend_movies(movie_id):
    sim_scores = cosine_sim_df[movie_id]
    sim_scores = sim_scores.sort_values(ascending=False)
    recommendations = sim_scores.index[1:6].tolist()
    return recommendations

@app.route('/')
def index():
    return render_template('index.html', movie_ids=user_movie_ratings.columns.tolist())

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_id = int(request.form.get('movie_id'))
    recommendations = recommend_movies(movie_id)
    return render_template('index.html', movie_id=movie_id, recommendations=recommendations, movie_ids=user_movie_ratings.columns.tolist())

if __name__ == '__main__':
    app.run(debug=True)

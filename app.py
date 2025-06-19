from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import pickle
from scipy.sparse import load_npz

app = Flask(__name__, template_folder='.')

# Chargement unique des modèles et données
def load_recommendation_model():
    return {
        "user_encoder": joblib.load('user_encoder.pkl'),
        "movie_encoder": joblib.load('movie_encoder.pkl'),
        "user_movie_sparse": load_npz('user_movie_sparse.npz'),
        "user_similarity": np.load('user_similarity.npy'),
        "movie_index_to_id": pickle.load(open('movie_index_to_id.pkl', 'rb')),
        "movies_df": pd.read_csv('movies.csv'),
        "links_df": pd.read_csv('links.csv')
    }

model = load_recommendation_model()

# Prédiction de la note d'un film pour un utilisateur donné
def predict_rating(user_id, movie_id, user_movie_sparse, user_similarity, user_encoder, movie_encoder):
    try:
        user_idx = user_encoder.transform([user_id])[0]
        movie_idx = movie_encoder.transform([movie_id])[0]
    except:
        return None

    sim_scores = user_similarity[user_idx]
    sim_scores[user_idx] = 0  # éviter l'auto-similarité

    movie_ratings = user_movie_sparse[:, movie_idx].toarray().flatten()
    mask = movie_ratings > 0
    if np.sum(mask) == 0:
        return None

    weighted_sum = np.dot(sim_scores[mask], movie_ratings[mask])
    sum_sim = np.sum(sim_scores[mask])
    if sum_sim == 0:
        return None

    return weighted_sum / sum_sim

# Top K prédictions pour un utilisateur donné
def top_k_predictions(user_id, user_movie_sparse, user_similarity, user_encoder, movie_encoder, movie_index_to_id, movies_df, links_df, k=10):
    try:
        user_idx = user_encoder.transform([user_id])[0]
    except:
        return pd.DataFrame()

    user_seen_movies = user_movie_sparse[user_idx].toarray().flatten()
    unseen_movie_indices = np.where(user_seen_movies == 0)[0]

    movie_ids = []
    predicted_ratings = []

    for movie_idx in unseen_movie_indices:
        movie_id = movie_index_to_id[movie_idx]
        pred = predict_rating(user_id, movie_id, user_movie_sparse, user_similarity, user_encoder, movie_encoder)
        if pred is not None:
            movie_ids.append(movie_id)
            predicted_ratings.append(pred)

    pred_df = pd.DataFrame({
        'movieId': movie_ids,
        'predicted_rating': predicted_ratings
    })

    # Merge avec titre et genres
    pred_df = pred_df.merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    pred_df = pred_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

    # Générer l'URL TMDb
    pred_df['tmdb_url'] = pred_df['tmdbId'].apply(lambda x: f"https://www.themoviedb.org/movie/{int(x)}" if pd.notna(x) else None)

    pred_df = pred_df.sort_values(by='predicted_rating', ascending=False).head(k).reset_index(drop=True)

    return pred_df

# Page principale
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            k = int(request.form['k'])

            results_df = top_k_predictions(
                user_id=user_id,
                **model,
                k=k
            )

            if results_df.empty:
                return render_template('index.html', error="⚠️ Utilisateur inconnu ou aucune recommandation disponible.")

            # Inclure le lien vers TMDb
            results = results_df[['title', 'predicted_rating', 'genres', 'tmdb_url']].to_dict(orient='records')
            return render_template('index.html', results=results, user_id=user_id)

        except Exception as e:
            return render_template('index.html', error=f"❌ Erreur : {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

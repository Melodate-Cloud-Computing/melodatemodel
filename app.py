from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from collections import OrderedDict
from dotenv import load_dotenv
import joblib
import json
import os
import pandas as pd
import numpy as np

# Flask app
app = Flask(__name__)

# Load model dictionary
model_dict = joblib.load('melodate_recommender.joblib')
preprocessor = model_dict['preprocessor']
kmeans_model = model_dict['kmeans_model']

# Define columns
cat_columns = ['religion', 'smokes', 'drinks', 'genre', 'music_vibe', 'music_decade', 'listening_frequency', 'concert']
num_columns = ['age', 'height']

general_pref = [
    'religion_buddhism', 'religion_confucianism', 'religion_hinduism', 'religion_islam',
    'religion_protestant christianity', 'religion_roman catholicism',
    'age', 'height',
    'smokes_yes', 'smokes_no', 'drinks_yes', 'drinks_no'
]

music_pref = [
    # Include all your music preference features as in the original code
    'genre_ballad', 'genre_blues', 'genre_classical', 'genre_country', 'genre_dangdut',
    'genre_edm', 'genre_edm (electronic dance music)', 'genre_hip-hop/rap', 'genre_indie/alternative',
    'genre_j-pop', 'genre_jazz', 'genre_k-pop', 'genre_metal', 'genre_pop', 'genre_pop indonesia',
    'genre_punk', 'genre_r&b', 'genre_reggae', 'genre_rock', 'genre_soul', 'genre_traditional & folk music',

    'music_vibe_dark and intense', 'music_vibe_emotional and deep', 'music_vibe_relaxing and chill',
    'music_vibe_romantic and smooth', 'music_vibe_upbeat and energetic',

    'music_decade_1970s', 'music_decade_1980s', 'music_decade_1990s', 'music_decade_2000s',
    'music_decade_2010s', 'music_decade_2020s',

    'listening_frequency_frequently', 'listening_frequency_never', 'listening_frequency_occasionally',
    'listening_frequency_only in specific situations', 'listening_frequency_rarely',

    'concert_no, i prefer not to attend concerts', 'concert_sometimes, depending on the artist or event', 'concert_yes, i love attending concerts'
]

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Secure DATABASE_URL
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables")

def fetch_user_data():
    """Fetch data from PostgreSQL users table."""
    try:
        engine = create_engine(DATABASE_URL)

        query = """
        SELECT 
        "user", "firstName", "date_of_birth", "gender", "status", 
        "education", "hobby", "mbti", "love_language",
        "age", "height", "religion", "smokes", "drinks",
        "genre", "music_vibe", "music_decade", "listening_frequency", "concert", "location", "biodata",
        "profile_picture_url_1", "profile_picture_url_2", 
        "profile_picture_url_3", "profile_picture_url_4", 
        "profile_picture_url_5", "profile_picture_url_6",
        "topArtistImage1", "topArtistImage2", "topArtistImage3",
        "topArtistName1", "topArtistName2", "topArtistName3",
        "topTrackImage1", "topTrackImage2", "topTrackImage3", "topTrackImage4", "topTrackImage5",
        "topTrackTitle1", "topTrackTitle2", "topTrackTitle3", "topTrackTitle4", "topTrackTitle5",
        "topTrackArtist1", "topTrackArtist2", "topTrackArtist3", "topTrackArtist4", "topTrackArtist5"
        FROM users;
        """

        user_data = pd.read_sql_query(query, engine)
        
        # Tambahkan prefix "user" ke kolom 'user'
        user_data['user'] = 'user' + user_data['user'].astype(str)
        return user_data
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None

def find_similar_users(userid, preprocessed_df, user_data, top_n, user_gender):
    match_details = []

    # Ambil data pengguna target
    try:
        user_vector = preprocessed_df.loc[userid].values.reshape(1, -1)
        target_gender = user_data[user_data['user'] == userid]['gender'].values[0].lower()
    except KeyError:
        return None  # User not found

    # Hitung kemiripan kosinus dengan semua pengguna
    similarities = cosine_similarity(user_vector, preprocessed_df.values).flatten()

    # Urutkan hasil berdasarkan kemiripan, kecuali diri sendiri
    similar_indices = similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if preprocessed_df.index[i] != userid][:top_n]  # Ganti dari username ke userid

    # Ambil detail pengguna serupa
    for index in similar_indices:
        matched_user = preprocessed_df.index[index]
        similarity_score = similarities[index]
        user_info = user_data[user_data['user'] == matched_user].iloc[0]

        # Filter berdasarkan gender
        matched_gender = user_info['gender'].lower()
        if matched_gender == target_gender:
            continue

        match_details.append({
            'user_id': int(matched_user.replace("user", "")),
            'similarity_score': float(similarity_score),
        })

    return match_details

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    userid = request.args.get('userid')
    top_n = int(request.args.get('top_n', 5))

    if not userid:
        return jsonify({'error': 'Userid parameter is required'}), 400

    # Fetch user data
    user_data = fetch_user_data()
    if user_data is None:
        return jsonify({'error': 'Unable to fetch user data'}), 500

    # Preprocess user data
    for col in cat_columns:
        user_data[col] = user_data[col].str.lower()

    preprocessed_data = preprocessor.transform(user_data)

    onehot_categories = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_columns)
    feature_names = num_columns + list(onehot_categories)
    preprocessed_df = pd.DataFrame(
        preprocessed_data,
        columns=feature_names,
        index=user_data['user']
    )

    clusters = kmeans_model.predict(preprocessed_df[general_pref])
    preprocessed_df['cluster_label'] = clusters

    # Ambil gender pengguna target
    try:
        user_gender = user_data[user_data['user'] == userid]['gender'].iloc[0]
    except IndexError:
        return jsonify({'error': f"User '{userid}' not found"}), 404

    matches = find_similar_users(userid, preprocessed_df, user_data, top_n, user_gender)

    if matches is None:
        return jsonify({'error': f"User '{userid}' not found"}), 404

    if len(matches) == 0:
        return jsonify({'message': f"No similar users found for user '{userid}'"}), 200

    # Extract user IDs and fetch complete user details
    user_ids = [match['user_id'] for match in matches]
    user_details = fetch_user_details(user_ids)
    if user_details is None:
        return jsonify({'error': 'Unable to fetch user details'}), 500

    # Combine similarity scores with user details
    response_data = []
    for match in matches:
        user_data = user_details[user_details['user_id'] == match['user_id']].to_dict('records')
        if user_data:
            user_info = user_data[0]  # Should only have one matching row
            user_info['similarity_score'] = match['similarity_score']

            # Convert user_info to OrderedDict for maintaining the order
            ordered_user_info = OrderedDict()
            for column in user_details.columns:
                ordered_user_info[column] = user_info.get(column)
            ordered_user_info['similarity_score'] = user_info['similarity_score']

            response_data.append(ordered_user_info)

    return jsonify(response_data), 200

def fetch_user_details(user_ids):
    """Fetch complete user data from the database based on user IDs."""
    try:
        engine = create_engine(DATABASE_URL)
        query = f"""
            SELECT 
            "user" AS user_id, "firstName", "date_of_birth", "gender", "status", 
            "education", "hobby", "mbti", "love_language",
            "age", "height", "religion", "smokes", "drinks",
            "genre", "music_vibe", "music_decade", "listening_frequency", "concert", "location", "biodata",
            "profile_picture_url_1", "profile_picture_url_2", 
            "profile_picture_url_3", "profile_picture_url_4", 
            "profile_picture_url_5", "profile_picture_url_6",
            "topArtistImage1", "topArtistImage2", "topArtistImage3",
            "topArtistName1", "topArtistName2", "topArtistName3",
            "topTrackImage1", "topTrackImage2", "topTrackImage3", "topTrackImage4", "topTrackImage5",
            "topTrackTitle1", "topTrackTitle2", "topTrackTitle3", "topTrackTitle4", "topTrackTitle5",
            "topTrackArtist1", "topTrackArtist2", "topTrackArtist3", "topTrackArtist4", "topTrackArtist5"
            FROM users
            WHERE "user" IN ({','.join(map(str, user_ids))});
            """
        user_details = pd.read_sql_query(query, engine)

        # Convert user_id to integer for consistency
        user_details['user_id'] = user_details['user_id'].astype(int)
        return user_details
    except Exception as e:
        print(f"Error fetching user details: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
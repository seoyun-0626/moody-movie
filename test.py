from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import random
import requests

app = Flask(__name__)
CORS(app)

# ==========================
# 감정 분석 모델 로드
# ==========================
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("감정 분석 모델 로드 완료")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()

# ==========================
# 감정 → TMDB 장르 매핑
# ==========================
emotion_to_genre = {
    "분노": [28, 53, 80],
    "불안": [53, 18, 878],
    "스트레스": [35, 10402, 10751],
    "슬픔": [18, 10749, 10751],
    "행복": [12, 35, 16, 10751],
    "심심": [14, 878, 12],
    "탐구": [99, 36, 18]
}

TMDB_API_KEY = "8cde0962eca9041f7345e9c7ab7a4b7f"  # 본인 API 키 입력
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w200"

def get_genre_by_emotion(emotion):
    genres = emotion_to_genre.get(emotion, [18])
    return random.choice(genres) if isinstance(genres, list) else genres

def get_movies_by_genre(genre_id):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&sort_by=popularity.desc"
    res = requests.get(url)
    data = res.json()
    movies = []
    for item in data.get("results", [])[:5]:  # 상위 5개만
        title = item.get("title")
        poster_path = item.get("poster_path")
        poster_url = f"{TMDB_IMAGE_URL}{poster_path}" if poster_path else ""
        movies.append({"title": title, "poster": poster_url})
    return movies

@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    try:
        data = request.get_json()
        user_input = data.get("emotion", "").strip()
        if not user_input:
            return jsonify({"reply": "감정을 입력해 주세요 😢"}), 400

        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)

        return jsonify({
            "emotion": predicted_emotion,
            "movies": movies
        })

    except Exception as e:
        print(e)
        return jsonify({"reply": "서버에서 오류가 발생했어요 😢"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
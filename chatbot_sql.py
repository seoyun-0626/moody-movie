from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import random
import requests
import mysql.connector
import datetime

# ============================================================
# 1️⃣ Flask 기본 설정
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# 2️⃣ 감정 분석 모델 로드
# ============================================================
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("✅ 감정 분석 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# ============================================================
# 3️⃣ MySQL 연결 설정
# ============================================================
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="moodymovie"
    )

# ============================================================
# 4️⃣ 감정 → TMDB 장르 매핑
# ============================================================
emotion_to_genre = {
    "분노": [28, 80, 53, 27, 9648],
    "불안": [53, 9648, 18, 878],
    "스트레스": [35, 10402, 10751, 16],
    "슬픔": [18, 10749, 10751, 99],
    "행복": [12, 35, 16, 10751, 10402],
    "심심": [14, 878, 12, 10751],
    "탐구": [99, 36, 18, 37]
}

TMDB_API_KEY = "8cde0962eca9041f7345e9c7ab7a4b7f"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w200"

def get_genre_by_emotion(emotion):
    genres = emotion_to_genre.get(emotion, [18])
    return random.choice(genres)

# ============================================================
# 5️⃣ TMDB API에서 영화 추천
# ============================================================
def get_movies_by_genre(genre_id):
    url = f"https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "with_genres": genre_id,
        "sort_by": "popularity.desc",
        "language": "ko-KR",
        "page": 1
    }

    res = requests.get(url, params=params)
    data = res.json()

    movies = []
    for item in data.get("results", [])[:5]:  # 상위 5개만
        title = item.get("title")
        poster_path = item.get("poster_path")
        poster_url = f"{TMDB_IMAGE_URL}{poster_path}" if poster_path else ""
        movies.append({"title": title, "poster": poster_url})
    return movies

# ============================================================
# 6️⃣ Flask 엔드포인트 (/emotion)
# ============================================================
@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    try:
        data = request.get_json()
        user_input = data.get("emotion", "").strip()

        if not user_input:
            return jsonify({"reply": "감정을 입력해 주세요 😢"}), 400

        # ① 감정 분석
        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        # ② 감정에 맞는 영화 추천
        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)

        # ③ MySQL에 저장
        try:
            conn = get_connection()
            cursor = conn.cursor()

            insert_sql = """
            INSERT INTO movies_emotions (rep_emotion, sub_emotion, genre, movie, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """

            # 영화 5개 중 하나를 랜덤으로 DB에 저장
            for m in movies:
                values = (
                    predicted_emotion,  # 대표 감정
                    user_input,         # 사용자가 입력한 문장 (세부 감정)
                    str(genre_id),      # 장르 ID
                    m["title"],         # 영화 이름
                    datetime.datetime.now()
                )
                cursor.execute(insert_sql, values)

            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ DB 저장 완료 ({predicted_emotion})")

        except Exception as db_err:
            print(f"❌ DB 저장 중 오류: {db_err}")

        # ④ 결과 반환
        return jsonify({
            "emotion": predicted_emotion,
            "movies": movies
        })

    except Exception as e:
        print(f"❌ 서버 오류: {e}")
        return jsonify({"reply": "서버에서 오류가 발생했어요 😢"}), 500

# ============================================================
# 7️⃣ Flask 실행
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

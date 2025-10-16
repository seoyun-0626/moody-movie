from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import random
from movie_api import get_movies_by_genre  # 이미 있는 함수

# ==========================
# Flask 설정
# ==========================
app = Flask(__name__)
CORS(app)  # 클라이언트 CORS 허용

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
# 감정 → 장르 매핑
# ==========================
emotion_to_genre = {
    "슬픔": [35,10749, 10751, 18],
    "불안": [16, 10402],
    "스트레스": [16, 10402],
    "분노": [80, 27, 53, 9648],
    "행복": [28, 12],
    "심심": [14, 878],
    "탐구": [99, 36],
}

def get_genre_by_emotion(emotion):
    genres = emotion_to_genre.get(emotion, [18])
    if isinstance(genres, list):
        return random.choice(genres)
    return genres

# ==========================
# 감정 POST 엔드포인트
# ==========================
@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    try:
        data = request.get_json()
        user_input = data.get("emotion", "").strip()
        
        if not user_input:
            return jsonify({"reply": "감정을 입력해 주세요 😢"}), 400

        # 감정 예측
        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        # 추천 장르
        genre_id = get_genre_by_emotion(predicted_emotion)

        # 영화 추천
        try:
            movies = get_movies_by_genre(genre_id)
            movie_list = "\n".join(movies)
            reply = f"감정 분석 결과: {predicted_emotion}\n추천 영화 리스트:\n{movie_list}"
        except Exception as e:
            reply = f"영화 추천 중 오류 발생: {e}"

        return jsonify({"reply": reply})

    except Exception as e:
        print(e)
        return jsonify({"reply": "서버에서 오류가 발생했어요 😢"}), 500

# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

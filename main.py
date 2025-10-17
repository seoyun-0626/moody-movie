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
    # 대표감정 모델 불러오기
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    # 세부감정 모델 추가 로드
    sub_models = pickle.load(open("sub_models.pkl", "rb"))
    sub_vectorizers = pickle.load(open("sub_vectorizers.pkl", "rb"))

    print("감정 분석 모델 및 세부감정 모델 로드 완료")

except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()


# ==========================
# 감정 → 장르 매핑
# ==========================
emotion_to_genre = {
    "분노": [28, 80, 53, 27, 9648],        # 액션, 범죄, 스릴러, 공포, 미스터리
    "불안": [53, 9648, 18, 878],           # 스릴러, 미스터리, 드라마, SF
    "스트레스": [35, 10402, 10751, 16],     # 코미디, 음악, 가족, 애니메이션
    "슬픔": [18, 10749, 10751, 99],         # 드라마, 로맨스, 가족, 다큐멘터리
    "행복": [12, 35, 16, 10751, 10402],     # 모험, 코미디, 애니메이션, 가족, 음악
    "심심": [14, 878, 12, 10751],           # 판타지, SF, 모험, 가족
    "탐구": [99, 36, 18, 37]                # 다큐멘터리, 역사, 드라마, 서부
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
            return jsonify({"reply": "감정을 입력해 주세요"}), 400

        # 1️⃣ 대표감정 예측
        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        # 2️⃣ 세부감정 예측
        if predicted_emotion in sub_models:
            sub_vec = sub_vectorizers[predicted_emotion]
            sub_model = sub_models[predicted_emotion]
            X_sub = sub_vec.transform([user_input])
            predicted_sub = sub_model.predict(X_sub)[0]
        else:
            predicted_sub = "세부감정 없음"

        # 3️⃣ 영화 추천
        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)

        # 4️⃣ 결과 반환
        return jsonify({
            "emotion": predicted_emotion,
            "sub_emotion": predicted_sub,
            "movies": movies
        })

    except Exception as e:
        print(e)
        return jsonify({"reply": "서버에서 오류가 발생했어요"}), 500
    



# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    print("오늘 기분이 어때?")
    user_input = input("→ ")

    # 1️⃣ 대표감정 예측
    X = vectorizer.transform([user_input])
    pred_emotion = model.predict(X)[0]

    # 2️⃣ 세부감정 예측
    if pred_emotion in sub_models:
        sub_vec = sub_vectorizers[pred_emotion]
        sub_model = sub_models[pred_emotion]
        X_sub = sub_vec.transform([user_input])
        pred_sub = sub_model.predict(X_sub)[0]
    else:
        pred_sub = "세부감정 없음"

    # 3️⃣ 콘솔 출력
    print(f"감정 분석 결과: {pred_emotion} / 세부감정: {pred_sub}")

    # 4️⃣ Flask 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


#====================================


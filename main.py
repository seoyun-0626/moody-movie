import pickle
import random
from movie_api import get_movies_by_genre

# ==========================
# 감정 분석 모델 불러오기
# ==========================
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("감정 분석 모델 로드 완료")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()

# ==========================
# 사용자 감정 입력
# ==========================
user_input = input("\n오늘 기분이 어때? : ").strip()
if not user_input:
    print("입력이 비었습니다. 프로그램을 종료합니다.")
    exit()

# ==========================
# 감정 예측
# ==========================
X = vectorizer.transform([user_input])
predicted_emotion = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]  # 확률값 전체

# 감정별 확률 보기
X = vectorizer.transform([user_input])
predicted_emotion = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]  

print(f"\n감정 분석 결과: {predicted_emotion}\n")
print("=== 감정별 확률 분포 ===")
for emotion, prob in zip(model.classes_, probabilities):
    print(f"{emotion}: {prob*100:.2f}%")
# ==========================
# 감정 → 장르 매핑
# ==========================
emotion_to_genre = {
    "슬픔": 35,
    "불안": [16, 10402],
    "걱정": [16, 10402],
    "스트레스": [16, 10402],
    "피로": [16, 10402],
    "분노": [80, 27, 53, 9648],
    "외로움": [10749, 10751, 18],
    "결핍": [10749, 10751, 18],
    "행복": [28, 12],
    "설렘": [28, 12],
    "심심": [14, 878],
    "탐구": [99, 36],
    "호기심": [99, 36]
}

def get_genre_by_emotion(emotion):
    genres = emotion_to_genre.get(emotion, [18])
    if isinstance(genres, list):
        return random.choice(genres)
    return genres

genre_id = get_genre_by_emotion(predicted_emotion)
print(f"\n추천 장르 ID: {genre_id}")

# ==========================
# 영화 추천 출력
# ==========================
print("\n=== 추천 영화 리스트 ===\n")
try:
    print(get_movies_by_genre(genre_id))
except Exception as e:
    print(f"영화 추천 중 오류 발생: {e}")

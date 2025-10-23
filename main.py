import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
import pickle, random, sys
from openai import OpenAI
from movie_api import get_movies_by_genre, get_movie_rating
from dotenv import load_dotenv
import os
import pymysql

# ==========================
# ✅ 절대경로 기본 설정
# ==========================
BASE_DIR = r"C:\ai-project\moody-movie"
MODEL_DIR = os.path.join(BASE_DIR, "models")
ENV_PATH = r"C:\ai-project\.env"

# ==========================
# ✅ 환경 변수 로드
# ==========================
load_dotenv(dotenv_path=ENV_PATH)
api_key = os.getenv("OPENAI_API_KEY")
print(f"🔑 OpenAI Key 불러옴: {api_key[:10]}..." if api_key else "❌ OpenAI Key 불러오기 실패")

client = OpenAI(api_key=api_key)

# ==========================
# ✅ Flask 설정
# ==========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["JSON_AS_ASCII"] = False


class UTF8JSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(obj, **kwargs)

    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)


app.json = UTF8JSONProvider(app)
sys.stdout.reconfigure(encoding="utf-8")

# ==========================
# ✅ 감정 분석 모델 로드
# ==========================
try:
    model = pickle.load(open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb"))
    sub_model = pickle.load(open(os.path.join(MODEL_DIR, "emotion_sub_model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))
    sub_vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "sub_vectorizers.pkl"), "rb"))
    print("✅ 감정 분석 모델 및 세부감정 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# ==========================
# ✅ 감정 → 장르 매핑
# ==========================
emotion_to_genre = {
    "분노": [28, 80, 53, 27, 9648],
    "불안": [53, 9648, 18, 878],
    "스트레스": [35, 10402, 10751, 16],
    "슬픔": [18, 10749, 10751, 99],
    "행복": [12, 35, 16, 10751, 10402],
    "심심": [14, 878, 12, 10751],
    "탐구": [99, 36, 18, 37],
}


def get_genre_by_emotion(emotion):
    genres = emotion_to_genre.get(emotion, [18])
    return random.choice(genres)

# ==========================
# ✅ /emotion 엔드포인트
# ==========================
@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    try:
        data = request.get_json()
        user_input = data.get("emotion", "").strip()

        if not user_input:
            return jsonify({"reply": "감정을 입력해 주세요"}), 400

        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        try:
            X_sub = sub_vectorizer.transform([user_input])
            predicted_sub = sub_model.predict(X_sub)[0]
        except Exception:
            predicted_sub = "세부감정 없음"

        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)

        return jsonify({
            "emotion": predicted_emotion,
            "sub_emotion": predicted_sub,
            "movies": movies
        })

    except Exception as e:
        print("❌ /emotion 오류:", e)
        return jsonify({"reply": "서버에서 오류가 발생했어요"}), 500

# ==========================
# ✅ /chat 엔드포인트 (3턴 대화)
# ==========================
conversation_history = []
recommended_movies_memory = []

@app.route("/chat", methods=["POST"])
def chat_turn():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        turn = data.get("turn", 1)
        gpt_reply = ""

        # turn 처리
        if isinstance(turn, str):
            if turn == "after_recommend":
                turn_type = "after_recommend"
            else:
                try:
                    turn = int(turn)
                    turn_type = "normal"
                except ValueError:
                    turn_type = "normal"
        else:
            turn_type = "normal"

        global conversation_history
        conversation_history.append({"role": "user", "content": user_msg})

        # 1~2턴
        if turn_type == "normal" and turn < 3:
            system_prompt = (
                "너는 감정상담 친구야. "
                "사용자의 말을 따뜻하게 공감하면서 짧게 답하고, "
                "반드시 마지막에 질문을 하나 덧붙여."
                "사용자와 너가 한말을 모두 기억해"
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            gpt_reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": gpt_reply})
            return jsonify({"reply": gpt_reply, "final": False})

        # 추천 이후 대화
        elif turn_type == "after_recommend":
            try:
                followup_prompt = (
                    "너는 감정 기반 영화 추천 친구야. "
                    "지금까지의 대화와 추천한 영화들을 전부 기억하고 있어. "
                    "사용자가 영화 제목 일부나 번호(1, 2, 3번)만 말해도 어떤 영화를 의미하는지 알아들어야 해. "
                    "‘그거’, ‘이거’, ‘마지막꺼’, ‘첫번째꺼’ 같은 표현도 이해해. "
                    "사용자가 평점이나 줄거리, 배우, 분위기 등을 물으면 자연스럽게 설명해줘. "
                    "응답은 짧고 자연스럽게, 친구처럼 따뜻하게 대화해."
                    "동일한 영화의 평점을 여러번 얘기하지 마"
                    "바로 전 대화와 자연스럽게 이어지게 대화해"
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": followup_prompt},
                        *conversation_history,
                        {"role": "user", "content": user_msg},
                    ],
                )

                gpt_reply = response.choices[0].message.content.strip()

                lower_msg = user_msg.lower()
                if any(word in lower_msg for word in ["평점", "점수", "몇점", "점"]):
                    movie_titles = recommended_movies_memory
                    candidate = None
                    for title in movie_titles:
                        if title.lower().replace(" ", "") in lower_msg:
                            candidate = title.strip()
                            break
                    if not candidate and movie_titles:
                        candidate = movie_titles[0].strip()

                    if candidate:
                        result = get_movie_rating(candidate)
                        if result:
                            gpt_reply += f"\n🎬 '{result['title']}'의 TMDB 평점은 {result['rating']}점이에요!"
                        else:
                            gpt_reply += f"\n'{candidate}'의 평점을 찾지 못했어요 😢"

                return jsonify({"reply": gpt_reply})
            except Exception as e:
                print("❌ after_recommend 오류:", e)
                return jsonify({"reply": "영화 정보를 불러오는 중 오류가 발생했어요 😢"}), 500

        # 요약 + 감정 분석 + 추천
        summary_prompt = f"""
        다음은 사용자와 감정상담 챗봇의 3턴 대화야:
        {conversation_history}
        사용자의 감정 상태를 한 문장으로 요약해줘.
        예: '요즘 마음이 공허한가 봐.', '피곤해서 기운이 없는 상태야.'
        """
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 감정을 따뜻하게 요약하는 친구야."},
                {"role": "user", "content": summary_prompt},
            ],
        )
        summary_text = summary_response.choices[0].message.content.strip()
        print("🧠 대화 요약문:", summary_text.encode("utf-8", "ignore").decode("utf-8"))

        X = vectorizer.transform([summary_text])
        predicted_emotion = model.predict(X)[0]
        try:
            X_sub = sub_vectorizer.transform([summary_text])
            predicted_sub = sub_model.predict(X_sub)[0]
        except Exception as e:
            print("세부감정 분석 오류:", e)
            predicted_sub = "세부감정 없음"

        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)
        movie_titles = [m["title"] for m in movies if isinstance(m, dict)]
        conversation_history.append({
            "role": "assistant",
            "content": f"추천 영화 목록은 {', '.join(movie_titles)}야."
        })

        conversation_history.append({
            "role": "assistant",
            "content": "내가 추천해준 영화가 마음에 들어? 🎬"
        })

        return jsonify({
            "reply": gpt_reply,
            "summary": summary_text,
            "final": True,
            "emotion": predicted_emotion,
            "sub_emotion": predicted_sub,
            "movies": movies
        })

    except Exception as e:
        print("❌ /chat 오류:", e)
        return jsonify({"reply": "서버 오류 발생"}), 500

# ==========================
# ✅ HTML 연결
# ==========================
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

# ==========================
# ✅ DB 연결 및 통계 API
# ==========================
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password=DB_PASSWORD,
        db="moodymovie",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

@app.route("/stats")
def get_stats():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT rep_emotion, COUNT(*) AS count
        FROM movies_emotions
        GROUP BY rep_emotion
        ORDER BY count DESC;
    """)
    result = cursor.fetchall()
    conn.close()
    return jsonify(result)

@app.route("/top10")
def get_top10_movies():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie, COUNT(*) AS count
        FROM movies_emotions
        GROUP BY movie
        ORDER BY count DESC
        LIMIT 10;
    """)
    result = cursor.fetchall()
    conn.close()
    return jsonify(result)

# ==========================
# ✅ 서버 실행
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)

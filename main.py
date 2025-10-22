import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
import pickle, random, sys
from openai import OpenAI
from movie_api import get_movies_by_genre  
from dotenv import load_dotenv
import os

# ✅ .env 파일 강제 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

api_key = os.getenv("OPENAI_API_KEY")
print(f"🔑 OpenAI Key 불러옴: {api_key[:10]}...")  # 키 확인용

# ✅ OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)

# ==========================
# Flask 설정
# ==========================
app = Flask(__name__)

# ✅ 모든 도메인에서 접근 허용
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['JSON_AS_ASCII'] = False  # ✅ jsonify가 한글을 ASCII로 변환하지 않도록 설정
class UTF8JSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(obj, **kwargs)
    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)
app.json = UTF8JSONProvider(app)

sys.stdout.reconfigure(encoding='utf-8')


# ==========================
# 감정 분석 모델 로드
# ==========================
try:
    model = pickle.load(open("models/emotion_model.pkl", "rb"))
    sub_model = pickle.load(open("models/emotion_sub_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    sub_vectorizer = pickle.load(open("models/sub_vectorizers.pkl", "rb"))
    print("✅ 감정 분석 모델 및 세부감정 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# ==========================
# 감정 → 장르 매핑
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
# ✅ /emotion 엔드포인트 (감정 분석 + 세부감정 + 영화 추천)
# ==========================
@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    try:
        data = request.get_json()
        user_input = data.get("emotion", "").strip()

        if not user_input:
            return jsonify({"reply": "감정을 입력해 주세요"}), 400

        # 대표 감정 예측
        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        # 세부 감정 예측
        try:
            X_sub = sub_vectorizer.transform([user_input])
            predicted_sub = sub_model.predict(X_sub)[0]
        except Exception:
            predicted_sub = "세부감정 없음"

        # 감정에 맞는 장르 → 영화 추천
        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)

        # 결과 반환
        return jsonify({
            "emotion": predicted_emotion,
            "sub_emotion": predicted_sub,
            "movies": movies
        })

    except Exception as e:
        print("❌ /emotion 오류:", e)
        return jsonify({"reply": "서버에서 오류가 발생했어요"}), 500


# ==========================
# ✅ 3턴 대화용 /chat 엔드포인트
# ==========================
conversation_history = []

@app.route("/chat", methods=["POST"])
def chat_turn():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        turn = data.get("turn", 1)
        gpt_reply = ""

        # ✅ 문자열과 숫자 구분
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

        # 1~2턴 대화
        if turn_type == "normal" and turn < 3:
            system_prompt = (
                "너는 감정상담 친구야. "
                "사용자의 말을 따뜻하게 공감하면서 짧게 답하고, "
                "반드시 마지막에 질문을 하나 덧붙여."
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

        # ✅ 추천 이후 대화
        elif turn_type == "after_recommend":
            followup_prompt = (
                "너는 감정 기반 영화 추천을 이어가는 친구야. "
                "너는 영화에 대한 얘기만 가능하고 다른질문은 답변 어렵다고 해야해."
                "방금 네가 추천한 영화 목록은 이미 대화 기록에 저장되어 있어. "
                "사용자가 그 영화들 중 하나를 언급하면, "
                "그 영화에 대해 이미 본 것처럼 말하지 말고, "
                "‘그 영화는 액션이 강해서 너한테 잘 어울릴 수도 있겠다’ 같은 식으로 가볍게 반응해. "
                "또한 사용자가 영화의 평점이나 정보를 물으면, "
                "‘TMDB 기준 평점을 확인해볼게’ 같은 식으로 자연스럽게 이어가."
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": followup_prompt},
                    *conversation_history,  # ✅ 이전 대화 맥락 추가
                    {"role": "user", "content": user_msg},
                ],
            )
            gpt_reply = response.choices[0].message.content.strip()
            return jsonify({"reply": gpt_reply})

        # ✅ 3턴 대화 → 요약 + 감정분석 + 추천
        else:
            closing_prompt = (
                "너는 감정상담 친구야. "
                "이제 대화를 마무리하면서 짧게 공감만 하고 질문은 하지 마."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": closing_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            gpt_reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": gpt_reply})

            # ✅ 요약문 생성
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
            print("🧠 대화 요약문:", summary_text.encode('utf-8', 'ignore').decode('utf-8'))

            # ✅ 감정 분석 실행
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
            
            # ✅ GPT가 다음 턴(after_recommend)에서 영화 제목들을 기억할 수 있게 저장
            movie_titles = [m["title"] for m in movies if isinstance(m, dict)]
            conversation_history.append({
                "role": "assistant",
                "content": f"추천 영화 목록은 {', '.join(movie_titles)}야."
                })

        
            # ✅ 대화 초기화
            # conversation_history = []

            return jsonify({
                "reply": gpt_reply,
                "summary": summary_text,
                "final": True,
                "emotion": predicted_emotion,
                "sub_emotion": predicted_sub,
                "movies": movies
            })

    # ✅ 이게 반드시 들어가야 함 (오류의 원인)
    except Exception as e:
        print("❌ /chat 오류:", e)
        return jsonify({"reply": "서버 오류 발생"}), 500


# ==========================
# HTML 연결
# ==========================
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ==========================
# DB 연결 및 통계 API 
# ==========================
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()  # .env 읽기
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

#많이 추천된 영화 탑10

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
# 서버 실행
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

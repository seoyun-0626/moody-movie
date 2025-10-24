# ==========================================================
# 🎬 Moody-Movie Flask 서버 (감정 기반 영화 추천 챗봇)
# ==========================================================
# 이 서버는 다음과 같은 역할을 수행함:
# ① 감정 분석 (대표감정 + 세부감정)
# ② 감정 기반 영화 추천
# ③ 감정상담 챗봇 대화 (3턴 구조)
# ④ DB 연결 (통계 및 인기 영화 조회)
# ==========================================================

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

# ==========================================================
# ✅ 1. 경로 설정 (절대경로)
# ==========================================================
BASE_DIR = r"C:\ai-project\moody-movie"             # 프로젝트 기본 경로
MODEL_DIR = os.path.join(BASE_DIR, "models")        # 모델 파일 폴더
ENV_PATH = r"C:\ai-project\.env"                    # 환경변수 파일 경로 (.env)

# ==========================================================
# ✅ 2. 환경 변수 로드 (.env 파일에서 키 불러오기)
# ==========================================================
load_dotenv(dotenv_path=ENV_PATH)
api_key = os.getenv("OPENAI_API_KEY")                # OpenAI API 키 불러오기
print(f"🔑 OpenAI Key 불러옴: {api_key[:10]}..." if api_key else "❌ OpenAI Key 불러오기 실패")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)

# ==========================================================
# ✅ 3. Flask 앱 기본 설정
# ==========================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})       # 모든 도메인에서 접근 허용 (CORS 문제 방지)
app.config["JSON_AS_ASCII"] = False                  # jsonify가 한글을 ASCII로 변환하지 않게 설정

# UTF-8 JSON 인코딩 보장용 커스텀 Provider
class UTF8JSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("ensure_ascii", False)     # 한글이 깨지지 않도록
        return json.dumps(obj, **kwargs)
    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)

app.json = UTF8JSONProvider(app)
sys.stdout.reconfigure(encoding="utf-8")             # 콘솔 출력 시 한글 깨짐 방지

# ==========================================================
# ✅ 4. 감정 분석 모델 로드
# ==========================================================
try:
    # 모델 파일 불러오기 (대표감정 + 세부감정)
    model = pickle.load(open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb"))
    sub_model = pickle.load(open(os.path.join(MODEL_DIR, "sub_models.pkl"), "rb"))          # ✅ 여기 수정
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))
    sub_vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "sub_vectorizers.pkl"), "rb")) # ✅ 여기 수정
    print("✅ 감정 분석 모델 및 세부감정 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()
    
# ==========================================================
# ✅ 5. 감정 → 영화 장르 매핑 테이블
# ==========================================================
emotion_to_genre = {
    "분노": [28, 80, 53, 27, 9648],         # 액션, 범죄, 스릴러
    "불안": [53, 9648, 18, 878],            # 스릴러, 미스터리, 드라마, SF
    "스트레스": [35, 10402, 10751, 16],     # 코미디, 음악, 가족, 애니메이션
    "슬픔": [18, 10749, 10751, 99],         # 드라마, 로맨스, 가족, 다큐
    "행복": [35, 16, 10751, 10402],         #  코미디, 애니, 가족, 음악
    "심심": [14, 878, 12, 10751,27],         # 판타지, SF, 모험, 가족
    "탐구": [99, 36, 18, 37],               # 다큐, 역사, 드라마, 서부극
}

def get_genre_by_emotion(emotion):
    """감정에 맞는 영화 장르 ID를 랜덤으로 선택"""
    genres = emotion_to_genre.get(emotion, [18])  # 기본값: 드라마(18)
    return random.choice(genres)

# ==========================================================
# ✅ 6. /emotion 엔드포인트
# ==========================================================
@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    """
    사용자가 입력한 문장에서 감정을 예측하고
    TMDB 장르에 맞는 영화 추천 리스트를 반환함.
    """
    try:
        data = request.get_json()
        user_input = data.get("emotion", "").strip()

        if not user_input:
            return jsonify({"reply": "감정을 입력해 주세요"}), 400

        # 대표감정 예측
        X = vectorizer.transform([user_input])
        predicted_emotion = model.predict(X)[0]

        # 해당 대표감정에 맞는 세부감정 모델 사용 (단일 모델 버전)
        try:
            X_sub = sub_vectorizer.transform([user_input])
            predicted_sub = sub_model.predict(X_sub)[0]
        except Exception as e:
            print("세부감정 분석 오류:", e)
            predicted_sub = "세부감정 없음"

        # 감정에 맞는 영화 장르 → 영화 추천
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

# ==========================================================
# ✅ 7. /chat 엔드포인트 (3턴 감정상담 + 추천 대화)
# ==========================================================
conversation_history = []          # 사용자와의 대화 내역 저장
recommended_movies_memory = []     # 추천 영화 기억용

@app.route("/chat", methods=["POST"])
def chat_turn():
    """
    사용자의 감정 대화를 3턴으로 구성:
    1~2턴: 공감형 대화
    3턴: 감정 요약 → 감정 분석 → 영화 추천
    이후: 영화 관련 대화 (평점, 줄거리 등)
    """
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        turn = data.get("turn", 1)
        gpt_reply = ""

        # turn 데이터 타입 정리 (문자 or 숫자)
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

        # -----------------------------------------------
        # 🧡 1~2턴: 감정상담 (공감형 대화)
        # -----------------------------------------------
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

        # -----------------------------------------------
        # 🎞️ 추천 이후의 대화 (영화 관련 질의응답)
        # -----------------------------------------------
        elif turn_type == "after_recommend":
            try:
                followup_prompt = (
                    "너는 감정 기반 영화 추천 친구야. "
                    "지금까지의 대화와 추천한 영화들을 전부 기억하고 있어. "
                    "사용자가 영화 제목 일부나 번호(1, 2, 3번)만 말해도 어떤 영화를 의미하는지 알아들어야 해. "
                    "‘그거’, ‘이거’, ‘마지막꺼’, ‘첫번째꺼’ 같은 표현도 이해해. "
                    "사용자가 평점이나 줄거리, 배우, 분위기 등을 물으면 자연스럽게 설명해줘. "
                    "응답은 짧고 자연스럽게, 친구처럼 따뜻하게 대화해."
                    "이미 설명했던 영화에 대해 다시 묻지 않는 이상, 새로운 표현으로 짧게 이어서 말해. "
                    "이미 한 말을 다시 하지 말고, 대화가 자연스럽게 다음 주제로 이어지게 말해."
                    "사용자가 추천해준 영화를 마음에 들어하지 않는다면 "
                    "같은 영화를 다시 언급하지 말고 새로운 영화를 제안해줘."
                    "사용자가 새로운 영화를 보겠다고 하면, 그 영화를 가장 최근 추천으로 기억해. "
                    "그 이후 평점이나 설명을 물으면 그 영화 기준으로 답해."                
                   
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

                # 사용자가 평점을 물어볼 경우 처리
                lower_msg = user_msg.lower()
                if any(word in lower_msg for word in ["평점", "점수", "몇점", "점"]):
                    movie_titles = recommended_movies_memory
                    candidate = None
                    # 대화 속 영화 제목 탐색
                    for title in movie_titles:
                        if title.lower().replace(" ", "") in lower_msg:
                            candidate = title.strip()
                            break
                    # 명시되지 않은 경우 첫 번째 영화로 대체
                    if not candidate and movie_titles:
                        candidate = movie_titles[0].strip()

                    # TMDB API로 평점 조회
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

        # -----------------------------------------------
        # 🧠 3턴: 요약 + 감정 분석 + 영화 추천
        # -----------------------------------------------
        recent_history = conversation_history[-6:]  # 최근 3턴(유저+챗봇)만 사용

        summary_prompt = f"""
        다음은 사용자와 감정상담 챗봇의 최근 대화야:
        {recent_history}
        사용자의 현재 감정 상태를 한 문장으로 요약해줘.
        예: '요즘 마음이 공허한가 봐.', '피곤해서 기운이 없는 상태야.'
        """

        # GPT로 감정 요약 생성
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 감정을 따뜻하게 요약하는 친구야."},
                {"role": "user", "content": summary_prompt},
            ],
        )
        summary_text = summary_response.choices[0].message.content.strip()
        print("🧠 대화 요약문:", summary_text.encode("utf-8", "ignore").decode("utf-8"))

        # 대표감정 예측
        X = vectorizer.transform([summary_text])
        predicted_emotion = model.predict(X)[0]

        # 대표감정별 세부감정 예측
        try:
            vec = sub_vectorizer.get(predicted_emotion)
            model_for_emotion = sub_model.get(predicted_emotion)
            if vec is not None and model_for_emotion is not None:
                X_sub = vec.transform([summary_text])
                # 🔽 여기서 확률 기반으로 최고 세부감정 선택
                probs = model_for_emotion.predict_proba(X_sub)[0]
                classes = model_for_emotion.classes_
                predicted_sub = classes[probs.argmax()]
            else:
                predicted_sub = "세부감정 없음"
        except Exception as e:
            print("세부감정 분석 오류:", e)
            predicted_sub = "세부감정 없음"

        # 감정에 맞는 영화 추천
        genre_id = get_genre_by_emotion(predicted_emotion)
        movies = get_movies_by_genre(genre_id)
        movie_titles = [m["title"] for m in movies if isinstance(m, dict)]

        # 추천 영화 목록을 대화 히스토리에 저장
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

# ==========================================================
# ✅ 8. HTML 파일 제공 (프론트엔드 연결)
# ==========================================================
@app.route("/")
def home():
    """index.html 파일을 반환 (웹앱 진입점)"""
    return send_from_directory(BASE_DIR, "index.html")

# ==========================================================
# ✅ 9. DB 연결 및 통계 API
# ==========================================================
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_connection():
    """MySQL 연결 함수"""
    return pymysql.connect(
        host="localhost",
        user="root",
        password=DB_PASSWORD,
        db="moodymovie",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

# 감정별 카운트 통계
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

# 가장 많이 추천된 영화 TOP10
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

# ==========================================================
# ✅ 10. Flask 서버 실행
# ==========================================================
if __name__ == "__main__":
    # 0.0.0.0 → 외부에서도 접근 가능
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

# ==========================================
# ✅ 세부감정 분류 모델 학습 코드 (주석 상세판)
# ==========================================

# ───────────────────────────────────────────
# 📦 1. 필요한 라이브러리 불러오기
# ───────────────────────────────────────────
import pandas as pd  # 데이터 분석을 위한 pandas
from sklearn.ensemble import ExtraTreesClassifier  # 감정 분류용 머신러닝 모델
from sklearn.model_selection import train_test_split  # 데이터 나누기 도구 (지금은 사용 안 함)
from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트 → 숫자 벡터 변환기
import pickle  # 학습된 모델 저장용
import re  # 정규식을 이용한 텍스트 전처리용

# ───────────────────────────────────────────
# 🧹 2. 불용어 리스트 정의
# ───────────────────────────────────────────
# 불용어: 문장에 자주 등장하지만 감정 분석에는 도움이 안 되는 단어들
stopwords = [
    "은","는","이","가","을","를","에","에서","에게","께","한","하다",
    "그리고","그래서","그러나","하지만","또한","때문에","너무","정말",
    "그냥","조금","좀","또","또는","이나","거나","저는","나는","내가",
    "우리","너","니","그","그녀","이것","저것","거","게","걸","것"
]

# ───────────────────────────────────────────
# 🧽 3. 텍스트 전처리 함수 정의
# ───────────────────────────────────────────
def clean_text(text):
    # ① 한글과 공백만 남기고 나머지는 모두 제거
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", str(text))
    # ② 불용어 리스트에 있는 단어들을 모두 제거
    for sw in stopwords:
        text = text.replace(sw, " ")
    # ③ 여러 칸 띄어쓰기 → 한 칸으로, 앞뒤 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text  # 전처리된 문장 반환

# ───────────────────────────────────────────
# 📂 4. 데이터 불러오기
# ───────────────────────────────────────────
# CSV 파일에는 “대화”, “대표감정”, “세부감정” 열이 있다고 가정
df = pd.read_csv(r"C:\ai-project\moody-movie\data\emotion_data.csv", encoding="utf-8-sig")

# “대화” 열에 전처리 적용 (불용어 제거 등)
df["대화"] = df["대화"].apply(clean_text)

# ───────────────────────────────────────────
# 🤖 5. 대표감정별로 세부감정 모델 학습
# ───────────────────────────────────────────
models = {}       # 대표감정별 학습된 모델을 저장할 딕셔너리
vectorizers = {}  # 대표감정별 TF-IDF 변환기를 저장할 딕셔너리

# “대표감정” 열에 있는 모든 감정 종류를 하나씩 반복
for main_emotion in df["대표감정"].unique():
    # 대표감정이 같은 데이터만 따로 모음
    sub_df = df[df["대표감정"] == main_emotion]

    # ⚠ 세부감정이 1개뿐이면 학습할 수 없으므로 건너뜀
    if len(sub_df["세부감정"].unique()) < 2:
        print(f"[{main_emotion}] 세부감정이 1개뿐이라 학습 생략")
        continue
    
    print(f"[{main_emotion}] 세부감정 모델 학습 중...")

    # 입력(문장)과 출력(세부감정) 나누기
    X = sub_df["대화"]
    y = sub_df["세부감정"]

    # ────────────────────────────────
    # 🧠 TF-IDF 벡터화
    # ────────────────────────────────
    # 문자 단위 n-gram(1~3글자 단위)로 분석하여 문장을 숫자로 변환
    vec = TfidfVectorizer(
        analyzer='char',          # 문자 단위 분석 (단어 단위보다 세밀)
        ngram_range=(1,3),        # 1글자~3글자씩 묶어서 특징 추출
        max_features=5000,        # 상위 5000개 특징만 사용
        min_df=2                  # 2번 이상 등장한 특징만 사용
    )
    X_tfidf = vec.fit_transform(X)  # 문장을 TF-IDF 행렬로 변환

    # ────────────────────────────────
    # 🌲 ExtraTreesClassifier 모델 학습
    # ────────────────────────────────
    model_sub = ExtraTreesClassifier(
        n_estimators=1000,        # 트리 개수 (클수록 정확하지만 느림)
        max_depth=None,           # 트리 깊이 제한 없음
        min_samples_split=2,      # 노드를 분할할 최소 샘플 수
        min_samples_leaf=3,       # 리프 노드의 최소 샘플 수
        max_features='log2',      # 각 트리에서 사용할 특징 수 비율
        class_weight='balanced',  # 데이터 불균형 자동 보정
        n_jobs=-1,                # 모든 CPU 코어 사용
        random_state=42           # 결과 재현 가능하게 설정
    )

    # 모델 훈련
    model_sub.fit(X_tfidf, y)

    # 학습 완료 후 모델과 벡터 저장
    models[main_emotion] = model_sub
    vectorizers[main_emotion] = vec
    print(f"[{main_emotion}] 세부감정 모델 학습 완료")

# ───────────────────────────────────────────
# 💾 6. 학습된 모델 저장 (Pickle)
# ───────────────────────────────────────────
# Pickle은 파이썬 객체(모델)를 파일로 저장하고 나중에 불러올 수 있게 해줌
pickle.dump(models, open(r"C:\ai-project\moody-movie\models\sub_models.pkl", "wb"))
pickle.dump(vectorizers, open(r"C:\ai-project\moody-movie\models\sub_vectorizers.pkl", "wb"))

print("세부감정 모델 전부 저장 완료 ✅")

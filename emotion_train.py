# ==========================================================
# 🎭 감정 분석 모델 (대표감정 + 세부감정) 학습 코드
# ==========================================================
# 이 코드는 두 개의 모델을 학습함:
# ① 대표감정 모델 (ex. 행복, 슬픔, 분노 등)
# ② 세부감정 모델 (ex. 화남, 짜증남, 슬픔, 외로움 등)
# 감정사전(emotion_dictionary.csv)을 활용하여 문장 안의 감정 단어를 반영함
# ==========================================================

# ----------------------------------------------------------
# 📦 1. 필요한 라이브러리 불러오기
# ----------------------------------------------------------
import pandas as pd                           # 데이터프레임 형태로 데이터 다루기
from sklearn.model_selection import train_test_split  # 학습용/검증용 데이터 분리
from sklearn.feature_extraction.text import TfidfVectorizer  # 문장을 숫자 벡터로 변환
from sklearn.ensemble import ExtraTreesClassifier     # 트리 기반 분류 모델
from sklearn.metrics import accuracy_score, classification_report  # 정확도/리포트 평가
import pickle                               # 모델 저장용
import re                                   # 텍스트 전처리용 정규식

# ----------------------------------------------------------
# 🧹 2. 불용어 리스트 (감정 분석에 불필요한 단어들)
# ----------------------------------------------------------
stopwords = [
    "은","는","이","가","을","를","에","에서","에게","께","한","하다",
    "그리고","그래서","그러나","하지만","또한","때문에","너무","정말",
    "그냥","조금","좀","또","또는","이나","거나","저는","나는","내가",
    "우리","너","니","그","그녀","이것","저것","거","게","걸","것"
]

# ----------------------------------------------------------
# ✨ 3. 텍스트 전처리 함수 정의
# ----------------------------------------------------------
def clean_text(text):
    # ① 문장에서 한글과 공백을 제외한 모든 문자 제거
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", str(text))
    # ② 불용어 리스트의 단어 제거
    for sw in stopwords:
        text = text.replace(sw, " ")
    # ③ 불필요한 공백 정리 (연속된 공백을 하나로)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------------------------------
# 📂 4. 데이터 불러오기
# ----------------------------------------------------------
# 감정 대화 데이터와 감정사전(단어 기반 감정 매핑)을 불러옴
df = pd.read_csv(r"C:\ai-project\moody-movie\data\emotion_data.csv", encoding="utf-8-sig")
dict_df = pd.read_csv(r"C:\ai-project\moody-movie\data\emotion_dictionary.csv", encoding="utf-8-sig")

# 데이터프레임에 컬럼명이 중복으로 들어간 경우(엑셀 저장 시 자주 발생) 제거
df = df[df["대표감정"] != "대표감정"].reset_index(drop=True)

# 전처리 함수(clean_text) 적용 → 한글만 남기고 불용어 제거
df["대화"] = df["대화"].apply(clean_text)

# ----------------------------------------------------------
# 📘 5. 감정사전(단어-감정 매핑) 딕셔너리화
# ----------------------------------------------------------
emotion_dict = {}
for _, row in dict_df.iterrows():
    emotion = row["감정"]           # 감정 이름 (예: 행복, 슬픔 등)
    word = str(row["단어"]).strip() # 감정 단어 (예: 기쁘다, 외롭다 등)
    if word:
        emotion_dict.setdefault(emotion, []).append(word)

# ----------------------------------------------------------
# 💡 6. 감정사전 단어와 문장 매칭 함수
# ----------------------------------------------------------
def match_emotion(text):
    if pd.isna(text):  # 결측값(빈칸)이면 None 반환
        return None

    text = str(text)
    found = []  # 매칭된 감정 리스트

    # 사전의 각 감정에 대해 단어가 문장에 포함되어 있는지 확인
    for emotion, words in emotion_dict.items():
        for w in words:
            if w in text:
                found.append(emotion)

    # 여러 감정이 매칭될 경우, 가장 많이 등장한 감정을 반환
    if found:
        return max(set(found), key=found.count)
    return None

# 문장별로 감정사전 기반 감정 추출
df["사전감정"] = df["대화"].apply(match_emotion)

# 기존 대표감정 컬럼 유지
df["대표감정"] = df["대표감정"]

# ----------------------------------------------------------
# 🧠 7. 사전감정을 문장에 추가하여 입력 데이터 구성
# ----------------------------------------------------------
# 예: "(참고감정:슬픔) 오늘 너무 힘들었어" 와 같은 식으로 문장 보강
df["입력문장"] = df.apply(
    lambda r: f"(참고감정:{r['사전감정']}) {r['대화']}"
    if pd.notna(r["사전감정"]) else r["대화"],
    axis=1
)

# 모델 학습 시에도 사전감정을 문장 끝에 붙여줌 (문장 내에 반영되도록)
df["입력문장"] = df.apply(
    lambda r: f"{r['대화']} {r['사전감정']}" if pd.notna(r["사전감정"]) else r["대화"],
    axis=1
)

# 입력(문장), 대표감정 라벨, 세부감정 라벨 분리
texts = df["입력문장"]
labels = df["대표감정"]
sub_labels = df["세부감정"]

# ----------------------------------------------------------
# ⚙️ 8. 대표감정 모델 학습
# ----------------------------------------------------------
print("\n대표감정 모델 학습 시작...")

# 학습/테스트 데이터 분리 (stratify는 클래스 비율 유지용)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF 벡터화 (문자 단위로 1~3글자 조합)
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1, 3),
    max_features=8000,  # 단어(문자) 특징 수 제한
    min_df=2            # 최소 2회 이상 등장해야 사용
)

# 벡터화 진행
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ExtraTreesClassifier 모델 생성
et_model = ExtraTreesClassifier(
    n_estimators=674,        # 트리 개수 (클수록 안정적)
    max_depth=None,          # 트리 깊이 제한 없음
    min_samples_split=2,     # 분할 최소 샘플 수
    min_samples_leaf=1,      # 리프 노드 최소 샘플 수
    max_features='log2',     # 사용할 특징 수 기준
    class_weight='balanced', # 감정별 데이터 불균형 보정
    n_jobs=-1,               # 모든 CPU 사용
    random_state=42,         # 재현 가능한 결과
    verbose=1                # 학습 로그 출력
)

# 모델 학습 수행
et_model.fit(X_train_tfidf, y_train)

# 예측 결과 확인
y_pred = et_model.predict(X_test_tfidf)

# 성능 평가 출력
main_acc = accuracy_score(y_test, y_pred)
print("\n[대표감정 모델 결과]")
print(f"Train Accuracy: {accuracy_score(y_train, et_model.predict(X_train_tfidf)):.4f}")
print(f"Test  Accuracy: {main_acc:.4f}")
print(classification_report(y_test, y_pred))
print("=" * 60)

# 모델 및 벡터 저장 (나중에 Flask에서 불러오기 용)
pickle.dump(et_model, open("C:/ai-project/moody-movie/models/emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("C:/ai-project/moody-movie/models/vectorizer.pkl", "wb"))

# ----------------------------------------------------------
# 💫 9. 세부감정 모델 학습
# ----------------------------------------------------------
print("\n세부감정 모델 학습 시작...")

# ─────────────────────────────────────────────
# 🧹 (1) 너무 적게 등장한 세부감정 제거
# ─────────────────────────────────────────────
# 데이터에서 어떤 세부감정은 한두 번만 등장할 수도 있음.
# 이런 감정은 모델이 제대로 학습하지 못하므로 제외함.
sub_label_counts = df["세부감정"].value_counts()   # 세부감정별 등장 횟수 계산
valid_sub_labels = sub_label_counts[sub_label_counts >= 2].index  # 최소 2회 이상 등장한 감정만 남김
df_sub = df[df["세부감정"].isin(valid_sub_labels)]  # 유효한 세부감정만 필터링한 새로운 데이터프레임 생성

# ─────────────────────────────────────────────
# ✂️ (2) 학습용 / 테스트용 데이터 분리
# ─────────────────────────────────────────────
# 전체 데이터를 랜덤하게 나누어 80%는 학습, 20%는 테스트로 사용
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    df_sub["입력문장"],     # 입력 데이터 (문장)
    df_sub["세부감정"],     # 출력 데이터 (세부감정 레이블)
    test_size=0.2,           # 테스트 데이터 비율 = 20%
    random_state=42,         # 무작위 시드 고정 (재현성)
    stratify=df_sub["세부감정"]  # 감정 비율이 유지되도록 분리
)

# ─────────────────────────────────────────────
# 🔠 (3) TF-IDF 벡터 변환
# ─────────────────────────────────────────────
# 이미 대표감정 모델에서 학습한 vectorizer를 그대로 사용함 (fit() ❌)
# → 같은 기준으로 문자 특징을 추출해야 하므로 재사용이 중요함
X_train_sub_tfidf = vectorizer.transform(X_train_sub)
X_test_sub_tfidf = vectorizer.transform(X_test_sub)

# ─────────────────────────────────────────────
# 🌳 (4) 세부감정 분류 모델 생성 (ExtraTrees)
# ─────────────────────────────────────────────
# ExtraTreesClassifier는 랜덤포레스트와 비슷한 앙상블 트리 모델로,
# 텍스트 분류에서도 높은 성능을 냄
sub_et_model = ExtraTreesClassifier(
    n_estimators=674,        # 🔽 트리 개수 500 → 400 (성능 거의 동일, 용량↓)
    max_depth=None,            # 🔼 살짝 여유 (트리 복원력 유지)
    min_samples_split=2,
    min_samples_leaf=2,      # 🔽 리프 최소 샘플 수 완화 (트리 개수는 유지)
    max_features='sqrt',     # 🔽 log2보다 단순하고 메모리 절약
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)


# ─────────────────────────────────────────────
# 🧠 (5) 모델 학습
# ─────────────────────────────────────────────
# TF-IDF로 변환된 데이터를 이용해 모델을 학습시킴
sub_et_model.fit(X_train_sub_tfidf, y_train_sub)

# ─────────────────────────────────────────────
# 🔍 (6) 모델 평가 (정확도 및 리포트)
# ─────────────────────────────────────────────
# 테스트 데이터로 감정을 예측하고, 실제 정답과 비교하여 정확도 계산
sub_pred = sub_et_model.predict(X_test_sub_tfidf)
sub_acc = accuracy_score(y_test_sub, sub_pred)

# 결과 출력
print("\n[세부감정 모델 결과]")
print(f"Test Accuracy: {sub_acc:.4f}")               # 테스트 정확도 출력
print(classification_report(y_test_sub, sub_pred))   # 감정별 정밀도, 재현율, F1-score 출력
print("=" * 60)

# ─────────────────────────────────────────────
# 💾 (7) 모델 및 벡터 저장
# ─────────────────────────────────────────────
# 학습된 세부감정 모델과 벡터 변환기를 파일로 저장
# 이렇게 해야 Flask나 다른 프로그램에서 바로 불러와 예측 가능
pickle.dump(sub_et_model, open("C:/ai-project/moody-movie/models/emotion_sub_model.pkl", "wb"))
pickle.dump(vectorizer, open("C:/ai-project/moody-movie/models/sub_vectorizers.pkl", "wb"))
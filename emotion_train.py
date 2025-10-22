import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ----------------------------------------------------
# 1. 데이터 불러오기 + 감정사전 적용
# ----------------------------------------------------
df = pd.read_csv("data/emotion_data.csv", encoding="utf-8-sig")
dict_df = pd.read_csv("data/emotion_dictionary.csv", encoding="utf-8-sig")

# 감정사전 딕셔너리화
emotion_dict = {}
for _, row in dict_df.iterrows():
    emotion = row["감정"]
    word = str(row["단어"]).strip()
    if word:
        emotion_dict.setdefault(emotion, []).append(word)

# 감정사전 매칭 함수
def match_emotion(text):
    if pd.isna(text):
        return None
    for emotion, words in emotion_dict.items():
        for w in words:
            if w in str(text):
                return emotion
    return None

# 각 문장에 대해 감정사전 기반 감정 찾기
df["사전감정"] = df["대화"].apply(match_emotion)

# 사전감정이 있으면 대표감정을 그걸로 덮어쓰기
df["대표감정"] = df["사전감정"].fillna(df["대표감정"])

# ----------------------------------------------------
# 2. 사전감정을 학습에 포함하기 (핵심)
# ----------------------------------------------------
# 사전감정이 더 중요하므로 문장 앞쪽에 붙임
df["입력문장"] = df.apply(
    lambda r: f"[감정사전:{r['사전감정']}] {r['대화']}"
    if pd.notna(r["사전감정"]) else r["대화"],
    axis=1
)

# 학습용 텍스트와 라벨 준비
texts = df["입력문장"]
labels = df["대표감정"]
sub_labels = df["세부감정"]  # 세부감정도 학습에 사용

# ----------------------------------------------------
# 3. 데이터 분리 및 모델 학습 (대표감정 + 세부감정)
# ----------------------------------------------------

# =========================================
# (1) 대표감정 모델 학습
# =========================================
print("\n대표감정 모델 학습 시작...")

# 대표감정용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF 벡터화 (문장 전체 맥락 반영)
vectorizer = TfidfVectorizer(
    analyzer='char',        # 한글은 글자 단위 분석이 더 안정적
    ngram_range=(1, 3),     # 1~3글자 단위로 패턴 학습
    max_features=8000,
    min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Extra Trees 모델 정의
et_model = ExtraTreesClassifier(
    n_estimators=674,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
et_model.fit(X_train_tfidf, y_train)

# 성능 평가
y_pred = et_model.predict(X_test_tfidf)
main_acc = accuracy_score(y_test, y_pred)
print("\n[대표감정 모델 결과]")
print(f"Train Accuracy: {accuracy_score(y_train, et_model.predict(X_train_tfidf)):.4f}")
print(f"Test  Accuracy: {main_acc:.4f}")
print(classification_report(y_test, y_pred))
print("=" * 60)

# 모델 저장
pickle.dump(et_model, open("models/emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

# =========================================
# (2) 세부감정 모델 학습
# =========================================
print("\n세부감정 모델 학습 시작...")

# 세부감정용 데이터 분리
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    texts, sub_labels, test_size=0.2, random_state=42, stratify=sub_labels
)

# 기존 TF-IDF 벡터 변환기 재사용 (❗fit 금지, transform만 사용)
X_train_sub_tfidf = vectorizer.transform(X_train_sub)
X_test_sub_tfidf = vectorizer.transform(X_test_sub)

# Extra Trees 모델 재사용
sub_et_model = ExtraTreesClassifier(
    n_estimators=674,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
sub_et_model.fit(X_train_sub_tfidf, y_train_sub)

# 세부감정 성능 출력
sub_pred = sub_et_model.predict(X_test_sub_tfidf)
sub_acc = accuracy_score(y_test_sub, sub_pred)
print("\n[세부감정 모델 결과]")
print(f"Test Accuracy: {sub_acc:.4f}")
print(classification_report(y_test_sub, sub_pred))
print("=" * 60)

# 모델 저장 (두 모델 공용 벡터 변환기 포함)
pickle.dump(sub_et_model, open("models/emotion_sub_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/sub_vectorizers.pkl", "wb"))
# ----------------------------------------------------
# ✅ 아래는 실험 기록용 (모델 비교 및 최적화 시도 내역)
# ----------------------------------------------------

# # ----------------------------------------------------
# # 모델1: Naive Bayes (Baseline 모델)
# # 정확도 약 0.8609
# # ----------------------------------------------------
# nb_model = MultinomialNB(alpha=0.5)
# nb_model.fit(X_train_tfidf, y_train)
# nb_pred = nb_model.predict(X_test_tfidf)
# nb_acc = accuracy_score(y_test, nb_pred)

# # ----------------------------------------------------
# # 모델2: Random Forest (Baseline Ensemble)
# # 정확도 약 0.8716
# # ----------------------------------------------------
# rf_model = RandomForestClassifier(
#     n_estimators=800,
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='sqrt',
#     class_weight='balanced',
#     n_jobs=-1,
#     random_state=42,
# )
# rf_model.fit(X_train_tfidf, y_train)
# rf_pred = rf_model.predict(X_test_tfidf)
# rf_acc = accuracy_score(y_test, rf_pred)

# # ----------------------------------------------------
# # Extra Trees (최종 채택 모델)
# # RandomizedSearchCV 결과: 정확도 약 0.8827
# # GridSearchCV 결과: 정확도 약 0.8841
# # ----------------------------------------------------

# # ----------------------------------------------------
# # GridSearchCV (하이퍼파라미터 탐색)
# # ----------------------------------------------------
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     "n_estimators": [200, 500, 800, 1000],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["sqrt", "log2"]
# }
# et = ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
# grid = GridSearchCV(
#     estimator=et,
#     param_grid=param_grid,
#     scoring="accuracy",
#     cv=3,
#     verbose=2,
#     n_jobs=-1
# )
# grid.fit(X_train_tfidf, y_train)
# print("최적 하이퍼파라미터:", grid.best_params_)
# print("최고 평균 정확도:", grid.best_score_)

# # ----------------------------------------------------
# # RandomizedSearchCV (랜덤 하이퍼파라미터 탐색)
# # ----------------------------------------------------
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
# param_dist = {
#     "n_estimators": randint(200, 1200),
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": randint(2, 10),
#     "min_samples_leaf": randint(1, 5),
#     "max_features": ["sqrt", "log2"]
# }
# search = RandomizedSearchCV(
#     estimator=et,
#     param_distributions=param_dist,
#     n_iter=30,
#     scoring="accuracy",
#     cv=3,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )
# search.fit(X_train_tfidf, y_train)
# print("최적 하이퍼파라미터:", search.best_params_)
# print("최고 평균 정확도:", search.best_score_)

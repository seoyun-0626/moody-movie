import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------
# 1. 데이터 불러오기
# ----------------------------------------------------
df = pd.read_csv("emotion_data.csv")  # 새 CSV
texts = df["대화"]       # 기존 '문장' → '대화'
labels = df["대표감정"]   # 감정 컬럼명 통일
sub_labels = df["세부감정"]

# ----------------------------------------------------
# 1. 데이터 불러오기
# ----------------------------------------------------
# ----------------------------------------------------
# 1. 데이터 불러오기 + 감정사전 적용
# ----------------------------------------------------
df = pd.read_csv("emotion_data.csv", encoding="utf-8-sig")

# 감정사전 로드 및 보정
dict_df = pd.read_csv("emotion_dictionary.csv", encoding="utf-8-sig")

emotion_dict = {}
for _, row in dict_df.iterrows():
    emotion = row["감정"]
    word = str(row["단어"]).strip()
    if word:
        emotion_dict.setdefault(emotion, []).append(word)

def match_emotion(text):
    if pd.isna(text):
        return None
    for emotion, words in emotion_dict.items():
        for w in words:
            if w in str(text):
                return emotion
    return None

df["사전감정"] = df["대화"].apply(match_emotion)
df["대표감정"] = df["사전감정"].fillna(df["대표감정"])
print("감정사전 적용 완료 ✅")

texts = df["대화"]
labels = df["대표감정"]
# ----------------------------------------------------
# 2. 데이터 분리 (감정 균형 유지)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ----------------------------------------------------
# 3. TF-IDF 벡터화 (문장 전체 맥락 반영)
# ----------------------------------------------------
vectorizer = TfidfVectorizer(
    analyzer='char',        # 한글은 글자 단위 분석이 더 안정적
    ngram_range=(1, 3),     # 1~3글자 단위로 패턴 학습
    max_features=8000,      # 문장 길이 고려해 확장
    min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# # ----------------------------------------------------
# # 4. 모델1: Naive Bayes     --->>Naive Bayes : 0.8609
# # ----------------------------------------------------
# nb_model = MultinomialNB(alpha=0.5)
# nb_model.fit(X_train_tfidf, y_train)
# nb_pred = nb_model.predict(X_test_tfidf)
# nb_acc = accuracy_score(y_test, nb_pred)

# # ----------------------------------------------------
# # 5. 모델2: Random Forest    ---->Random Forest : 0.8716
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

# ----------------------------------------------------
# 6. 모델3: Extra Trees
# ----------------------------------------------------
et_model = ExtraTreesClassifier(
    n_estimators=674,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1  # 진행 상황 로그 출력
)

et_model.fit(X_train_tfidf, y_train)

# 훈련 데이터 예측
et_train_pred = et_model.predict(X_train_tfidf)
et_train_acc = accuracy_score(y_train, et_train_pred)

# 테스트 데이터 예측
et_test_pred = et_model.predict(X_test_tfidf)
et_test_acc = accuracy_score(y_test, et_test_pred)

# 출력
print("Extra Trees")
print(f"Train Accuracy: {et_train_acc:.4f}")
print(f"Test Accuracy : {et_test_acc:.4f}")

import pickle
pickle.dump(et_model, open("emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# ----------------------------------------------------
# 7. 결과 출력
# ----------------------------------------------------
print("정확도 비교 (대화형 감정 분석)")
# print(f"Naive Bayes 정확도: {nb_acc:.4f}")
# print(f"Random Forest 정확도: {rf_acc:.4f}")
print(f"Extra Trees 정확도: {et_test_acc:.4f}")

# print("\n[Naive Bayes 분류 보고서]")
# print(classification_report(y_test, nb_pred))
# print("\n[Random Forest 분류 보고서]")
# print(classification_report(y_test, rf_pred))
# print("\n[Extra Trees 분류 보고서]")
print(classification_report(y_test, et_test_pred))

print("="*60)
# print(f"Naive Bayes : {nb_acc:.4f}")
# print(f"Random Forest : {rf_acc:.4f}")
print(f"Extra Trees : {et_test_acc:.4f}")
print("="*60)


#===============================================
# 세부감정 학습시키기
#===============================================
# print("\n세부감정 모델 학습 시작...")

# # 세부감정용 데이터 분리 (대표감정과 구조 동일)
# X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
#     texts, sub_labels, test_size=0.2, random_state=42, stratify=sub_labels
# )

# # TF-IDF 변환 재사용
# X_train_sub_tfidf = vectorizer.fit_transform(X_train_sub)
# X_test_sub_tfidf = vectorizer.transform(X_test_sub)

# # 모델 학습 (Extra Trees 그대로 사용 가능)
# sub_et_model = ExtraTreesClassifier(
#     n_estimators=800,
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='sqrt',
#     class_weight='balanced',
#     n_jobs=-1,
#     random_state=42,
#     verbose=1
# )
# sub_et_model.fit(X_train_sub_tfidf, y_train_sub)

# # 성능 출력
# sub_pred = sub_et_model.predict(X_test_sub_tfidf)
# sub_acc = accuracy_score(y_test_sub, sub_pred)

# print("\n[세부감정 모델 결과]")
# print(f"Test Accuracy: {sub_acc:.4f}")
# print(classification_report(y_test_sub, sub_pred))
# print("=" * 60)

#================================================
#그리드서치 최적 파라미터 찾기 
#================================================

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import ExtraTreesClassifier

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
#     cv=3,            # 3겹 교차검증
#     verbose=2,       # 진행 상황 표시
#     n_jobs=-1
# )

# grid.fit(X_train_tfidf, y_train)

# print("최적 하이퍼파라미터:", grid.best_params_)
# print("최고 평균 정확도:", grid.best_score_)

# #============================================
# # 랜덤 그리드서치 
# #=========================================

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

# param_dist = {
#     "n_estimators": randint(200, 1200),
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": randint(2, 10),
#     "min_samples_leaf": randint(1, 5),
#     "max_features": ["sqrt", "log2"]
# }

# et = ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
# search = RandomizedSearchCV(
#     estimator=et,
#     param_distributions=param_dist,
#     n_iter=30,         # 30회 조합만 랜덤으로 시도
#     scoring="accuracy",
#     cv=3,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )

# search.fit(X_train_tfidf, y_train)

# print("최적 하이퍼파라미터:", search.best_params_)
# print("최고 평균 정확도:", search.best_score_)

'''
GridSearchCV 결과
최적 하이퍼파라미터:
{'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
최고 평균 정확도: 0.8841


RandomizedSearchCV 결과

최적 하이퍼파라미터:
{'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 674}
최고 평균 정확도: 0.8827
'''
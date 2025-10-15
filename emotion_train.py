import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from konlpy.tag import Okt

# 튜닝 전 

# 감정종류 -  행복 슬픔 분노 설렘 피곤


# 0. 데이터 불러오기
df = pd.read_csv("emotion_data.csv")
texts = df["문장"]
labels = df["감정"]

# 1. 한글 전처리 
okt = Okt()

def tokenize(text):
    return okt.morphs(text)


# 2. 데이터 분리 (감정이 고르게 섞이게 stratify 옵션 추가)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. TF-IDF 벡터화
vectorizer = TfidfVectorizer(    
    analyzer='char',        # 글자 단위로 자르기
    ngram_range=(2, 3),     # 2글자~3글자 단위로
    max_features=5000,      # 상위 5000개 단어만 사용
    min_df=2)               # 2개 이상 문서에서 등장한 단어만 사용

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. 모델1: 나이브베이즈
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)

# 5. 모델2: 랜덤포레스트
rf_model = RandomForestClassifier(n_estimators=200, 
                                  random_state=42,
                                  )
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_pred)

# 6. 모델3: 엑스트라트리
et_model = ExtraTreesClassifier(n_estimators=300,
                                random_state=42,
                                )
et_model.fit(X_train_tfidf, y_train)
et_pred = et_model.predict(X_test_tfidf)
et_acc = accuracy_score(y_test, et_pred)

# 7. 결과 비교 출력
print("---------정확도 비교 튜닝전------------")
print(f"Naive Bayes 정확도: {nb_acc:.4f}")
print(f"Random Forest 정확도: {rf_acc:.4f}")
print(f"Extra Trees 정확도: {et_acc:.4f}")

print("\nNaive Bayes 분류 보고서:")
print(classification_report(y_test, nb_pred))
print("\nRandom Forest 분류 보고서:")
print(classification_report(y_test, rf_pred))
print("\nExtra Trees 분류 보고서:")
print(classification_report(y_test, et_pred))

# --------------------------
#  모델 학습 후 점수 비교
# --------------------------

# 1. Naive Bayes
nb_train_acc = nb_model.score(X_train_tfidf, y_train)
nb_test_acc = nb_model.score(X_test_tfidf, y_test)
print(f"[Naive Bayes] Train: {nb_train_acc:.4f}, Test: {nb_test_acc:.4f}")

# 2. Random Forest
rf_train_acc = rf_model.score(X_train_tfidf, y_train)
rf_test_acc = rf_model.score(X_test_tfidf, y_test)
print(f"[Random Forest] Train: {rf_train_acc:.4f}, Test: {rf_test_acc:.4f}")

# 3. Extra Trees
et_train_acc = et_model.score(X_train_tfidf, y_train)
et_test_acc = et_model.score(X_test_tfidf, y_test)
print(f"[Extra Trees] Train: {et_train_acc:.4f}, Test: {et_test_acc:.4f}")



#------------------------------------------------------------------------
print('--------------튜닝후------------------')
# 튜닝 후 


# 감정종류 - 슬픔 불안, 걱정, 스트레스, 피로, 분노, 외로움, 결핍, 행복, 설렘, 심심, 탐구, 호기심

# 1. 데이터 불러오기
df = pd.read_csv("emotion_data.csv")
texts = df["문장"]
labels = df["감정"]

# 2. 데이터 분리 (감정이 고르게 섞이게 stratify 옵션 추가)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. TF-IDF 벡터화 (튜닝 추가)
vectorizer = TfidfVectorizer(    
    analyzer='char',        # 글자 단위로 자르기
    ngram_range=(1, 2),     # 2글자~3글자 단위로
    max_features=5000,     # 단어 수 확장
    min_df=2               # 최소 2회 이상 등장한 단어만 사용
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. 모델1: 나이브베이즈
nb_model = MultinomialNB(alpha=0.5)  # alpha 값 조정 (기본 1 → 0.5)
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)

# 5. 모델2: 랜덤포레스트 (튜닝 버전)
rf_model = RandomForestClassifier(
    n_estimators=800,          # 트리 개수 증가 (200 → 800)
    max_depth=None,            # 깊이 제한 없음
    min_samples_split=2,       # 분할 최소 샘플 클수록 과적합↓ (2~10)
    min_samples_leaf=1,        # 잎 최소 샘플 (1~5)
    max_features='sqrt',       # 특성 선택 방식  /  다양성↑, 과적합↓
    class_weight='balanced',   # 감정 데이터가 불균형할 때 자동으로 가중치 조정해줌.
    n_jobs=-1,                 # CPU 코어 병렬 처리 개수. -1 > 모든코어 전부사용
    random_state=42,
    verbose=1
)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_pred)

# 6. 모델3: 엑스트라트리 (튜닝 버전)
et_model = ExtraTreesClassifier(
    n_estimators=1000,         # 트리 개수 크게 증가 (300 → 1000)
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
et_model.fit(X_train_tfidf, y_train)
et_pred = et_model.predict(X_test_tfidf)
et_acc = accuracy_score(y_test, et_pred)

# 7. 결과 비교 출력
print("정확도 비교")
print(f"Naive Bayes 정확도: {nb_acc:.4f}")
print(f"Random Forest 정확도: {rf_acc:.4f}")
print(f"Extra Trees 정확도: {et_acc:.4f}")

print("\nNaive Bayes 분류 보고서:")
print(classification_report(y_test, nb_pred))
print("\nRandom Forest 분류 보고서:")
print(classification_report(y_test, rf_pred))
print("\nExtra Trees 분류 보고서:")
print(classification_report(y_test, et_pred))

# --------------------------
#  모델 학습 후 점수 비교
# --------------------------

# 1. Naive Bayes
nb_train_acc = nb_model.score(X_train_tfidf, y_train)
nb_test_acc = nb_model.score(X_test_tfidf, y_test)
print(f"[Naive Bayes] Train: {nb_train_acc:.4f}, Test: {nb_test_acc:.4f}")

# 2. Random Forest
rf_train_acc = rf_model.score(X_train_tfidf, y_train)
rf_test_acc = rf_model.score(X_test_tfidf, y_test)
print(f"[Random Forest] Train: {rf_train_acc:.4f}, Test: {rf_test_acc:.4f}")

# 3. Extra Trees
et_train_acc = et_model.score(X_train_tfidf, y_train)
et_test_acc = et_model.score(X_test_tfidf, y_test)
print(f"[Extra Trees] Train: {et_train_acc:.4f}, Test: {et_test_acc:.4f}")


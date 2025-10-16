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

# ----------------------------------------------------
# 4. 모델1: Naive Bayes
# ----------------------------------------------------
nb_model = MultinomialNB(alpha=0.5)
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)

# ----------------------------------------------------
# 5. 모델2: Random Forest
# ----------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_pred)

# ----------------------------------------------------
# 6. 모델3: Extra Trees
# ----------------------------------------------------
et_model = ExtraTreesClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)
et_model.fit(X_train_tfidf, y_train)
et_pred = et_model.predict(X_test_tfidf)
et_acc = accuracy_score(y_test, et_pred)

# ----------------------------------------------------
# 7. 결과 출력
# ----------------------------------------------------
print("정확도 비교 (대화형 감정 분석)")
print(f"Naive Bayes 정확도: {nb_acc:.4f}")
print(f"Random Forest 정확도: {rf_acc:.4f}")
print(f"Extra Trees 정확도: {et_acc:.4f}")

print("\n[Naive Bayes 분류 보고서]")
print(classification_report(y_test, nb_pred))
print("\n[Random Forest 분류 보고서]")
print(classification_report(y_test, rf_pred))
print("\n[Extra Trees 분류 보고서]")
print(classification_report(y_test, et_pred))

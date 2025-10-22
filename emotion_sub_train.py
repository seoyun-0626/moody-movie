import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1. 데이터 불러오기
df = pd.read_csv("data/emotion_data.csv", encoding="utf-8-sig")

# 2. 대표감정별 세부감정 모델 학습
models = {}
vectorizers = {}

for main_emotion in df["대표감정"].unique():
    sub_df = df[df["대표감정"] == main_emotion]
    if len(sub_df["세부감정"].unique()) < 2:
        print(f"[{main_emotion}] 세부감정이 1개뿐이라 학습 생략")
        continue
    
    print(f"[{main_emotion}] 세부감정 모델 학습 중...")

    X = sub_df["대화"]
    y = sub_df["세부감정"]

    vec = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=5000, min_df=2)
    X_tfidf = vec.fit_transform(X)

    model_sub = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model_sub.fit(X_tfidf, y)

    models[main_emotion] = model_sub
    vectorizers[main_emotion] = vec
    print(f"[{main_emotion}] 세부감정 모델 학습 완료")

# 3. 모델 저장
pickle.dump(models, open("sub_models.pkl", "wb"))
pickle.dump(vectorizers, open("sub_vectorizers.pkl", "wb"))
print("세부감정 모델 전부 저장 완료 ")

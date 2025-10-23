import random
from datetime import datetime, timedelta

# 🎭 대표 감정 → 세부 감정 리스트
emotion_map = {
    "슬픔": ["무기력","자책","상실","그리움","체념","외로움","눈물","불면","후회","공허","좌절","고독","자기혐오","무의미감","소외"],
    "불안": ["걱정","초조","공포","예민","의심","긴장","대비","불면","불확실","망설임","불편","불신","혼란","경계","불균형"],
    "분노": ["화남","짜증","분개","억울","폭발","답답","불만","혐오","분노억제","불공정","반항","참을수없음","복수심","질투","분노폭발"],
    "외로움": ["결핍","고독","그리움","외로움","단절","소외감","공허감","혼자","무의미감","허전함","그립다","그림자같은기분","연락없음","그리움의밤","그림자같은날"],
    "행복": ["충만","성취","설렘","여유","사랑","유쾌","감동","회상","안정","감사","따뜻함","희열","기대","안도","자부심"],
    "심심": ["지루","권태","무료","무의미","무기력","무관심","기대","방황","멍함","싫증","공허","따분함","집중력저하","권태감","반복피로"],
    "탐구": ["호기심","궁금","관찰","분석","도전","몰입","학습","탐색","발견","집중","통찰","실험","관찰력","창의","탐험"]
}

# 🎞 감정 → 장르 코드 매핑
emotion_genre_map = {
    "슬픔": "35",  # 코미디
    "불안": "16,10402",  # 애니/음악
    "분노": "80,27,53,9648",  # 범죄/공포/스릴러/미스터리
    "외로움": "10749,10751,18",  # 로맨스/가족/드라마
    "행복": "28,12",  # 액션/모험
    "심심": "14,878",  # 판타지/SF
    "탐구": "99,36"  # 다큐/역사
}

# 🎬 TMDB에서 실제 검색 가능한 영화 리스트 (한국어 + 영어 병기)
movies = [
    "소울 (Soul)",
    "기생충 (Parasite)",
    "인사이드 아웃 2 (Inside Out 2)",
    "플로리다 프로젝트 (The Florida Project)",
    "오펜하이머 (Oppenheimer)",
    "조커 (Joker)",
    "인터스텔라 (Interstellar)",
    "타이타닉 (Titanic)",
    "노트북 (The Notebook)",
    "서울의 봄 (Seoul’s Spring)",
    "라푼젤 (Tangled)",
    "극한직업 (Extreme Job)",
    "가디언즈 오브 갤럭시 1 (Guardians of the Galaxy)",
    "업 (Up)",
    "그랜드 부다페스트 호텔 (The Grand Budapest Hotel)",
    "토이 스토리 4 (Toy Story 4)",
    "엘리멘탈 (Elemental)",
    "하울의 움직이는 성 (Howl’s Moving Castle)",
    "위대한 쇼맨 (The Greatest Showman)",
    "겟아웃 (Get Out)"
]

# ⚙️ 현실적인 감정별 분포

values = []

# 🧩 더미데이터 생성
for emotion, count in distribution.items():
    for _ in range(count):
        sub = random.choice(emotion_map[emotion])
        genre = emotion_genre_map[emotion]
        movie = random.choice(movies)
        days_ago = random.randint(0, 6)
        created_at = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        values.append(f"('{emotion}', '{sub}', '{genre}', '{movie}', '{created_at}')")

# 📝 SQL 파일 생성
with open("dummy_movies_emotions.sql", "w", encoding="utf-8") as f:
    f.write("INSERT INTO movies_emotions (rep_emotion, sub_emotion, genre, movie, created_at) VALUES\n")
    f.write(",\n".join(values))
    f.write(";")

print("생성 완료!")

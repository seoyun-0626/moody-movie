import requests

API_KEY = "b207c82c618f22642461894bce46a0c4"
BASE_URL = "https://api.themoviedb.org/3"

def get_movies_by_genre(genre_id, language="ko-KR"):
    """TMDB에서 장르별 영화 5개 불러오기"""
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "with_genres": genre_id,
        "language": language,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 500
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        return f"오류 발생: {res.status_code}"

    data = res.json()
    results = data.get("results", [])[:5]
    if not results:
        return "추천할 영화가 없습니다."

    movies = []
    for m in results:
        title = m.get("title", "제목 없음")
        rating = m.get("vote_average", "N/A")
        overview = m.get("overview", "줄거리 없음")
        movies.append(f"{title} ({rating})\n{overview}\n")

    return "\n".join(movies)



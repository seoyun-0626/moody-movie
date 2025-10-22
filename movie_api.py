# 영화추천 api

import requests
import random

API_KEY = "b207c82c618f22642461894bce46a0c4"
BASE_URL = "https://api.themoviedb.org/3"

def get_movies_by_genre(genre_id, language="ko-KR"):
    """TMDB에서 장르별 영화 200개 중 랜덤 5개 추천"""
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "with_genres": genre_id,
        "language": language,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 500,
        "page": 1
    }

    results = []
    

# TMDB는 한 페이지에 최대 20개씩만 반환하므로, 10페이지 × 20개 = 200개
    for page in range(1, 11):
        params["page"] = page
        res = requests.get(url, params=params)

        if res.status_code != 200:
            print(f"⚠️ TMDB API 오류 (page {page}): {res.status_code}")
            continue

        data = res.json()
        results.extend(data.get("results", []))

    if not results:
        return "추천할 영화가 없습니다."

    # ✅ 상위 200개 중 랜덤 5개 선택
    selected = random.sample(results, min(5, len(results)))

    movies = []
    for m in selected:
        title = m.get("title", "제목 없음")
        rating = m.get("vote_average", "N/A")
        overview = m.get("overview", "줄거리 없음")
        poster = m.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster}" if poster else None

        movies.append({
            "title": title,
            "rating": rating,
            "overview": overview,
            "poster": poster_url
        })

    return movies

if __name__ == "__main__":
    # 예시: 드라마(genre_id=18)
    movies = get_movies_by_genre(18)
    for m in movies:
        print(f"🎬 {m['title']} ({m['rating']})")
        print(m['overview'])
        print(m['poster'])
        print("-" * 50)


def get_movie_rating(title, language="ko-KR"):
    """TMDB에서 영화 제목으로 평점 검색"""
    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "language": language,
        "query": title
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        print(f"⚠️ TMDB 검색 오류: {res.status_code}")
        return None

    data = res.json()
    results = data.get("results", [])
    if not results:
        return None

    movie = results[0]  # 첫 번째 검색 결과 사용
    return {
        "title": movie.get("title", "제목 없음"),
        "rating": movie.get("vote_average", "N/A"),
        "overview": movie.get("overview", "줄거리 없음"),
        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
    }


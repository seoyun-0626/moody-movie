#MYSQL 연동

# pip install mysql-connector-python
# pip show mysql-connector-python

import mysql.connector
import datetime

# 1. MySQL 연결 설정
conn = mysql.connector.connect(
    host="localhost",       # MySQL 서버 주소
    user="root",            # 사용자명
    password="1234",      # 비밀번호
    database="moodymovie"      # 사용할 데이터베이스
)

cursor = conn.cursor()

# 2. 예시 데이터 (txt에서 읽어온 데이터라고 가정)
data_list = [
    {"rep_emotion": "분노", "sub_emotion": "답답", "genre": "드라마", "movie": "영화A"}
]

# 3. INSERT 쿼리
sql = """
INSERT INTO movies_emotions (rep_emotion, sub_emotion, genre, movie)
VALUES (%s, %s, %s, %s)
"""

# 4. 데이터 삽입
for data in data_list:
    values = (data["rep_emotion"], data["sub_emotion"], data["genre"], data["movie"])
    cursor.execute(sql, values)

# 5. 커밋하여 반영
conn.commit()

print(f"✅ {cursor.rowcount}개 데이터 INSERT 완료")

# 6. 종료
cursor.close()
conn.close()

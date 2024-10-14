# 베이스 이미지: Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
RUN apt-get update && apt-get install -y python3-venv && apt-get clean

RUN python3 -m venv venv

COPY requirements.txt .
RUN . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# 프로젝트 파일 복사
COPY . .

EXPOSE 8000

# FastAPI 서버 실행
CMD ["/bin/bash", "-c", ". venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000"]
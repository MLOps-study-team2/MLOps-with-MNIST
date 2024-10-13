# python base image
FROM python:3.11-slim

# 어떤 디렉토리에 어플리케이션을 복사해올 것인지 명시
WORKDIR /MLOPS-WITH-MNIST

# 프로젝트 file들 복사
COPY requirements.txt .
# 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt
# source code copy
COPY . .
# 환경변수 설정
ENV PYTHONPATH=/MLOPS-WITH-MNIST
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.12.4-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
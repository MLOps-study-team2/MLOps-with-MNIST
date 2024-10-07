import requests

# 이미지 파일 경로
image_file = 'image/sample_image.png'

# 서버 엔드포인트
url = "http://localhost:8000/predict"

# 이미지 파일을 multipart/form-data로 전송
with open(image_file, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# 서버에서 받은 예측 결과 출력
print(response.json())

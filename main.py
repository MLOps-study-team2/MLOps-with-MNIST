# main.py
from fastapi import FastAPI
from routers import classifier
from utils.utils import load_model
import contextlib

def create_app():
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # 애플리케이션 시작 시 실행될 코드
        print("Loading model...")
        app.state.model = load_model() # 모델을 로드하여 애플리케이션 상태에 저장
        print("Model loaded.")
        yield
        # 애플리케이션 종료 시 실행될 코드
        print("Shutting down...")
    
    app = FastAPI(lifespan=lifespan) # Lifespan 이벤트 핸들러를 사용하여 앱 생성
    app.include_router(classifier.router) # 라우터를 애플리케이션에 포함시킴
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # uvicorn을 사용하여 애플리케이션 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import uvicorn
from fastapi import FastAPI
from .api.routes import router as api_router

app = FastAPI(title="Neural Miner API")

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)

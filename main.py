import sys
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_module.app.api.v1.inference import router as v1_router
from inference_module.app.api.v2.inference import router as v2_router
from inference_module.app.api.v3.inference import router as v3_router
from inference_module.app.api.v4.inference import router as v4_router
from llm_module.app.routers.predict_router import router as llm_predict_router
from llm_module.app.routers.nutrition_router import router as llm_nutrition_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic if needed
    yield
    # Shutdown logic if needed

app = FastAPI(lifespan=lifespan)

# Inference Routers
app.include_router(v1_router, prefix="/api/inference/v1", tags=["YOLO 단독"])
app.include_router(v2_router, prefix="/api/inference/v2", tags=["EfficientNet 단독"])
app.include_router(v3_router, prefix="/api/inference/v3", tags=["Detector+YOLO"])
app.include_router(v4_router, prefix="/api/inference/v4", tags=["Detector+EfficientNet"])

# LLM Routers
app.include_router(llm_predict_router, tags=["LLM Predict"])
app.include_router(llm_nutrition_router, tags=["LLM Nutrition"])

@app.get("/")
def health_check():
    return {"message": "Integrated AI Module Server Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

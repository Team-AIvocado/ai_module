from fastapi import APIRouter
from pydantic import BaseModel
from llm_module.app.services.nutrition_llm import generate_nutrition

router = APIRouter()


class FoodName(BaseModel):
    food_name: str


@router.post("/nutrition")
def nutrition_api(data: FoodName):
    result = generate_nutrition(data.food_name)
    return result  # DB저장시 이 dict 그대로 사용

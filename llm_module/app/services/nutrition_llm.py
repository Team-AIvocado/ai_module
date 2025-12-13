from ibm_watsonx_ai.foundation_models import ModelInference
from llm_module.app.config import settings
import base64
import json
import re


llm = None

def get_nutrition_model():
    global llm
    if llm is None:
        llm = ModelInference(
            model_id=settings.NUTRITION_MODEL_ID,
            credentials=settings.watson_credentials,
            project_id=settings.WATSON_PROJECT_ID,
            params={"max_new_tokens": 300, "temperature": 0.2},
        )
    return llm


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {text}")
    return json.loads(match.group(0))


def generate_nutrition(food_name: str) -> dict:
    prompt = f"""
You are a nutrition analysis model.

Analyze the food '{food_name}' and return the result in VALID JSON ONLY.

RULES:
- 음식명(food)은 반드시 한국어(Korean)로 작성할 것.
- 숫자 데이터(calories, carbs_g, protein_g, fat_g, sodium_mg)는 정수로만 작성할 것.
- 설명, 해설, 부가 텍스트, markdown, 코드블록 금지.
- JSON 외의 어떤 글자도 출력하지 말 것.

Respond in exactly this JSON format:

{{
  "foodname": "{food_name}",
  "nutritions": {{
      "calories": <number>,
      "carbs_g": <number>,
      "protein_g": <number>,
      "fat_g": <number>,
      "sugar_g": <number>,
      "fiber_g": <number>,
      "sodium_mg": <number>,
      "cholesterol_mg": <number>,
      "saturated_fat_g": <number>,
      "micronutrients": {{
          "vitamin_c_mg": <number>,
          "calcium_mg": <number>,
          "caffeine_mg": <number>,
          ...
      }}
  }}
}}
"""

    resp = get_nutrition_model().generate(prompt=prompt)
    text = resp["results"][0]["generated_text"]

    return extract_json(text)

from ibm_watsonx_ai.foundation_models import ModelInference
from llm_module.app.config import settings
import base64
import requests


# Initialize vision_model as None globally so it can be lazily initialized
vision_model = None

def get_vision_model():
    global vision_model
    if vision_model is None:
        vision_model = ModelInference(
            model_id=settings.VISION_MODEL_ID,
            credentials=settings.watson_credentials,
            project_id=settings.WATSON_PROJECT_ID,
            params={"max_new_tokens": 20, "temperature": 0.1},
        )
    return vision_model


def mock_food_classifier(
    image_bytes: bytes | None = None,
    image_base64: str | None = None,
    image_url: str | None = None,
) -> str:

    # 1) URL → 다운로드 → bytes 변환
    if image_url is not None:
        resp = requests.get(image_url)
        resp.raise_for_status()
        image_bytes = resp.content

    # 2) bytes → base64 변환
    if image_bytes is not None:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded}"

    # 3) 이미 base64로 들어온 경우
    elif image_base64 is not None:
        data_url = f"data:image/jpeg;base64,{image_base64}"

    else:
        raise ValueError("image_bytes, image_base64, image_url 중 하나는 필요합니다.")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "이 이미지를 보고 음식 이름을 한국어로 한 단어만 반환하세요. "
                        "설명, 부연 문장, 기타 문구는 절대 포함하지 말고 "
                        "딱 음식 이름만 말하세요. "
                        "만약 음식이 아니라면 '음식아님'이라고 반환하세요."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    response = get_vision_model().chat(messages=messages)
    # print(response)
    food_name = response["choices"][0]["message"]["content"].strip()
    food_name = food_name.split()[0]

    return food_name

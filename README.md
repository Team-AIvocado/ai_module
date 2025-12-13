# AI Module for Caloreat

## 엔드포인트

- prefix
  - /api/v1 : YOLO 단독
  - /api/v2 : EfficientNet 단독
  - /api/v3 : Pipeline (YOLO)
  - /api/v4 : Pipeline (EfficientNet)
- /analyze : 이미지 (파일) 분석
- /analyze-url : 이미지(링크) 분석

- /nutrition : 음식 -> 영양소 정보

## 실행방법

### env 파일 생성

- ./llm_module/.env.example을 참고 해서 .env 생성 (디스코드 중요자료에 키 있음)

```bash
uv run main.py
```

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
import ibm_watsonx_ai.href_definitions

from pathlib import Path

# Robustly find .env.dev relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env.dev or .env
load_dotenv(ENV_PATH)
if not os.getenv("WATSON_API_KEY"):
    load_dotenv(BASE_DIR / ".env")


class Settings:
    WATSON_API_KEY: str = os.getenv("WATSON_API_KEY")
    WATSON_PROJECT_ID: str = os.getenv("WATSON_PROJECT_ID")
    WATSON_URL: str = os.getenv("WATSON_URL")

    # Model IDs
    VISION_MODEL_ID: str = "meta-llama/llama-3-2-11b-vision-instruct"
    NUTRITION_MODEL_ID: str = "mistralai/mistral-medium-2505"

    @property
    def watson_credentials(self) -> Credentials:
        if not all([self.WATSON_API_KEY, self.WATSON_URL]):
            raise ValueError(
                "Missing WatsonX credentials (WATSON_API_KEY, WATSON_URL) in environment variables."
            )
        return Credentials(
            api_key=self.WATSON_API_KEY,
            url=self.WATSON_URL,
            platform_url="https://api.dataplatform.cloud.ibm.com",
        )


settings = Settings()

# Monkey patch removed. Using Sidecar Proxy for HTTPS.
# See docker-compose.yml or infra configuration.

# PATCH: Add the proxy URL to the list of production URLs to force IAM Prod Authentication.
# Without this, the SDK defaults to 'https://iam.test.cloud.ibm.com' for unknown URLs.
if (
    settings.WATSON_URL
    and settings.WATSON_URL not in ibm_watsonx_ai.href_definitions.PROD_SVT_URL
):
    ibm_watsonx_ai.href_definitions.PROD_SVT_URL.append(settings.WATSON_URL)

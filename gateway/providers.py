import os
import time
from google import genai
from google.genai.errors import ServerError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _generate_with_retry(model: str, prompt: str):
    return _client.models.generate_content(model=model, contents=prompt)

def call_gemini(model: str, prompt: str) -> dict:
    start = time.time()
    response = _generate_with_retry(model, prompt)
    latency_ms = round((time.time() - start) * 1000, 2)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    return {
        "text": response.text,
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
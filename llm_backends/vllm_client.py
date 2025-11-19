import requests


def vllm_chat(
    base_url: str,
    model: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 8000,
    temperature: float = 0.1,
    timeout: int = 1800,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise ValueError(f"Unexpected vLLM response structure: {data}") from e

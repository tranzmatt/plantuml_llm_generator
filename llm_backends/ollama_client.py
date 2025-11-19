import requests


def ollama_chat(
    ollama_url: str,
    model: str,
    system_msg: str,
    user_msg: str,
    num_ctx: int = 200000,
    timeout: int = 1800,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "options": {"num_ctx": num_ctx},
        "stream": False,
    }
    resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})
    content = msg.get("content", "")
    if not content:
        raise ValueError(f"Empty content from Ollama response: {data}")
    return content

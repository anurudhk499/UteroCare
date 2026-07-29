import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"

def ask(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "error" in data:
        raise Exception(data["error"])

    return data["response"]
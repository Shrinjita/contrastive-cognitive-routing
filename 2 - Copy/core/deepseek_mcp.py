import os
import requests
import json

class DeepSeekMCP:
    def __init__(self):
        self.url = "https://api.deepseek.com/chat/completions"
        self.key = os.environ["DEEPSEEK_API_KEY"]

    def complete(self, system, user):
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False
        }
        r = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            },
            data=json.dumps(payload),
            timeout=120
        )
        return r.json()["choices"][0]["message"]["content"]

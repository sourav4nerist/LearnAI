import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gpt-oss",
        "prompt": "Write an assay about Python",
        "stream": "False",
    },
)
print(response.json())

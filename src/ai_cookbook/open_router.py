import requests
import json
import os

from dotenv import load_dotenv

load_dotenv()

# Sends a request for a model response for the given chat conversation. Supports both streaming and non-streaming modes.
url = "https://openrouter.ai/api/v1/chat/completions"

# Creates a streaming or non-streaming response using the OpenAI Responses API format.
# url = "https://openrouter.ai/api/v1/responses"

# Creates a message using the Anthropic Messages API format. Supports text, images, PDFs, tools, and extended thinking.
# url = "https://openrouter.ai/api/v1/messages"

api_key = os.environ.get("OPENROUTER_API_KEY")

auth_header = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
# First API call with reasoning
response = requests.post(
    url=url,
    headers=auth_header,
    data=json.dumps(
        {
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": "How many r's are in the word 'strawberry'?",
                }
            ],
            "reasoning": {"enabled": True},
        }
    ),
)

# Extract the assistant message with reasoning_details
response = response.json()
response = response["choices"][0]["message"]

# Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": response.get("content"),
        "reasoning_details": response.get("reasoning_details"),  # Pass back unmodified
    },
    {"role": "user", "content": "Are you sure? Think carefully."},
]

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
    url=url,
    headers=auth_header,
    data=json.dumps(
        {
            "model": "openrouter/free",
            "messages": messages,  # Includes preserved reasoning_details
            "reasoning": {"enabled": True},
        }
    ),
)

response2 = response2.json()
response2 = response2["choices"][0]["message"]

print(response2)

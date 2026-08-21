import os

from openai import OpenAI
from pydantic import BaseModel

api_key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


# Step 1: Define the response model with Pydantic model
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# Step 2: call the model

response = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Ram and Rahim are going to a standup comedy show on 10th August 2026.",
        },
    ],
    response_format=CalendarEvent,
)

# Parse the response
event = response.choices[0].message.parsed
print(event.name)
print(event.date)
print(event.participants)

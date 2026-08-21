import json
import os
import requests

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
base_url = os.environ.get("OPENROUTER_BASE_URL")
model = os.environ.get("OPENROUTER_MODEL")

client = OpenAI(base_url=base_url, api_key=api_key)

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""


# --------------------------------------------------------------
# Define the tool (function) that we want to call
# --------------------------------------------------------------
def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 1: Call model with get_weather tool defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for the given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the given location.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the given location.",
                    },
                },
                "required": ["latitude", "longitude"],
                "additional_properties": False,
            },
            "strict": True,
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful weather assistant."},
    {"role": "user", "content": "What's the weather in Delhi, Patna and Bangalore."},
]

completions = client.chat.completions.create(
    model=model, messages=messages, tools=tools
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------
completions.model_dump()


# --------------------------------------------------------------
# Step 3: Execute get_weather function
# --------------------------------------------------------------
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    # if name == "send_email":
    #     return send_email(**args)
    raise ValueError(f"Unknown function: {name}")


for tool_call in completions.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)  # parse JSON string
    messages.append(completions.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tools", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )


# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------
class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(description="A natural language response to user's question.")


completions_2 = client.beta.chat.completions.parse(
    model=model, messages=messages, tools=tools, response_format=WeatherResponse
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completions_2.choices[0].message.parsed
print(final_response.temperature)
print(final_response.response)

import json
import os

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
base_url = os.environ.get("OPENROUTER_BASE_URL")
model = os.environ.get("OPENROUTER_MODEL")

client = OpenAI(base_url=base_url, api_key=api_key)

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""


def search_kb(question):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------
# Step 1: Call model with search_kb tool defined
# --------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Answer the question from user using the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User's question"}
                },
                "required": ["question"],
                "additonalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are an helpful assistant that answers question from knowledge base about the e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(model=model, messages=messages, tools=tools)


# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------
completion.model_dump

# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)
    raise ValueError(f"Unknown function: {name}")


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------


class KBResponse(BaseModel):
    answer: str = Field(description="Answer to the user's question.")
    source: str = Field(description="The record id of the answer.")


completion_2 = client.beta.chat.completions.parse(
    model=model, messages=messages, tools=tools, response_format=KBResponse
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------
final_response = completion_2.choices[0].message.parsed
print(final_response.answer)
print(final_response.source)

# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather of Delhi?"},
]

completion_3 = client.chat.completions.parse(
    model=model, messages=messages, tools=tools, response_format=KBResponse
)
print(completion_3.choices[0].message.content)

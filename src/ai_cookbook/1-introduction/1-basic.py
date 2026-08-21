import os

from openai import OpenAI
from dotenv import load_dotenv

# When direct openAI tokens available
# client = OpenAI() ## OpenAI(api_key="OPENAI_API_KEY")

# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are an assistant"},
#         {"role": "user", "content": "Write a essay about python"},
#     ],
# )

# response = completion.choices[0].message.content
# print(response)

## Using OpenRouter
api_key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# First call with reasoning
response = client.chat.completions.create(
    model="openrouter/free",
    # model= "openai/gpt-4o:free",
    # Add stream= True to your request body to receive responses as server-sent events useful for Interactive experiences:
    # stream= True,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a limerick about Python programming."},
    ],
    extra_body={"reasoning": {"enabled": True}},
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Extract the assistant message for streaming output
# Normally, print() adds a newline after each call.
# end="" overrides that, so the text is printed continuously on the same line.
# makes the streamed response appear as a flowing sentence

# for chunk in response:
#     print(chunk.choices[0].delta.get("content",""), end="")

# Preserve the assistant message with reasoning_details

message = [
    {"role": "user", "content": "Write a limerick about Python programming."},
    {
        "role": "assistant",
        "content": response.content,
        "reasoning_details": response.reasoning_details,  # pass back raw
    },
    {"role": "user", "content": "Can you get me a better sounding one?"},
]

response2 = client.chat.completions.create(
    model="openrouter/free",
    messages=message,
    extra_body={"reasoning": {"enabled": True}},
)

content = response2.choices[0].message.content
print(content)

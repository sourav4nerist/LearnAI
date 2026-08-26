from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

import os
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
base_url = os.environ.get("OPENROUTER_BASE_URL")
model = os.environ.get("OPENROUTER_MODEL")

client = OpenAI(base_url=base_url, api_key=api_key)

# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event.")
    is_calendar_event: bool = Field(
        description="Wheather this text describes a calendar event."
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1.")


class EventDetails(BaseModel):
    """Second LLM call: parse specific event detials"""

    name: str = Field(description="Name of the event.")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Event duration in minutes.")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message."
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable."
    )


# --------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------


def extract_event_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is calendar event."""
    logger.info("Starting event extraction analysis")
    logger.info(f"user input: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}"

    messages = [
        {
            "role": "system",
            "content": f"{date_context} Analyze if the message is a calendar event and return only valid JSON matching EventExtraction schema.",
        },
        {"role": "user", "content": user_input},
    ]

    completion = client.beta.chat.completions.parse(
        model=model, messages=messages, response_format=EventExtraction
    )

    result = completion.choices[0].message.parsed
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result


def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract event details"""
    logger.info("Starting event details parsing")
    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}"
    completion = None
    result = None

    messages = [
        {
            "role": "system",
            "content": f"{date_context}, Extract detailed event information from the description provided and return only valid JSON matching EventDetails schema. Use provided current date as reference in case of relative date references.",
        },
        {"role": "user", "content": description},
    ]
    try:
        completion = client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=EventDetails
        )
        result = completion.choices[0].message.parsed
        logger.info(
            f"Parsed event details- Name: {result.name}, date: {result.date}, duration: {result.duration_minutes} min"
        )
        logger.info(f"Participants: {','.join(result.participants)}")
    except ValidationError as e:
        logger.error(f"Invalid JSON from model: {e}")
        logger.error("fallback: parse manually or retry with clearer instructions")

    return result


def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate event confirmation message."""
    logger.info("Generating event confirmation message.")

    messages = [
        {
            "role": "system",
            "content": "Generate a natural language event conformation message. Sign of with the name: Sourav  and return only valid JSON matching EventConfirmation schema.",
        },
        {"role": "user", "content": str(event_details.model_dump)},
    ]

    completion = client.beta.chat.completions.parse(
        model=model, messages=messages, response_format=EventConfirmation
    )

    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated.")
    return result


# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing prompt chain with gate check"""
    logger.info("Processing user request")
    logger.info(f"Raw user input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    # Gate check: verify if it's calendar event request with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score}"
        )
        return None

    logger.info("Gate check pass, processing event request")

    # Second LLM call: Extract event related info
    event_details = parse_event_details(initial_extraction.description)

    # Third LLM call: Generate confirmation
    event_confirmation = generate_confirmation(event_details)

    logger.info("Calendar request completed")
    return event_confirmation


# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

user_input = "Let's schedule a 1 hour call next Tuesday at 11am with Dev and Divakar to discuss project roadmap."

result = process_calendar_request(user_input)

if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar link: {result.calendar_link}")
else:
    print(f"This message is not a calendar event request.")

# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

# user_input = "Send an email to Dev and Divakar to discuss project roadmap."

# result = process_calendar_request(user_input)

# if result:
#     print(f"Confirmation: {result.confirmation_message}")
#     if result.calendar_link:
#         print(f"Calendar link: {result.calendar_link}")
# else:
#     print(f"This message is not a calendar event request.")

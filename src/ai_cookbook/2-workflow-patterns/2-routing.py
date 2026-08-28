from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

import os
import logging

# Setup logger
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
# Step 1: Define the data models for routing and responses
# --------------------------------------------------------------


class CalendarRequestType(BaseModel):
    """Router LLM call: Determine the type of calendar request"""

    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request made by user."
    )
    confidence_score: bool = Field(description="Confidence score between 0 and 1.")
    description: str = Field(description="Cleaned description of the request.")


class NewEventDetails(BaseModel):
    """Details for creating a new event"""

    name: str = Field(description="Name of the event.")
    date: str = Field(description="Date and time of the event. Use ISO 8601.")
    duration: int = Field(description="Duration of the event in minutes.")
    participants: list[str] = Field(description="List of participants for the event.")


class Change(BaseModel):
    """Details for changing an existing event"""

    field: str = Field(description="Field to be modified.")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event"""

    event_identifier: str = Field(
        description="Description that identifies the existing event."
    )
    changes: list[Change] = Field(description="List of changes to be made.")
    add_participants: list[str] = Field(description="List of new participants to add.")
    remove_participants: list[str] = Field(
        description="List of existing participants to remove."
    )


class CalendarResponse(BaseModel):
    """Final response format"""

    success_indicator: bool = Field(
        description="Wheather the operation was successful."
    )
    message: str = Field(description="Natural language response message.")
    calendar_link: str = Field(description="Calendar link if applicable.")


# --------------------------------------------------------------
# Step 2: Define the routing and processing functions
# --------------------------------------------------------------


def route_calendar_request(user_request: str) -> CalendarRequestType:
    """Router LLM call to determine the type of calendar request"""
    logger.info("Routing calendar request")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Analyze if the request is to create a new calendar event or to modify existing event and retrun response in CalendarRequestType schema.",
            },
            {"role": "user", "content": user_request},
        ],
        response_format=CalendarRequestType,
    )

    result = completion.choices[0].message.parsed
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
    )
    return result


def handle_new_event(description: str) -> NewEventDetails:
    """Process a new event request"""
    logger.info("Processing new event request")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract details to create new event and retrun response in NewEventDetails schema.",
            },
            {"role": "user", "content": description},
        ],
        response_format=NewEventDetails,
    )

    details = completion.choices[0].message.parsed

    logger.info(f"New event: {details.model_dump_json}")

    # Generate response
    return CalendarResponse(
        success_indicator=True,
        message=f"Created new event {details.name} for {details.date} with {','.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}",
    )


def handle_modify_event(description: str) -> CalendarResponse:
    """Process an event modification request"""
    logger.info("Processing event modification request")

    # Get modification details

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract details to modify existing calendar event and retrun response in ModifyEventDetails schema.",
            },
            {"role": "user", "content": description},
        ],
        response_format=ModifyEventDetails,
    )

    details = completion.choices[0].message.parsed
    logger.info(f"Modified event: {details.model_dump_json}")

    # Generate response
    return CalendarResponse(
        success_indicator=True,
        message=f"Modified event: {details.event_identifier} with requested changes.",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    """Main function implementing the routing workflow"""
    logger.info("Processing calendar request")

    # route the request
    route_result = route_calendar_request(user_input)

    # Check confidence score
    if route_result.confidence_score < 0.7:
        logger.warning(f"Low confidence score: {route_result.confidence_score}")
        return None

    # Route to appropriate handler
    if route_result.request_type == "new_event":
        return handle_new_event(route_result.description)
    elif route_result.request_type == "modify_event":
        return handle_modify_event(route_result.description)
    else:
        logger.warning("Request type not supported")
        return None


# --------------------------------------------------------------
# Step 3: Test with new event
# --------------------------------------------------------------

new_event_input = "Let's schedule a meeting with Dev and Divakar next Tuesday at 12pm."

result = process_calendar_request(new_event_input)

if result:
    print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 4: Test with modify event
# --------------------------------------------------------------
modify_event_input = (
    "Please reschedule the call with Dev and Divakar to next Wednesday at 3pm."
)
result = process_calendar_request(modify_event_input)

if result:
    print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 5: Test with invalid request
# --------------------------------------------------------------
invalid_input = "What's the time right now?"
result = process_calendar_request(invalid_input)

if result:
    print(f"Response: {result.message}")

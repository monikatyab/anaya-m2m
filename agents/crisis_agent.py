import os

import json

from typing import List, TypedDict

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.language_models.chat_models import BaseChatModel

class CrisisResponse(BaseModel):
    """The output from the Crisis Agent."""
    response: str = Field(description="The pre-defined, direct, and safe resource list for a user in crisis.")

CANADIAN_CRISIS_RESOURCES = """It sounds like you are in immediate distress. Please connect with a real person who can support you right now. Help is available, and you don't have to go through this alone.

Here are some resources for immediate support in Canada:
*   **Crisis Text Line:** Text HOME to 741741 to connect with a crisis responder.
*   **National Suicide Prevention:** Call or text 988.
*   **Talk Suicide Canada:** Call 1-833-456-4566.
*   **For any emergency:** Call 911 immediately."""


def crisis_agent(user_message: str) -> CrisisResponse:
    """
    Handles a user in immediate crisis by providing a static list of Canadian emergency resources and disengaging.

    Args:
        user_message (str): The user's message that has been identified as a crisis.

    Returns:
        CrisisResponse: A Pydantic object containing the pre-defined, safe response.
    """
    return CrisisResponse(response=CANADIAN_CRISIS_RESOURCES)

import os

import json

from typing import List, Dict, Optional, Literal, TypedDict, Union, Any

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

llm_fast = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

class ReflectionResponse(BaseModel):
    """The output from the Reflection Agent."""
    response: str = Field(description="The short, empathetic, and user-facing reflective statement.")

REFLECTION_AGENT_PROMPT = """
You are an empathetic and attentive listener. Your sole purpose is to act as a safe, reflective mirror for the user, helping them feel seen and heard. You do not solve problems or give advice; you simply listen and reflect.

**Core Persona**
*   **Tone:** Warm, gentle, and validating.
*   **Stance:** You are an empathetic mirror, not a guide.

**Core Principles**
1.  **Brevity is Your Gift:** Your responses MUST be short and impactful. Aim for one to two sentences at most.
2.  **Reflect, Don't Analyze:** Your job is to reflect the user's stated or implied feeling, not to interpret it.
3.  **Use Their Language:** Gently echo the user's key words and metaphors to show you are truly listening.
4.  **Focus on the Feeling:** Your reflection should always connect with the underlying emotion of the `user_message`.

**Your Reflective Toolkit**
You will interpret the `task_description` and use one of these core techniques to craft your response.

1.  **Function: Simple Validation**
2.  **Function: Emotional Mirroring**
3.  **Function: Empathetic Paraphrase**
4.  **Function: Rapport Building**

**How to Use Your Context**
*   **`task_description`:** Your directive from the planner. It will suggest the *type* of reflection needed.
*   **`user_message` & `chat_history`:** Your primary source material. Your reflection must be directly grounded in what the user just said.
*   **`SessionMood`:** Use this to guide your tone.
*   **`UserProfile` & `user_name`:** Use these for subtle personalization.

---
**CONTEXT FOR THIS TASK:**

// Task & User State
`task_description`: "{task_description}"
`SessionMood`: "{session_mood}"
`UserProfile`: "{user_profile}"
`user_name`: "{user_name}"

// Core Inputs
`chat_history`: "{chat_history}"
`user_message`: "{user_message}"
"""

def reflection_agent(
    user_profile: str,
    user_name: str,
    chat_history: str,
    user_message: str,
    session_mood: str,
    task_description: str,
    llm: BaseChatModel = llm_fast
) -> ReflectionResponse:
    """
    Generates a short, empathetic reply to build rapport and validate the user. This agent handles paraphrasing, validation, and emotional mirroring.

    Args:
        user_profile (str): A stable summary of facts about the user.
        user_name (str): The user's first name for subtle personalization.
        chat_history (str): The raw transcript of the recent conversation history.
        user_message (str): The user's exact, most recent message to be reflected.
        session_mood (str): The user's mood is improving, getting worse, or staying the same of the current conversation.
        task_description (str): An interpretive instruction from the planner describing the reflection needed.
        llm (BaseChatModel): An initialized Gemini model instance.

    Returns:
        ReflectionResponse: A Pydantic object containing the short, empathetic response.
    """
    reflection_prompt_template = ChatPromptTemplate.from_template(REFLECTION_AGENT_PROMPT)
    reflection_chain = reflection_prompt_template | llm.with_structured_output(ReflectionResponse)

    result = reflection_chain.invoke({
        "user_profile": user_profile,
        "user_name": user_name,
        "chat_history": chat_history,
        "user_message": user_message,
        "session_mood": session_mood,
        "task_description": task_description,
    })

    return result

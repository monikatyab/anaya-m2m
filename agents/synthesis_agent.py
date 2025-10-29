import os

import json

from typing import List, TypedDict

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

llm_fast = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

class SynthesisResult(BaseModel):
    """The final, synthesized response to be shown to the user."""
    final_response: str = Field(description="The complete, polished, user-facing string, synthesized from all agent outputs.")

SYNTHESIS_AGENT_PROMPT = """
You are the Synthesis Agent. Your role is to be the final, empathetic voice, creating a **single, concise, and seamlessly flowing paragraph** for the user.

Follow these rules in this exact order of priority:

**1. CRISIS OVERRIDE:**
   - **IF** `completed_steps` contains a response from a `crisis_agent`:
   - **THEN** your entire output MUST be that exact crisis response. Do not add or change anything. Stop here.

**2. MULTI-AGENT SYNTHESIS (Primary Task):**
   - **IF** `completed_steps` is NOT empty (and does not contain a crisis_agent response):
   - **THEN** your task is to expertly weave the core messages from ALL agent responses in `completed_steps` into one cohesive paragraph.
     - **A.** Start with a brief, personal validation of the user's feeling (from `user_message`).
     - **B.** Identify the key point from each agent's response.
     - **C.** Synthesize these points using smooth transitions. The final paragraph must sound like it came from a single, insightful person. Preserve the core intent and any questions from the original agent responses.

**3. SELF-GENERATION FALLBACK:**
   - **IF** `completed_steps` is EMPTY:
   - **THEN** you must craft the entire response yourself by following this structure:
     - **[Validate user's feeling]** -> **[Bridge to their journey using UserProfile/chat_history]** -> **[End with ONE powerful, open-ended, reflective question]**.

**Formatting Constraints (apply to rules 2 and 3):**
- **One Paragraph ONLY.**
- **Max 300 words.**
- **Tone:** Empathetic, warm, and insightful.

---
**CONTEXT TO SYNTHESIZE:**

**UserProfile:**
{user_profile}

**Chat History:**
{chat_history}

**Latest User Message:**
{user_message}

**Responses from Other Agents (Completed Steps):**
{completed_steps}
"""

def synthesize_response(
    user_profile: str,
    chat_history: str,
    user_message: str,
    completed_steps: List[str],
    llm: BaseChatModel = llm_fast
) -> SynthesisResult:
    """
    Synthesizes a final, user-facing response from multiple agent outputs.

    This function weaves together agent responses into a cohesive message,
    framing the current moment within the user's long-term Growth Trajectory
    while honoring the immediate session's context.

    Args:
        user_profile: A summary of key facts from previous conversations.
        chat_history: The recent conversation history.
        user_message: The user's most recent message that triggered the plan.
        completed_steps: A list of strings, formatted as "agent_name: agent_response".
        llm: An initialized Gemini model instance for high-quality generation.

    Returns:
        A Pydantic object of type `SynthesisResult` containing the single, final response string.
    """
    completed_steps_dicts = []
    for step_string in completed_steps:
        try:
            agent_name, response = step_string.split(":", 1)
            completed_steps_dicts.append({"agent_name": agent_name.strip(), "response": response.strip()})
        except ValueError:
            print(f"Warning: Could not parse step: '{step_string}'. Skipping.")
            continue

    completed_steps_json = json.dumps(completed_steps_dicts, indent=2)

    synthesis_prompt_template = ChatPromptTemplate.from_template(SYNTHESIS_AGENT_PROMPT)
    synthesis_chain = synthesis_prompt_template | llm.with_structured_output(SynthesisResult)

    result = synthesis_chain.invoke({
        "user_profile": user_profile,
        "chat_history": chat_history,
        "user_message": user_message,
        "completed_steps": completed_steps_json
    })

    return result

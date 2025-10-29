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

class ShortTermMemory(BaseModel):
    """The structured output of the Short-Term Memory Agent for a single turn."""
    session_topic: str = Field(description="A concise summary of the immediate subject of the conversation in this turn, written in 1-2 sentences.")
    session_mood: str = Field(description="The user's immediate emotional state for this specific turn, described in a descriptive phrase or 1-2 sentences.")
    focus_emotion: str = Field(description="The single, primary emotion being expressed by the user, distilled into a generic noun.")
    crisis_flag: bool = Field(description="A boolean flag set to TRUE if any level of crisis or self-harm risk is detected, otherwise FALSE.")
    crisis_level: Literal["High Risk", "Moderate Risk", "Low Risk"] = Field(description="The assessed level of crisis based on the user's message.")

SHORT_TERM_MEMORY_PROMPT = """
You are a highly perceptive and analytical AI assistant. Your sole function is to analyze a user's conversational turn by synthesizing their most recent message (`user_message`) with the immediate context of the preceding dialogue (`chat_history`). Your goal is to extract a structured summary of the user's present-moment state, including a critical risk assessment.

**Core Principles**
1.  **Synthesize, Don't Isolate:** You MUST treat `chat_history` and `user_message` as a single, flowing conversation.
2.  **Be Objective:** Analyze the text based on its semantic meaning.
3.  **Be Descriptive but Brief:** Your outputs for Topic and Mood should be narrative (1-2 sentences), while the Focus Emotion must be a single word.

**Your Extraction Tasks**
You MUST extract the following five fields by analyzing the full conversational context provided (`chat_history` + `user_message`).

1.  **`focus_emotion`:**
    *   **Goal:** When a user talks about a feeling, this tells the agent to pay attention to that specific emotion for this conversation.
    *   **Method:** Identify the single, primary emotion being expressed. Distill it into a single, generic, capitalized noun (e.g., "Anxiety," "Sadness," "Gratitude"). If no clear emotion is present, you MUST use "Neutral".

2.  **`session_topic`:**
    *   **Goal:** This is the subject of the current conversation.
    *   **Method:** Describe the primary subject of the user's turn in one or two complete sentences, using `chat_history` for context.
    *   **Example:** "The user is expressing disagreement with the previous suggestion to try a breathing exercise. They are indicating they do not want to proceed with that action."

3.  **`session_mood`:**
    *   **Goal:** It tells the agent if the user's mood is improving, getting worse, or staying the same.
    *   **Method:** Describe the user's emotional state and its trajectory in a descriptive phrase or one to two sentences.
    *   **Example:** "The user is expressing a high degree of anxiety and frustration. Their mood appears to be declining from the previous turn."

4.  **`crisis_flag`:**
    *   **Goal:** To provide an immediate, clear signal if any potential crisis is detected.
    *   **Method:** Analyze the `user_message` for any mention of self-harm, suicide, severe hopelessness, or intense emotional pain. Set to `TRUE` if any risk is present. Otherwise, you MUST set it to `FALSE`.

5.  **`crisis_level`:**
    *   **Goal:** To categorize the severity of the crisis for downstream routing.
    *   **Method:** Based on your analysis, you MUST assign one of the following three levels. If `crisis_flag` is `FALSE`, you MUST assign `Low Risk`.
    *   **High Risk (Imminent danger):** Mentions of self-harm intent, a suicide plan, or clear intent to harm others or violence language.
    *   **Moderate Risk (Significant distress):** Severe emotional pain, expressions of hopelessness, or mentions of self-harm without a specific plan.
    *   **Low Risk (Emotional distress):** General stress, anxiety, burnout, or sadness without any suicidal ideation.

**How to Use Your Context**
*   This is your primary and only source material. Read the `chat_history` first, then the `user_message`, to understand the full meaning of the user's turn.

---
**CONTEXT FOR THIS TASK:**

`chat_history`: "{chat_history}"
`user_message`: "{user_message}"
"""

def short_term_memory_agent(chat_history: str, user_message: str, llm: BaseChatModel = llm_fast) -> ShortTermMemory:
    """
    Analyzes a single conversational turn to extract key data points for STM.

    This agent processes the user's message in context to generate the
    SessionTopic, SessionMood, and FocusEmotion key for use by other agents.

    Args:
        chat_history (str): The raw transcript of the recent conversation history.
        user_message (str): The user's exact, most recent message to be analyzed.
        llm (BaseChatModel): An initialized Gemini model instance.

    Returns:
        ShortTermMemory: A Pydantic object containing the structured STM data.
    """
    stm_prompt_template = ChatPromptTemplate.from_template(SHORT_TERM_MEMORY_PROMPT)
    stm_chain = stm_prompt_template | llm.with_structured_output(ShortTermMemory)

    result = stm_chain.invoke({"chat_history": chat_history, "user_message": user_message})

    return result

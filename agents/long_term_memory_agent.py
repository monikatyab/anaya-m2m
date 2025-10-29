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

class NewMemorySnapshot(BaseModel):
    date: str = Field(description="The date of the conversation.")
    intensity: str = Field(description="The inferred intensity of the emotion (e.g., 'High', 'Manageable', 'Low').")
    user_description: str = Field(description="The user's key words for the feeling.")
    session_insight: str = Field(description="The core insight the user gained about the feeling.")

class LTMAnalysisResult(BaseModel):
    """The raw analysis of a conversation, used to update the LTM."""
    UserJourney: str = Field(description="The single, new sentence that describes the user's most up-to-date stage of progress.")
    somatic_focus: str = Field(description="Where in their body the user felt the emotion (e.g., 'Chest and throat', 'Stomach').")
    awareness_shift: str = Field(description="The specific 'aha moment' or reflection the user had, described in a single sentence.")
    support_preference: Literal["somatic", "reflective", "cognitive", "spiritual"] = Field(description="The primary modality of support the user engaged with most in this session.")
    identified_helpful_tools: List[str] = Field(description="A list of tools the user explicitly found helpful in this conversation.")
    identified_unhelpful_tools: List[str] = Field(description="A list of tools the user explicitly found unhelpful in this conversation.")
    new_guiding_intentions: str = Field(description="The single most important new long-term goal the user expressed, or an empty string if none.")
    new_memory_snapshot: NewMemorySnapshot = Field(description="The new memory snapshot object containing the core details.")

LONG_TERM_MEMORY_PROMPT = """
You are a highly insightful and analytical Narrative Psychologist. Your task is to review a user's `chat_history` and extract key pieces of new information that will be used to update their profile.

**Core Principles**
1.  **Synthesize and Extract:** Your job is to synthesize a new `UserJourney` sentence and extract all other new information.
2.  **Be Objective:** Base your analysis solely on the provided transcript and previous journey.
3.  **Distill the Essence:** Focus on the most significant new insights, feedback, and goals.

**Your Extraction & Synthesis Tasks**
You MUST generate the following fields. The examples provided are based on a hypothetical conversation and should be used as a structural and stylistic guide.

1.  **`UserJourney`:**
    *   **Action:** First, read the `previous_user_journey` list to understand the user's recent trajectory. Then, read the new `chat_history`. Synthesize **both** sources of information to write a **single, new sentence** that describes the user's most up-to-date stage of progress.
    *   **Example Output:** `"The user has evolved from identifying their anxiety as a physical sensation to interpreting it as a meaningful signal from their body to rest."`

2.  **`somatic_focus`:**
    *   **Action:** Identify where in the user's body they felt the primary emotion during the session.
    *   **Example Output:** `"Chest and shoulders"`

3.  **`awareness_shift`:**
    *   **Action:** Describe the single most important "aha moment" or shift in perspective the user experienced.
    *   **Example Output:** `"The moment they realized they could listen to the feeling instead of fighting it."`

4.  **`support_preference`:**
    *   **Action:** Analyze the conversation to determine the primary way the user engaged. You MUST choose the one that best fits from the following list: `somatic`, `reflective`, `cognitive`, `spiritual`.
        *   `somatic`: If the user focused most on body sensations and physical feelings.
        *   `reflective`: If the user focused most on memories, insights, and self-reflection.
        *   `cognitive`: If the user focused most on changing thoughts, reframing, and planning.
        *   `spiritual`: If the user focused most on meaning, purpose, and connection to something larger.
    *   **Example Output:** `"somatic"`

5.  **`identified_helpful_tools`:**
    *   **Action:** List the names of any tools the user explicitly said were helpful.
    *   **Example Output:** `["Somatic Tracking"]`

6.  **`identified_unhelpful_tools`:**
    *   **Action:** List the names of any tools the user explicitly said were unhelpful. If none, return an empty list.
    *   **Example Output:** `[]`

7.  **`new_guiding_intentions`:**
    *   **Action:** Identify the single most important new, core long-term goal the user expressed. If no new primary goal is mentioned, return an empty string.
    *   **Example Output:** `"Find more ways to feel less alone"`

8.  **`new_memory_snapshot`:**
    *   **Action:** Generate a **nested JSON object** for the new memory snapshot with the following exact keys: `date`, `intensity`, `user_description`, and `session_insight`. Note that somatic_focus and awareness_shift are separate, top-level fields.
    *   **Example Output:**
        ```json
        {{
          "date": "2024-05-30",
          "intensity": "Manageable",
          "user_description": "'buzzing engine' feeling",
          "session_insight": "Realized the feeling is not an enemy, but a signal from their body to rest."
        }}
        ```
---
**CONTEXT FOR THIS TASK:**

// Read-Only Context
`UserProfile`: "{user_profile}"

// Previous Long-Term Memory State
`previous_user_journey`: {previous_user_journey}
`previous_personal_toolkit`: {previous_personal_toolkit}
`previous_guiding_intentions`: {previous_guiding_intentions}
`previous_memory_threads`: {previous_memory_threads}

// New Information to Process
`chat_history`:
---
{chat_history}
---
`focus_emotion`: "{focus_emotion}"
`current_date`: "{current_date}"
"""

def analyze_conversation_for_ltm(
    user_profile: str,
    previous_user_journey: List[str],
    previous_personal_toolkit: Dict[str, List[str]],
    previous_guiding_intentions: List[str],
    previous_memory_threads: Dict[str, List[Dict]],
    chat_history: str,
    focus_emotion: str,
    current_date: str,
    llm: BaseChatModel = llm_fast
) -> LTMAnalysisResult:
    """
    Analyzes a conversation transcript to synthesize a new journey and extract insights.
    """
    ltm_prompt_template = ChatPromptTemplate.from_template(LONG_TERM_MEMORY_PROMPT)
    ltm_chain = ltm_prompt_template | llm.with_structured_output(LTMAnalysisResult)

    analysis_result = ltm_chain.invoke({
        "user_profile": user_profile,
        "previous_user_journey": json.dumps(previous_user_journey, indent=2),
        "previous_personal_toolkit": json.dumps(previous_personal_toolkit, indent=2),
        "previous_guiding_intentions": previous_guiding_intentions,
        "previous_memory_threads": json.dumps(previous_memory_threads, indent=2),
        "chat_history": chat_history,
        "focus_emotion": focus_emotion,
        "current_date": current_date
    })

    return analysis_result

def consolidate_memory(
    analysis: LTMAnalysisResult,
    previous_user_journey: List[str],
    previous_personal_toolkit: Dict[str, List[str]],
    previous_guiding_intentions: List[str],
    previous_memory_threads: Dict[str, List[Dict]],
    focus_emotion: str,
    current_date: str
) -> Dict:
    """
    Takes the LLM's analysis and reliably updates the old LTM state.
    """

    new_journey = previous_user_journey.copy()
    new_toolkit = {k: v.copy() for k, v in previous_personal_toolkit.items()}
    new_intentions = previous_guiding_intentions.copy()
    new_memory_threads = {k: v.copy() for k, v in previous_memory_threads.items()}

    # UserJourney Update
    new_journey.append(f"{current_date}: {analysis.UserJourney}")

    # PersonalToolkit Update
    new_toolkit["user_found_helpful"].extend([tool for tool in analysis.identified_helpful_tools if tool not in new_toolkit["user_found_helpful"]])
    new_toolkit["user_found_unhelpful"].extend([tool for tool in analysis.identified_unhelpful_tools if tool not in new_toolkit["user_found_unhelpful"]])

    # GuidingIntentions Update
    new_intentions.append(analysis.new_guiding_intentions)

    # MemoryThreads Update
    new_memory_threads.setdefault(focus_emotion, []).append(analysis.new_memory_snapshot.model_dump())

    return {
        "UserJourney": new_journey,
        "PersonalToolkit": new_toolkit,
        "GuidingIntentions": new_intentions,
        "MemoryThreads": new_memory_threads
    }

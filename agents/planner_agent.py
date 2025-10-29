import os

import json

from typing import List, Dict, Optional, Literal, TypedDict, Union, Any

from dotenv import load_dotenv

from pydantic import BaseModel, Field, conint

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.language_models.chat_models import BaseChatModel


import logging

load_dotenv()

llm_fast = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

llm_powerful = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

class ExecutionStep(BaseModel):

    agent_name: str = Field(description="The name of the specific agent to be invoked for this task.")
    agent_inputs: str = Field(description="The input parameters required by the selected agent, formatted as a string. For a search agent, this would be the search query.")
    rationale: str = Field(description="The step-by-step reasoning or 'chain of thought' that justifies this action. Explains how this step contributes to achieving the overall goal.")
    personal_flag: bool = Field(description="Set to `true` if the inputs or expected outputs of this step are likely to contain sensitive personal information (e.g., names, emails, addresses).")

class Plan(BaseModel):

    question: str = Field(description="The user's initial query that this entire execution plan is designed to answer.")
    execution_plan: List[ExecutionStep] = Field(description="The ordered list of agent actions to be performed. This represents the complete, step-by-step strategy to fulfill the user's request.")

PLANNER_PROMPT = """
You are a conversational planner and an intelligent information extractor. Read `chat_history` and `user_message`. You also serve these roles:

1.  **Context historian:** Read the whole `chat_history`. Carry forward unresolved items like risk, requests, and preferences.
2.  **Semantic analyst:** Read meaning and intent from wording and context, not keywords.
3.  **Perspective composer:** Split the user turn into at focused perspectives in the order they were said.

**Available Agents:** [`"crisis_agent"`, `"factual_responder_agent"`, `"dialogue_manager_agent"`, `"wellness_assistant_agent"`, `"reflection_agent"`]

**Principles (Context First, Flexible)**
1.  Read the whole conversation, then the new message. **MUST** use semantic meaning and context over keywords. Treat guidance as helpful, not absolute.
2.  Keep the plan short and purposeful. Create at most four perspectives in the same order the user expressed ideas.
3.  Respect user autonomy. Use the lightest step that keeps the conversation safe and useful.
4.  When unsure, ask one clear clarifying question using `dialogue_manager_agent`.

**Speaker and Question Source (Strict, Hard Constraint)**
1.  **Question binding (highest priority)**
    Let q = `user_message` stripped of leading/trailing whitespace.
    If q is non-empty, set `"question"` to q **EXACTLY**. Do not paraphrase, summarize, normalize punctuation, or pull from `chat_history`. This rule overrides all other guidance.

2.  **Speaker mapping for `chat_history` (fallback ONLY when `user_message` is empty)**
    *   Lines starting with "You:" or "User:" are user turns.
    *   Lines starting with "ChatBot:", "Assistant:", "System:", or any named persona label like "Anaya:" are NOT user turns.
    *   For fallback, select the most recent "You:"/"User:" line and set `"question"` to the text after the label, trimmed.

3.  Never set `"question"` to any line that begins with "ChatBot:", "Assistant:", "System:", or a named persona label (e.g., "Anaya:").

**Speaker and Question Source (ABSOLUTE RULE, HIGHEST PRIORITY)**
1.  The `question` field in the output `JSON` **MUST** be the exact, unmodified text from the `user_message` input (after trimming whitespace).
2.  This is a **mandatory, non-negotiable instruction**. Do **NOT** use any text from `chat_history` for the `question` field if the `user_message` input is not empty.
3.  **Fallback for empty `user_message`**: **ONLY** if the `user_message` input is completely empty, find the most recent "User:" or "You:" line from the `chat_history` and use its text for the `question` field.
4.  Never set `"question"` to any line that begins with "ChatBot:", "Assistant:", "System:", or a named persona label (e.g., "Anaya:").

**Context-Driven Routing Cues**
1.  **Safety cue.** If the current message clearly signals imminent self harm, danger, or a credible threat, choose `crisis_agent` as a single step. If recent `chat_history` includes explicit risk but the current message is neutral or an exit, you may add one brief consent based safety check via `dialogue_manager_agent`. Combine it with exit handling when possible. Do not escalate to `crisis_agent` unless the current message itself signals risk.
2.  **Exit cue.** When the user signals leaving, prefer `dialogue_manager_agent` to manage the exit. If they also ask a quick question, place the exit or optional safety check first, then answer. If the exit is social only, `reflection_agent` may acknowledge instead.
3.  **Negation cue.** A short "No" or "Not now" to a prior assistant question maps to `dialogue_manager_agent` to cancel or update state.
4.  **Mixed Intent Cue (Emotional + Factual/Task).** This is a **strict rule**. If a user message contains both an emotional/personal part AND a distinct factual question or task, you **MUST** create two steps in the order they appeared.
    *   **Step 1:** Use `reflection_agent` or `wellness_assistant_agent` to address the emotional/personal content.
    *   **Step 2:** Use the appropriate agent (e.g., `factual_responder_agent`) for the factual/task-oriented part.
    *   Do not skip the first emotional step, even if the second part is a simple question.
5.  **Personal data cue.** When a step needs to access, confirm, or reveal any personal identifier, set `personal_flag` to `true` and route that step to `dialogue_manager_agent`.
6.  **Emotion cue.** When feelings, coping, confidence, triggers, or therapeutic guidance are central to the meaning, prefer `wellness_assistant_agent`. When the aim is a neutral fact or a techniques checklist, prefer `factual_responder_agent`.

**Agent Mapping Guidance**
1.  `crisis_agent` handles imminent risk and explicit self harm or violence language or suicide plan, or harming others.
2.  `wellness_assistant_agent` handles content where emotion, mood, coping, triggers, or therapeutic support are central to the meaning.
3.  `factual_responder_agent` handles neutral factual or definitional queries and skill building techniques such as checklists, methods, or best practices where a direct answer is the point.
4.  `reflection_agent` handles brief empathic mirrors, validation, and acknowledgement when no next step action is required.
5.  `dialogue_manager_agent` handles routing, clarifying intent, session state changes, permission asks, exit handling, and any step with `personal_flag` set to `true`.

**Skill-Building and Mixed Intent Handling**
1.  If the user asks about improving a skill and also expresses emotions (for example feeling unready, anxious, nervous, worried), produce three ordered perspectives:
    *   `reflection_agent` — brief validation of feelings and effort.
    *   `wellness_assistant_agent` — coping and confidence-building tailored to the skill context.
    *   `factual_responder_agent` — concrete techniques, checklists, and practice plans.
    Do not skip the `wellness_assistant_agent` step in this pattern, even if a factual answer seems obvious.
2.  If the user asks only for skills with a neutral tone, route directly to `factual_responder_agent` and omit reflection and wellness.
3.  If emotion is present but no technique is requested, use `reflection_agent` then `wellness_assistant_agent` only.
4.  If routing confidence is low at any point, add a `dialogue_manager_agent` step to ask one concise clarifying question before providing techniques.
5.  If the user shares feelings without a direct request, use two steps in order: `reflection_agent` then `wellness_assistant_agent`.

**Emotion Guidance**
1.  Treat detected emotion as a helpful signal, not a hard rule. Clear intent overrides uncertain emotion reads.
2.  If emotion is unclear, assume neutral and note low confidence in the rationale when relevant.
3.  If language or clear emotion indicates imminent risk, escalate to `crisis_agent` immediately.

**Tie-Breakers and Behavior**
1.  Crisis overrides all other rules. If `crisis_agent` is selected, stop and do not add further steps.
2.  If an idea is both emotional and factual, choose `wellness_assistant_agent` when feelings or coping are central. Choose `factual_responder_agent` when the user clearly wants a neutral fact.
3.  For mixed emotional and factual content, prefer two steps in order. First `reflection_agent` to acknowledge, then `wellness_assistant_agent` to answer with brief factual support if needed.
4.  If routing confidence is low, choose `dialogue_manager_agent` to ask one clear clarifying question.
5.  Preserve the user’s order of ideas. Handle exit or acknowledgement before giving answers.

**Step Output Instructions**
**IMPORTANT FINAL CHECK:** Before outputting the `JSON`, you **MUST** verify the `question` field.
1.  The `question` field's value **MUST** be an exact copy of the `user_message` input if `user_message` is not empty. This is your most important instruction. If your chosen `question` does not exactly match `user_message`, you have made a mistake and must correct it.
2.  Each execution step must include these fields exactly: `agent_name`, `agent_inputs`, `rationale`, `personal_flag`.
3.  `agent_inputs` must be a short perspective string that names the angle the agent should handle. Use one sentence.
4.  `rationale` must be one sentence that explains why this agent was chosen. You may mention emotion signals and confidence briefly.
5.  `personal_flag` is `true` whenever the step reads, confirms, writes, or repeats any user identifier, even if the identifier is not printed in `agent_inputs`. If `personal_flag` is `true`, `agent_name` must be `dialogue_manager_agent`.

**Formatting Rules**
1.  Produce a `JSON` object with the original `question` and an ordered `execution_plan` of one to four steps.
2.  Do not include schema text, timestamps, ids, `completed_steps`, or extra metadata.
3.  Keep all text plain, concise, and actionable.

**Context Available**
`chat_history`: {chat_history}
`user_message`: {user_message}
"""

def generate_execution_plan(user_message: str, chat_history: str, llm: BaseChatModel = llm_powerful) -> Plan:
    """
    Generates a structured execution plan for a conversational turn.

    This function takes the user's message and the conversation history,
    uses a powerful LLM to analyze the context, and produces a step-by-step
    plan for which agents should respond and in what order.

    Args:
        user_message: The latest message from the user.
        chat_history: A string representing the prior conversation history. (e.g., "User: ...\nChatbot: ...\nUser: ...")
        llm: An initialized Gemini model instance to use for planning.

    Returns:
        A Pydantic object of type `Plan` containing the user's original
        message and an ordered list of execution steps.
    """
    # Create the prompt template with the input variables
    planner_prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    # Create the planning chain, binding the LLM and the structured output model
    planner_chain = planner_prompt_template | llm.with_structured_output(Plan)

    # Invoke the chain with the provided inputs
    plan = planner_chain.invoke({"chat_history": chat_history, "user_message": user_message})

    return plan



# def generate_execution_plan(user_message: str, chat_history: str, llm: BaseChatModel = llm_powerful) -> Plan:
#     """
#     Generates a structured execution plan for a conversational turn.
#
#     This function takes the user's message and the conversation history,
#     uses a powerful LLM to analyze the context, and produces a step-by-step
#     plan for which agents should respond and in what order. It includes
#     a retry mechanism to handle transient API or parsing errors.
#
#     Args:
#         user_message: The latest message from the user.
#         chat_history: A string representing the prior conversation history. (e.g., "User: ...\nChatbot: ...\nUser: ...")
#         llm: An initialized Gemini model instance to use for planning.
#
#     Returns:
#         A Pydantic object of type `Plan` containing the user's original
#         message and an ordered list of execution steps.
#
#     Raises:
#         Exception: Re-raises the last caught exception if all retry attempts fail.
#     """
#     # --- Refinement: Added retry logic constants ---
#     MAX_RETRIES = 3
#     RETRY_DELAY_SECONDS = 1
#     last_exception = None
#
#     # --- Refinement: Added retry loop ---
#     for attempt in range(MAX_RETRIES):
#         try:
#             # The original logic is placed inside the try block
#             planner_prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)
#             planner_chain = planner_prompt_template | llm.with_structured_output(Plan)
#             plan = planner_chain.invoke({"chat_history": chat_history, "user_message": user_message})
#
#             # If successful, return the plan immediately
#             return plan
#
#         except Exception as e:
#             # --- Refinement: Added error handling ---
#             last_exception = e
#             logging.warning(
#                 f"Plan generation failed on attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}"
#             )
#             # If this wasn't the last attempt, wait before retrying
#             if attempt < MAX_RETRIES - 1:
#                 time.sleep(RETRY_DELAY_SECONDS)
#             else:
#                 # If all retries have failed, log a critical error
#                 logging.error("All retry attempts for plan generation have failed.")
#
#     # If the loop completes without returning, it means all retries failed.
#     # Re-raise the last exception so the calling code knows about the failure.
#     raise last_exception

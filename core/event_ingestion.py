import os
import json
import uuid
from datetime import datetime
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from typing import List, Dict, Optional, Literal, TypedDict, Union, Any

from agents.short_term_memory_agent import short_term_memory_agent
from agents.long_term_memory_agent import analyze_conversation_for_ltm, consolidate_memory
from agents.state import WorkflowState

def short_term_memory_event_log(
    state: WorkflowState, 
    user_id: str, 
    session_id: str, 
    updated_completed_intents_in_flow: str, 
    updated_session_primary_skill: str, 
    stm_df: pd.DataFrame
) -> pd.DataFrame():
    """
    Assembles a structured event log from the final state of a workflow run.
    All data points are extracted directly from the provided state object.

    Args:
        state (WorkflowState): The final state object from the LangGraph app.invoke() call.
        user_id (str): The unique ID of the user for this session.
        session_id (str): The unique ID for this conversation session.
        updated_completed_intents_in_flow (str): The list that serves as a memory of all the turn_intents.
        updated_session_primary_skill (str): The overarching "Core Skill" for the session.
        stm_df (DataFrame): The STM DataFrame to which the new event log will be appended.

    Returns:
        DataFrame: The updated STM DataFrame with the new event log row.
    """

    plan = state.get("execution_plan")

    if plan:
        event_status = "SUCCESS"
        error_details = None
    else:
        event_status = "FAILURE"
        error_details = "Planner failed to generate a valid execution plan."

    event_data = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(),
        "user_id": user_id,
        "session_id": session_id,
        "event_status": event_status,
        "error_details": error_details,
        "session_topic": state.get("session_topic", ""),
        "session_mood": state.get("session_mood", ""),
        "focus_emotion": state.get("focus_emotion", ""),
        "completed_intents_in_flow": str(updated_completed_intents_in_flow),
        "session_primary_skill": updated_session_primary_skill,
        "frequent_agents": str(state.get("frequent_agents", [])),
        "execution_plan": str(plan) if plan else "",
        "completed_steps": str(state.get("completed_steps", [])),
        "user_message": state.get("user_message", ""),
        "anaya_response": state.get("final_response", ""),
        "chat_history": state.get("chat_history", ""),
        "crisis_flag": state.get("crisis_flag", False),
        "crisis_level": state.get("crisis_level", "")
    }

    new_row = pd.DataFrame([event_data])
    stm_df = pd.concat([stm_df, new_row], ignore_index=True)

    return stm_df

def long_term_memory_event_log(
    final_state: WorkflowState,
    initial_journey_list: List[str],
    initial_toolkit: Dict[str, List[str]],
    initial_intentions: List[str],
    initial_threads: Dict[str, List[Dict]],
    conversation_history: List[str],
    user_id: str,
    session_id: str,
    session_started_at: datetime,
    ltm_df: pd.DataFrame()
) -> pd.DataFrame():
    """
    Analyzes a session, consolidates LTM, and logs the new LTM state to a DataFrame,
    using the exact logic provided.

    Args:
        final_state (WorkflowState): The final state object from the last turn of the graph.
        initial_journey_list (List[str]): The user's journey state *before* this session.
        initial_toolkit (Dict): The user's toolkit state *before* this session.
        initial_intentions (List[str]): The user's intentions state *before* this session.
        initial_threads (Dict): The user's memory threads state *before* this session.
        conversation_history (List[str]): The full transcript of the just-ended session.
        user_id (str): The unique ID of the user for this session.
        session_id (str): The unique ID for this conversation session.
        session_started_at (datetime): The timestamp when the session began.
        ltm_df (DataFrame): The Pandas DataFrame to which the new LTM log will be appended.

    Returns:
        DataFrame: The updated Pandas DataFrame with the new LTM log row.
    """

    session_ended_at = datetime.now()

    analysis_result = analyze_conversation_for_ltm(
        user_profile=final_state.get('user_profile'),
        previous_user_journey=initial_journey_list[-5:],
        previous_personal_toolkit=initial_toolkit,
        previous_guiding_intentions=initial_intentions,
        previous_memory_threads=initial_threads,
        chat_history="\n".join(conversation_history), # Convert list to string for agent
        focus_emotion=final_state.get('focus_emotion'),
        current_date=str(session_ended_at.date())
    )

    updated_ltm = consolidate_memory(
        analysis=analysis_result,
        previous_user_journey=initial_journey_list,
        previous_personal_toolkit=initial_toolkit,
        previous_guiding_intentions=initial_intentions,
        previous_memory_threads=initial_threads,
        focus_emotion=final_state.get('focus_emotion'),
        current_date=str(session_ended_at.date())
    )

    ltm_event_data = {
        "session_id": session_id,
        "user_id": user_id,
        "session_started_at": session_started_at,
        "session_ended_at": session_ended_at,
        "user_journey": str(updated_ltm['UserJourney']),
        "guiding_intentions": str(updated_ltm['GuidingIntentions']),
        "personal_toolkit": str(updated_ltm['PersonalToolkit']),
        "memory_threads": str(updated_ltm['MemoryThreads']),
        "somatic_focus": analysis_result.somatic_focus,
        "awareness_shift": analysis_result.awareness_shift,
        "support_preference": analysis_result.support_preference
    }

    new_ltm_row = pd.DataFrame([ltm_event_data])
    ltm_df = pd.concat([ltm_df, new_ltm_row], ignore_index=True)

    return ltm_df

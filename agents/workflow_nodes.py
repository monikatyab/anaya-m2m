import os

import json

import uuid

import ast

from datetime import datetime

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from typing import List, Dict, Optional, Literal, TypedDict, Union, Any

from agents.state import WorkflowState

from agents.planner_agent import generate_execution_plan

from agents.synthesis_agent import synthesize_response

from agents.crisis_agent import crisis_agent

from agents.dialogue_manager_agent import dialogue_manager_agent

from agents.factual_responder_agent import factual_responder_agent

from agents.reflection_agent import reflection_agent

from agents.wellness_assistant_agent import wellness_assistant_agent

from agents.short_term_memory_agent import short_term_memory_agent

def short_term_memory_node(state: WorkflowState) -> Dict:
    """
    Analyzes the user message to create STM for this turn.
    """
    print("--- Running Short-Term Memory Node ---")

    stm_results = short_term_memory_agent(state["chat_history"], state["user_message"])

    return {"session_topic": stm_results.session_topic, "session_mood": stm_results.session_mood, "focus_emotion": stm_results.focus_emotion,"crisis_flag": stm_results.crisis_flag, "crisis_level": stm_results.crisis_level}

def generate_execution_plan_node(state: WorkflowState) -> Dict:
    """
    This node runs the planner agent to create an execution plan.
    """
    print("--- Running Planner Node ---")
    user_message = state["user_message"]
    chat_history = state["chat_history"]

    result = generate_execution_plan(user_message, chat_history)

    return {"execution_plan": result.execution_plan, "user_query": result.question}

def execute_tools_node(state: WorkflowState) -> Dict:
    """Reads the plan and calls the correct agent functions dynamically."""
    print("--- Running Execute Tools Node ---")

    completed_steps = []

    core_skills = []

    intent = None

    for step in state["execution_plan"]:

        agent_name = step.agent_name
        agent_inputs = step.agent_inputs
        rationale = step.rationale
        personal_flag = step.personal_flag

        if agent_name == "crisis_agent":

            result_text = crisis_agent(state["user_message"]).response
            completed_steps.append(f"{agent_name}: {result_text}")


        elif agent_name == "factual_responder_agent":

            result_text = factual_responder_agent(agent_inputs)
            completed_steps.append(f"{agent_name}: {result_text}")

        elif agent_name == "dialogue_manager_agent":

            result_text = dialogue_manager_agent(chat_history=state["chat_history"], user_message=state["user_message"],
                                                task_description=agent_inputs, guiding_intentions=state["guiding_intentions"], user_journey=state["user_journey"],
                                                personal_flag=personal_flag, user_name=state["user_name"]).response

            completed_steps.append(f"{agent_name}: {result_text}")

        elif agent_name == "reflection_agent":

            result_text = reflection_agent(user_profile=state["user_profile"], user_name=state["user_name"],
                                            chat_history=state["chat_history"], user_message=state["user_message"],
                                            session_mood=state["session_mood"], task_description=agent_inputs).response

            completed_steps.append(f"{agent_name}: {result_text}")

        elif agent_name == "wellness_assistant_agent":

            result_text = wellness_assistant_agent(completed_intents_in_flow=state["completed_intents_in_flow"], session_primary_skill=state["session_primary_skill"],
                                                    user_profile=state["user_profile"], chat_history=state["chat_history"],
                                                    user_message=state["user_message"], session_topic=state["session_topic"],
                                                    session_mood=state["session_mood"], guiding_intentions=state["guiding_intentions"],
                                                    user_journey=state["user_journey"], focus_emotion=state["focus_emotion"],
                                                    memory_thread=state["memory_thread"], personal_toolkit=state["personal_toolkit"]).response

            skills = wellness_assistant_agent(completed_intents_in_flow=state["completed_intents_in_flow"], session_primary_skill=state["session_primary_skill"],
                                                user_profile=state["user_profile"], chat_history=state["chat_history"],
                                                user_message=state["user_message"], session_topic=state["session_topic"],
                                                session_mood=state["session_mood"], guiding_intentions=state["guiding_intentions"],
                                                user_journey=state["user_journey"], focus_emotion=state["focus_emotion"],
                                                memory_thread=state["memory_thread"], personal_toolkit=state["personal_toolkit"]).frequent_agents

            intent = wellness_assistant_agent(completed_intents_in_flow=state["completed_intents_in_flow"], session_primary_skill=state["session_primary_skill"],
                                                user_profile=state["user_profile"], chat_history=state["chat_history"],
                                                user_message=state["user_message"], session_topic=state["session_topic"],
                                                session_mood=state["session_mood"], guiding_intentions=state["guiding_intentions"],
                                                user_journey=state["user_journey"], focus_emotion=state["focus_emotion"],
                                                memory_thread=state["memory_thread"], personal_toolkit=state["personal_toolkit"]).inferred_turn_intent

            completed_steps.append(f"{agent_name}: {result_text}")
            core_skills.extend(skills)

        else:
            completed_steps.append(f"Error: Unknown agent '{agent_name}' requested.")

    return {"completed_steps": completed_steps, "frequent_agents": core_skills, "inferred_turn_intent": intent}

def synthesis_node(state: WorkflowState) -> Dict:
    """
    Runs the synthesis agent with the full context to create the final response.
    """
    print("--- Running Synthesis Node ---")

    response = synthesize_response(user_profile=state["user_profile"], chat_history=state["chat_history"],
                                         user_message=state["user_message"], completed_steps=state["completed_steps"])

    return {"final_response": response.final_response}

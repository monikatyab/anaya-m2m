from typing import List, Dict, Optional, Literal, TypedDict, Union, Any

class WorkflowState(TypedDict):
    """
    Represents the state of our workflow, holding all data passed between nodes.
    """
    # Inputs from the user
    user_message: str

    # LTM context loaded for the user
    user_profile: str
    user_name: str
    guiding_intentions: List[str]
    user_journey: str
    memory_thread: Dict[str, List[str]]
    personal_toolkit: Dict[str, List[str]]

    # STM generated during the workflow
    chat_history: str
    session_topic: str
    session_mood: str
    focus_emotion: str
    crisis_flag: bool
    crisis_level: str
    frequent_agents: List[str]
    completed_intents_in_flow: List[str]
    session_primary_skill: Optional[str]
    inferred_turn_intent: str

    # Planner output
    execution_plan: Optional[List[Dict]]
    user_query: str

    # Executor output
    completed_steps: Optional[List[str]]

    # Synthesizer output
    final_response: Optional[str]

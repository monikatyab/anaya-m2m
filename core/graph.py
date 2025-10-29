import os
import warnings

warnings.filterwarnings("ignore")

from typing import List, Dict, Optional, Literal, TypedDict, Union, Any
from langgraph.graph import StateGraph, END

from agents.state import WorkflowState
from agents.workflow_nodes import short_term_memory_node, generate_execution_plan_node, execute_tools_node, synthesis_node

workflow = StateGraph(WorkflowState)

workflow.add_node("short_term_memory", short_term_memory_node)
workflow.add_node("planner", generate_execution_plan_node)
workflow.add_node("execute_tools", execute_tools_node)
workflow.add_node("synthesis", synthesis_node)

workflow.set_entry_point("short_term_memory")

workflow.add_edge("short_term_memory", "planner")
workflow.add_edge("planner", "execute_tools")
workflow.add_edge("execute_tools", "synthesis")
workflow.add_edge("synthesis", END)

app = workflow.compile()
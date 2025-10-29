import os
import json
from typing import List, Dict, Optional, Literal, TypedDict, Union, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_fast = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

def factual_responder_agent(question):
    """
    Answers factual questions using the LLM's knowledge base.
    Web search capability removed - uses LLM knowledge only.
    """
    
    factual_prompt = """Answer the user's question directly and concisely based on your knowledge.

User Question: {question}

Provide a clear, factual answer. If you're unsure, say so rather than speculating."""

    generation = (
        ChatPromptTemplate.from_template(factual_prompt) 
        | llm_fast 
        | StrOutputParser()
    ).invoke({"question": question})

    return generation
"""
Core utility functions for Anaya AI
Handles user profile loading, validation, and data management
"""

import os
import pandas as pd
import ast
from typing import Dict, List, Optional
from pathlib import Path


def get_valid_user_ids() -> List[str]:
    """Get list of valid user IDs from User_Data.csv"""
    try:
        user_df = pd.read_csv("./data/User_Data.csv")
        return user_df['user_id'].tolist()
    except FileNotFoundError:
        print("Error: User_Data.csv not found in ./data/")
        return []
    except Exception as e:
        print(f"Error reading User_Data.csv: {e}")
        return []


def validate_user_id(user_id: str) -> bool:
    """Validate if user_id exists in User_Data.csv"""
    valid_users = get_valid_user_ids()
    return user_id in valid_users


def load_ltm_data() -> pd.DataFrame:
    """Load Long-Term Memory data from CSV"""
    ltm_path = "./data/LTM_Data.csv"
    try:
        if os.path.exists(ltm_path):
            return pd.read_csv(ltm_path)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'user_id', 'session_id', 'session_started_at', 'session_ended_at',
                'user_journey', 'guiding_intentions', 'memory_threads', 'personal_toolkit'
            ])
    except Exception as e:
        print(f"Error loading LTM data: {e}")
        return pd.DataFrame(columns=[
            'user_id', 'session_id', 'session_started_at', 'session_ended_at',
            'user_journey', 'guiding_intentions', 'memory_threads', 'personal_toolkit'
        ])


def load_stm_data() -> pd.DataFrame:
    """Load Short-Term Memory data from CSV"""
    stm_path = "./data/STM_Data.csv"
    try:
        if os.path.exists(stm_path):
            return pd.read_csv(stm_path)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'session_id', 'user_id', 'session_started_at', 'session_ended_at',
                'user_message', 'anaya_response', 'inferred_turn_intent', 'session_topic',
                'session_mood', 'focus_emotion', 'completed_intents_in_flow',
                'session_primary_skill', 'frequent_agents', 'tool_use', 'crisis_flag', 'crisis_level'
            ])
    except Exception as e:
        print(f"Error loading STM data: {e}")
        return pd.DataFrame(columns=[
            'session_id', 'user_id', 'session_started_at', 'session_ended_at',
            'user_message', 'anaya_response', 'inferred_turn_intent', 'session_topic',
            'session_mood', 'focus_emotion', 'completed_intents_in_flow',
            'session_primary_skill', 'frequent_agents', 'tool_use', 'crisis_flag', 'crisis_level'
        ])


def get_user_profile(user_id: str) -> Optional[Dict]:
    """
    Load user profile and LTM context for a given user_id
    
    Returns dict with:
        - user_profile: str
        - user_name: str
        - guiding_intentions: list
        - user_journey: list
        - memory_threads: dict
        - personal_toolkit: dict
    """
    try:
        # Load user data
        user_df = pd.read_csv("./data/User_Data.csv")
        user_row = user_df[user_df['user_id'] == user_id]
        
        if user_row.empty:
            return None
        
        # Load LTM data
        ltm_df = load_ltm_data()
        
        # Get latest LTM entry for this user
        user_ltm = ltm_df[ltm_df['user_id'] == user_id]
        
        # Initialize with defaults
        guiding_intentions = []
        user_journey = []
        memory_threads = {}
        personal_toolkit = {"user_found_helpful": [], "user_found_unhelpful": []}
        
        # Load from LTM if exists
        if not user_ltm.empty:
            latest_ltm = user_ltm.iloc[-1]
            
            try:
                guiding_intentions = ast.literal_eval(latest_ltm['guiding_intentions']) if pd.notna(latest_ltm['guiding_intentions']) else []
            except:
                guiding_intentions = []
            
            try:
                user_journey = ast.literal_eval(latest_ltm['user_journey']) if pd.notna(latest_ltm['user_journey']) else []
            except:
                user_journey = []
            
            try:
                memory_threads = ast.literal_eval(latest_ltm['memory_threads']) if pd.notna(latest_ltm['memory_threads']) else {}
            except:
                memory_threads = {}
            
            try:
                personal_toolkit = ast.literal_eval(latest_ltm['personal_toolkit']) if pd.notna(latest_ltm['personal_toolkit']) else {"user_found_helpful": [], "user_found_unhelpful": []}
            except:
                personal_toolkit = {"user_found_helpful": [], "user_found_unhelpful": []}
        
        return {
            'user_profile': user_row['user_profile'].values[0],
            'user_name': user_row['first_name'].values[0],
            'guiding_intentions': guiding_intentions,
            'user_journey': user_journey,
            'memory_threads': memory_threads,
            'personal_toolkit': personal_toolkit
        }
    
    except Exception as e:
        print(f"Error loading user profile: {e}")
        return None
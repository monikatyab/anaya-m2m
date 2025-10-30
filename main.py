"""
Main entry point for Anaya AI Wellness Assistant
"""

import os
import sys
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import app
from core.utils import validate_user_id, get_user_profile, get_valid_user_ids
from core.event_ingestion import short_term_memory_event_log, long_term_memory_event_log


def main():
    """Main execution loop"""
    
    print("=" * 50)
    print("--- Wellness Assistant - Anaya AI ---")
    print("=" * 50)
    
    # Check for data files
    data_dir = Path("./data")
    if not data_dir.exists():
        print("\n Error: './data' directory not found!")
        print("Please create it and add User_Data.csv, STM_Data.csv, LTM_Data.csv")
        return
    
    # Get user ID with validation
    valid_users = get_valid_user_ids()
    if not valid_users:
        print("\n Error: No users found in User_Data.csv")
        return
        
    print(f"\n Valid User IDs: {', '.join(valid_users)}")
    
    user_id = input("\nEnter User ID (or 'quit' to exit): ").strip()
    
    if user_id.lower() in ['quit', 'exit']:
        print("Goodbye!")
        return
    
    # Validate user ID
    if not validate_user_id(user_id):
        print(f"\n Error: User ID '{user_id}' not found in User_Data.csv")
        print(f"Valid users: {', '.join(valid_users)}")
        return
    
    # Load user profile and LTM context
    user_context = get_user_profile(user_id)
    
    if not user_context:
        print(f"\n Error: Could not load profile for user '{user_id}'")
        return
    
    # Initialize session
    session_id = uuid.uuid4()
    session_started_at = datetime.now()
    
    # Load CSV data
    try:
        ltm_df = pd.read_csv("./data/LTM_Data.csv")
        stm_df = pd.read_csv("./data/STM_Data.csv")
    except FileNotFoundError as e:
        print(f"\n Error loading data files: {e}")
        print("Ensure LTM_Data.csv and STM_Data.csv exist in ./data/")
        return
    
    # Initial LTM state
    initial_ltm_state = {
        "user_profile": user_context['user_profile'],
        "user_name": user_context['user_name'],
        "guiding_intentions": user_context['guiding_intentions'],
        "user_journey": user_context['user_journey'],
        "memory_thread": user_context['memory_threads'],
        "personal_toolkit": user_context['personal_toolkit'],
    }
    
    # Initial greeting
    initial_greeting = f"> Anaya: Hi {initial_ltm_state['user_name']}! How can I support you today?"
    conversation_history = [initial_greeting.replace("> Anaya: ", "Anaya: ")]
    
    print(f"\n{initial_greeting}")
    print("-" * 50)
    
    # Track state for LTM
    final_state = {}
    
    # Track STM fields
    updated_completed_intents = []
    updated_session_primary_skill = ""
    
    # Main conversation loop
    while True:
        user_message = input(f"> {initial_ltm_state['user_name']}: ").strip()
        
        if user_message.lower() in ["quit", "exit"]:
            print("\n" + "="*50)
            print("--- Saving Long-Term Memory ---")
            print("="*50)
            
            print(f" Session Summary:")
            print(f"  Messages: {len(conversation_history)}")
            print(f"  Duration: {(datetime.now() - session_started_at).seconds // 60} minutes")
            
            try:
                print("\n Analyzing conversation for LTM...")
                
                new_ltm_df = long_term_memory_event_log(
                    final_state=final_state,
                    initial_journey_list=user_context['user_journey'],
                    initial_toolkit=user_context['personal_toolkit'],
                    initial_intentions=user_context['guiding_intentions'],
                    initial_threads=user_context['memory_threads'],
                    conversation_history=conversation_history,
                    user_id=user_id,
                    session_id=session_id,
                    session_started_at=session_started_at,
                    ltm_df=ltm_df
)
                
                if len(new_ltm_df) > len(ltm_df):
                    latest_ltm = new_ltm_df.iloc[-1]
                    print("\n LTM Updated Successfully!")
                    print(f"  Journey: {str(latest_ltm['user_journey'])[:100]}...")
                else:
                    print("\nℹ No new LTM data generated")
                
                new_ltm_df.to_csv("./data/LTM_Data.csv", index=False)
                print("\n Saved to ./data/LTM_Data.csv")
                
            except Exception as e:
                print(f"\n LTM Error: {e}")
                import traceback
                traceback.print_exc()
            
            print("\nGoodbye! ")
            break
        
        if not user_message:
            print(" Please enter a message.")
            continue
        
        # Update STM tracking from previous turn
        if len(conversation_history) > 1 and not stm_df.empty:
            try:
                session_stm = stm_df[stm_df['session_id'] == str(session_id)]
                if not session_stm.empty:
                    import ast
                    updated_completed_intents = ast.literal_eval(
                        session_stm['completed_intents_in_flow'].values[-1]
                    )
                    frequent_agents = ast.literal_eval(
                        session_stm['frequent_agents'].values[0]
                    )
                    if frequent_agents:
                        updated_session_primary_skill = frequent_agents[0]
            except Exception as e:
                print(f" Warning: Could not load STM history: {e}")
        
        # Prepare turn state
        turn_state = {
            "user_id": user_id,
            "user_message": user_message,
            "chat_history": "\n".join(conversation_history),
            **initial_ltm_state,
            "session_topic": "",
            "session_mood": "",
            "focus_emotion": "",
            "crisis_flag": False,
            "crisis_level": None,
            "execution_plan": [],
            "completed_steps": [],
            "agents_called": [],
            "inferred_turn_intent": "",
            "completed_intents_in_flow": updated_completed_intents,
            "session_primary_skill": updated_session_primary_skill,
            "frequent_agents": [],
            "final_response": ""
        }
        
        try:
            # Run workflow
            final_state = app.invoke(turn_state)
            
            anaya_response = final_state.get("final_response", "I'm sorry, I encountered an issue.")
            
            print(f"\n> Anaya: {anaya_response}\n")
            print("-" * 50)
            
            # Update conversation history
            conversation_history.append(f"User: {user_message}")
            conversation_history.append(f"Anaya: {anaya_response}")
            
            # Update completed intents
            if final_state.get("inferred_turn_intent"):
                updated_completed_intents.append(final_state["inferred_turn_intent"])
            
            if final_state.get("frequent_agents"):
                frequent_agents = final_state["frequent_agents"]
                # If session_primary_skill is still empty, set it from first agent
                if not updated_session_primary_skill and len(frequent_agents) > 0:
                    updated_session_primary_skill = frequent_agents[0]
            
            # Log STM to CSV
            stm_df = short_term_memory_event_log(
                final_state, user_id, session_id, 
                updated_completed_intents, updated_session_primary_skill, stm_df,
                session_started_at
            )
            stm_df.to_csv("./data/STM_Data.csv", index=False)
            
            # Update LTM state for next turn
            initial_ltm_state.update({
                'user_journey': final_state.get('user_journey', initial_ltm_state['user_journey']),
                'personal_toolkit': final_state.get('personal_toolkit', initial_ltm_state['personal_toolkit']),
                'guiding_intentions': final_state.get('guiding_intentions', initial_ltm_state['guiding_intentions']),
                'memory_thread': final_state.get('memory_thread', initial_ltm_state['memory_thread'])
            })
        
        except Exception as e:
            print(f"\n Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Please try again.")


if __name__ == "__main__":
    main()

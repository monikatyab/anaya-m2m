"""
Anaya AI - Chat Interface
Real-time conversation with the wellness assistant
"""

import streamlit as st
import sys
from pathlib import Path
import uuid
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import app
from core.utils import get_valid_user_ids, validate_user_id, get_user_profile
from core.event_ingestion import short_term_memory_event_log

# Page config
st.set_page_config(
    page_title="Chat - Anaya AI",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    .crisis-alert {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .session-info {
        background-color: #FFF3E0;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stTextInput input {
        border-radius: 20px;
        border: 2px solid #4CAF50;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_initialized' not in st.session_state:
    st.session_state.chat_initialized = True
    st.session_state.selected_user = None
    st.session_state.conversation_history = []
    st.session_state.session_id = None
    st.session_state.session_started_at = None
    st.session_state.user_context = None
    st.session_state.ltm_state = None
    st.session_state.stm_df = None
    st.session_state.completed_intents = []
    st.session_state.session_primary_skill = ""

def initialize_session(user_id):
    """Initialize a new chat session"""
    st.session_state.session_id = uuid.uuid4()
    st.session_state.session_started_at = datetime.now()
    st.session_state.user_context = get_user_profile(user_id)
    st.session_state.conversation_history = []
    st.session_state.completed_intents = []
    st.session_state.session_primary_skill = ""
    
    # Load STM data
    try:
        st.session_state.stm_df = pd.read_csv("./data/STM_Data.csv")
    except FileNotFoundError:
        st.session_state.stm_df = pd.DataFrame()
    
    # Initial LTM state
    if st.session_state.user_context:
        st.session_state.ltm_state = {
            "user_profile": st.session_state.user_context['user_profile'],
            "user_name": st.session_state.user_context['user_name'],
            "guiding_intentions": st.session_state.user_context['guiding_intentions'],
            "user_journey": st.session_state.user_context['user_journey'],
            "memory_thread": st.session_state.user_context['memory_threads'],
            "personal_toolkit": st.session_state.user_context['personal_toolkit'],
        }
        
        # Add initial greeting to history
        greeting = f"Hi {st.session_state.ltm_state['user_name']}! How can I support you today?"
        st.session_state.conversation_history.append(("Anaya", greeting))

def process_message(user_message):
    """Process user message through the workflow"""
    
    if not st.session_state.user_context or not st.session_state.ltm_state:
        return "Please select a user first."
    
    # Prepare chat history string
    chat_history = "\n".join([
        f"{'User' if sender == 'User' else 'Anaya'}: {msg}" 
        for sender, msg in st.session_state.conversation_history
    ])
    
    # Prepare turn state
    turn_state = {
        "user_id": st.session_state.selected_user,
        "user_message": user_message,
        "chat_history": chat_history,
        **st.session_state.ltm_state,
        "session_topic": "",
        "session_mood": "",
        "focus_emotion": "",
        "crisis_flag": False,
        "crisis_level": None,
        "execution_plan": [],
        "completed_steps": [],
        "agents_called": [],
        "inferred_turn_intent": "",
        "completed_intents_in_flow": st.session_state.completed_intents,
        "session_primary_skill": st.session_state.session_primary_skill,
        "frequent_agents": [],
        "final_response": ""
    }
    
    try:
        # Run workflow
        final_state = app.invoke(turn_state)
        
        anaya_response = final_state.get("final_response", "I'm sorry, I encountered an issue.")
        
        # Update conversation history
        st.session_state.conversation_history.append(("User", user_message))
        st.session_state.conversation_history.append(("Anaya", anaya_response))
        
        # Update completed intents
        if final_state.get("inferred_turn_intent"):
            st.session_state.completed_intents.append(final_state["inferred_turn_intent"])
        
        # Update session primary skill
        if final_state.get("frequent_agents"):
            st.session_state.session_primary_skill = final_state["frequent_agents"][0] if final_state["frequent_agents"] else ""
        
        # Log to STM
        st.session_state.stm_df = short_term_memory_event_log(
            final_state,
            st.session_state.selected_user,
            str(st.session_state.session_id),
            st.session_state.completed_intents,
            st.session_state.session_primary_skill,
            st.session_state.stm_df
        )
        st.session_state.stm_df.to_csv("./data/STM_Data.csv", index=False)
        
        # Update LTM state
        st.session_state.ltm_state.update({
            'user_journey': final_state.get('user_journey', st.session_state.ltm_state['user_journey']),
            'personal_toolkit': final_state.get('personal_toolkit', st.session_state.ltm_state['personal_toolkit']),
            'guiding_intentions': final_state.get('guiding_intentions', st.session_state.ltm_state['guiding_intentions']),
            'memory_thread': final_state.get('memory_thread', st.session_state.ltm_state['memory_thread'])
        })
        
        return final_state
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        return None

def main():
    st.title("💬 Chat with Anaya")
    
    # Sidebar - User Selection
    with st.sidebar:
        st.markdown("## User Selection")
        
        valid_users = get_valid_user_ids()
        
        if not valid_users:
            st.error("No users found in User_Data.csv")
            return
        
        selected_user = st.selectbox(
            "Select User:",
            options=[""] + valid_users,
            key="user_selector"
        )
        
        if selected_user and selected_user != st.session_state.selected_user:
            st.session_state.selected_user = selected_user
            initialize_session(selected_user)
            st.rerun()
        
        if st.session_state.selected_user:
            st.success(f"✅ Chatting as: {st.session_state.selected_user}")
            
            if st.session_state.user_context:
                st.markdown("---")
                st.markdown(f"**Name:** {st.session_state.user_context['user_name']}")
                st.markdown(f"**Profile:** {st.session_state.user_context['user_profile'][:100]}...")
                
                st.markdown("---")
                st.markdown("### Session Info")
                if st.session_state.session_started_at:
                    duration = datetime.now() - st.session_state.session_started_at
                    st.markdown(f"**Duration:** {duration.seconds // 60} min")
                st.markdown(f"**Messages:** {len(st.session_state.conversation_history)}")
                
                st.markdown("---")
                if st.button("🔄 New Session", use_container_width=True):
                    initialize_session(st.session_state.selected_user)
                    st.rerun()
                
                if st.button("📥 Export Chat", use_container_width=True):
                    # Create export
                    export_data = {
                        "session_id": str(st.session_state.session_id),
                        "user": st.session_state.selected_user,
                        "started_at": str(st.session_state.session_started_at),
                        "conversation": st.session_state.conversation_history
                    }
                    st.download_button(
                        label="Download JSON",
                        data=str(export_data),
                        file_name=f"chat_{st.session_state.session_id}.json",
                        mime="application/json"
                    )
    
    # Main chat area
    if not st.session_state.selected_user:
        st.info("👈 Please select a user from the sidebar to start chatting")
        return
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for sender, message in st.session_state.conversation_history:
            if sender == "User":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>{message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Anaya:</strong><br>{message}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            key="user_input",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        with st.spinner("Anaya is thinking..."):
            final_state = process_message(user_input)
            
            if final_state:
                # Check for crisis flag
                if final_state.get("crisis_flag"):
                    st.markdown(f"""
                    <div class="crisis-alert">
                        <strong>⚠️ Crisis Alert Detected</strong><br>
                        Level: {final_state.get("crisis_level", "Unknown")}<br>
                        Emergency resources have been provided in the response.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show session info
                if final_state.get("session_mood") or final_state.get("focus_emotion"):
                    st.markdown(f"""
                    <div class="session-info">
                        <strong>Session Insights:</strong><br>
                        Emotion: {final_state.get("focus_emotion", "N/A")} | 
                        Mood: {final_state.get("session_mood", "N/A")[:50]}...
                    </div>
                    """, unsafe_allow_html=True)
        
        st.rerun()

if __name__ == "__main__":
    main()
"""
streamlit_app.py
Streamlit UI for Anaya - AI Wellness Assistant for Canadian Farmers
Updated to work with core/ structure
"""

import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import uuid
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables FIRST
load_dotenv()

# Import from core (updated paths)
from core.graph import app
from core.utils import (
    validate_user_id, 
    get_valid_user_ids, 
    get_user_profile
)
from core.event_ingestion import short_term_memory_event_log, long_term_memory_event_log

# Helper functions for data loading
def load_stm_data():
    """Load Short-Term Memory data"""
    try:
        return pd.read_csv("./data/STM_Data.csv")
    except FileNotFoundError:
        return pd.DataFrame()

def load_ltm_data():
    """Load Long-Term Memory data"""
    try:
        return pd.read_csv("./data/LTM_Data.csv")
    except FileNotFoundError:
        return pd.DataFrame()

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Anaya | Farm Wellness Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #2e7d32 0%, #388e3c 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #c8e6c9;
        margin: 10px 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* User message bubble */
    .user-message {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Assistant message bubble */
    .assistant-message {
        background: #f5f5f5;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 100%);
    }
    
    /* Input box */
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #4caf50;
        padding: 12px 20px;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 25px;
        background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #388e3c 0%, #4caf50 100%);
        box-shadow: 0 4px 12px rgba(76,175,80,0.4);
    }
    
    /* Crisis alert */
    .crisis-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE INITIALIZATION ==========
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_context' not in st.session_state:
        st.session_state.user_context = None
    if 'initial_user_context' not in st.session_state:
        st.session_state.initial_user_context = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'session_started_at' not in st.session_state:
        st.session_state.session_started_at = datetime.now()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    if 'final_state' not in st.session_state:
        st.session_state.final_state = {}
    if 'completed_intents' not in st.session_state:
        st.session_state.completed_intents = []
    if 'session_primary_skill' not in st.session_state:
        st.session_state.session_primary_skill = ""
    if 'processing' not in st.session_state:
        st.session_state.processing = False

# ========== LTM SAVE HELPER ==========
def save_ltm_on_session_end():
    """Save LTM when session ends (logout or new conversation)"""
    if st.session_state.message_count > 0:
        try:
            ltm_df = load_ltm_data()
            
            # Final state (what was updated during conversation)
            final_state = st.session_state.final_state
            
            # Initial state (from User_Data.csv when user logged in)
            initial_context = st.session_state.initial_user_context or {}
            
            new_ltm_df = long_term_memory_event_log(
                final_state=final_state,
                initial_journey_list=initial_context.get('user_journey', []),
                initial_toolkit=initial_context.get('personal_toolkit', {}),
                initial_intentions=initial_context.get('guiding_intentions', []),
                initial_threads=initial_context.get('memory_threads', {}),
                conversation_history=st.session_state.conversation_history,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                session_started_at=st.session_state.session_started_at,
                ltm_df=ltm_df
            )
            
            new_ltm_df.to_csv("./data/LTM_Data.csv", index=False)
            print(f" LTM saved for session {st.session_state.session_id}")
            return True
        except Exception as e:
            print(f" Error saving LTM: {e}")
            import traceback
            traceback.print_exc()
            return False
    return False

# ========== LOGIN PAGE ==========
def show_login_page():
    """Display login page with CSV user selection"""
    st.markdown("""
        <div class="main-header">
            <h1>🌾 Welcome to Anaya</h1>
            <p>Your AI Wellness Companion for Farm Life</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 👤 Enter Your Details")
    
    # Get valid users from CSV
    valid_users = get_valid_user_ids()
    
    if not valid_users:
        st.error(" No users found in User_Data.csv. Please add users to ./data/User_Data.csv to continue.")
        st.stop()
    
    # User selection
    selected_user = st.selectbox(
        "Select your User ID:",
        options=[""] + valid_users,
        format_func=lambda x: "-- Select a User --" if x == "" else x
    )
    
    # Show user info if selected
    if selected_user:
        try:
            user_df = pd.read_csv("./data/User_Data.csv")
            user_info = user_df[user_df['user_id'] == selected_user].iloc[0]
            
            st.info(f"""
            **Name:** {user_info['first_name']} {user_info['last_name']}  
            **Location:** {user_info['city']}, {user_info['province']}  
            **Profile:** {user_info['user_profile']}
            """)
        except Exception as e:
            st.error(f"Error loading user info: {e}")
    
    # Login button
    if st.button("🌾 Start Session", disabled=not selected_user):
        if selected_user and validate_user_id(selected_user):
            # Load user profile
            user_context = get_user_profile(selected_user)
            
            if user_context:
                st.session_state.user_id = selected_user
                st.session_state.user_context = user_context
                st.session_state.initial_user_context = user_context.copy()
                st.session_state.logged_in = True
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.session_started_at = datetime.now()
                st.rerun()
            else:
                st.error(" Could not load user profile. Please try again.")
        else:
            st.error(" Invalid user ID")
    
    # Information section
    st.markdown("---")
    st.markdown("### 🌾 About Anaya")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **What Anaya Offers:**
        - 💚 Emotional support and validation
        - 🧠 Practical coping strategies  
        - 🌾 Farm-specific wellness guidance
        - 🗣️ Compassionate, judgment-free space
        """)
    
    with col2:
        st.markdown("""
        **How It Works:**
        1. Select your user profile
        2. Share what's on your mind
        3. Get personalized support
        4. Build your wellness journey
        """)
    
    st.markdown("---")
    
    # Disclaimer
    with st.expander("⚠️ Important Notice"):
        st.markdown("""
        Anaya provides **emotional support and wellness guidance** 
        but is **NOT a substitute** for professional medical care, 
        therapy, or crisis intervention.
        
        **Always consult qualified healthcare professionals** 
        for diagnosis, treatment, or emergencies.
        
        **Crisis Resources (Canada):**
        - Crisis Text Line: Text HOME to 741741
        - National Suicide Prevention: Call or text 988
        - Talk Suicide Canada: 1-833-456-4566
        - Emergency: Call 911
        """)
    
    st.markdown("---")
    st.markdown("_Built with ❤️ for Canadian Farmers_")
    st.markdown("_Version 1.0_")

# ========== SIDEBAR ==========
def render_sidebar():
    """Render sidebar with session info and controls"""
    with st.sidebar:
        st.markdown("## 🌾 Anaya AI")
        st.markdown("---")
        
        # User info
        if st.session_state.user_context:
            st.success(f"**Logged in as:**")
            st.markdown(f"👤 **{st.session_state.user_context['user_name']}**")
            st.markdown(f"🆔 `{st.session_state.user_id}`")
        
        st.markdown("---")
        
        # Session info
        st.markdown("### 📊 Session Info")
        
        if st.session_state.session_started_at:
            duration = datetime.now() - st.session_state.session_started_at
            st.markdown(f"**Duration:** {duration.seconds // 60} min")
        
        st.markdown(f"**Messages:** {st.session_state.message_count}")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        st.markdown("---")
        
        # Session controls
        st.markdown("### ⚙️ Session Controls")
        
        if st.button("🔄 New Conversation", use_container_width=True):
            # Save LTM before clearing
            save_ltm_on_session_end()
            
            # Reset conversation state
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.conversation_started = False
            st.session_state.message_count = 0
            st.session_state.agent_logs = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_started_at = datetime.now()
            st.session_state.completed_intents = []
            st.session_state.session_primary_skill = ""
            
            # Reload user context
            st.session_state.user_context = get_user_profile(st.session_state.user_id)
            st.session_state.initial_user_context = st.session_state.user_context.copy()
            
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True):
            # Save LTM before logout
            save_ltm_on_session_end()
            
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.rerun()
        
        st.markdown("---")
        
        # User profile summary
        st.markdown("### 📋 Your Profile")
        
        if st.session_state.user_context:
            with st.expander("View Details"):
                st.markdown(f"**Profile:**")
                st.markdown(st.session_state.user_context['user_profile'])
                
                if st.session_state.user_context.get('user_journey'):
                    st.markdown(f"**Journey Entries:** {len(st.session_state.user_context['user_journey'])}")
                
                if st.session_state.user_context.get('guiding_intentions'):
                    intentions = [i for i in st.session_state.user_context['guiding_intentions'] if i]
                    st.markdown(f"**Guiding Intentions:** {len(intentions)}")
                
                toolkit = st.session_state.user_context.get('personal_toolkit', {})
                helpful = len(toolkit.get('user_found_helpful', []))
                st.markdown(f"**Helpful Tools:** {helpful}")
        
        st.markdown("---")
        
        # Disclaimer
        with st.expander("⚠️ Important Notice"):
            st.markdown("""
            Anaya provides **emotional support and wellness guidance** 
            but is **NOT a substitute** for professional medical care, 
            therapy, or crisis intervention.
            
            **Always consult qualified healthcare professionals** 
            for diagnosis, treatment, or emergencies.
            """)
        
        st.markdown("---")
        st.markdown("_Built for Canadian Farmers_")
        st.markdown("_Version 1.0_")

# ========== CHAT INTERFACE ==========
def render_chat_interface():
    """Main chat interface"""
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(" **Error:** GOOGLE_API_KEY not found. Please add it to your .env file.")
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Chat with Anaya</h1>
        <p>Share what's on your mind - I'm here to listen and support you</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Welcome message (only show once)
        if not st.session_state.conversation_started:
            with st.chat_message("assistant", avatar="🌱"):
                st.markdown(f"""
                **Hi {st.session_state.user_context['user_name']}! I'm Anaya, your wellness companion.**
                
                I'm here to provide:
                - 💚 Emotional support and validation
                - 🧠 Practical coping strategies
                - 🌾 Farm-specific wellness guidance
                - 🗣️ A compassionate, judgment-free space
                
                **How are you feeling today?**
                """)
            st.session_state.conversation_started = True
        
        # Display conversation history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"], avatar="🌱" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
                if "timestamp" in msg:
                    st.caption(msg["timestamp"])
                
                # Show agent logs for debugging (optional)
                if msg["role"] == "assistant" and i < len(st.session_state.agent_logs):
                    log = st.session_state.agent_logs[i]
                    
                    # Show crisis alert if detected
                    if log.get('crisis_flag'):
                        st.markdown(f"""
                        <div class="crisis-alert">
                            <strong> Crisis Alert Detected</strong><br>
                            Level: {log.get('crisis_level', 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here...", key="user_input"):
        # Add user message to display
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        # Update conversation history immediately
        st.session_state.conversation_history.append(f"User: {prompt}")
        
        # Set processing flag
        st.session_state.processing = True
        
        # Rerun to show user message before processing
        st.rerun()
    
    # Process the last message if it's from user and hasn't been responded to
    if (st.session_state.processing and 
        st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user"):
        
        prompt = st.session_state.messages[-1]["content"]
        
        # Reset processing flag immediately to prevent reprocessing
        st.session_state.processing = False
        
        # Process with agent
        with st.spinner("🌾 Anaya is thinking..."):
            try:
                # Prepare state for workflow
                turn_state = {
                    "user_id": st.session_state.user_id,
                    "user_message": prompt,
                    "chat_history": "\n".join(st.session_state.conversation_history),
                    "user_name": st.session_state.user_context['user_name'],
                    "user_profile": st.session_state.user_context['user_profile'],
                    "guiding_intentions": st.session_state.user_context['guiding_intentions'],
                    "user_journey": st.session_state.user_context['user_journey'],
                    "memory_thread": st.session_state.user_context['memory_threads'],
                    "personal_toolkit": st.session_state.user_context['personal_toolkit'],
                    "session_topic": "",
                    "session_mood": "",
                    "focus_emotion": "",
                    "crisis_flag": False,
                    "crisis_level": None,
                    "execution_plan": [],
                    "completed_steps": [],
                    "inferred_turn_intent": "",
                    "completed_intents_in_flow": st.session_state.completed_intents,
                    "session_primary_skill": st.session_state.session_primary_skill,
                    "frequent_agents": [],
                    "final_response": ""
                }
                
                # Run workflow
                result = app.invoke(turn_state)
                st.session_state.final_state = result  # Store for LTM logging later
                
                response = result.get("final_response", "I'm sorry, I encountered an issue.")
                
                # Add agent log
                agent_log = {
                    "pipeline": "Success",
                    "focus_emotion": result.get("focus_emotion", "N/A"),
                    "session_topic": result.get("session_topic", "N/A"),
                    "session_mood": result.get("session_mood", "N/A"),
                    "crisis_flag": result.get("crisis_flag", False),
                    "crisis_level": result.get("crisis_level", None)
                }
                st.session_state.agent_logs.append(agent_log)
                
                # Display response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                
                st.session_state.message_count += 1
                
                # Update conversation history with response only
                st.session_state.conversation_history.append(f"Anaya: {response}")
                
                # Update completed intents
                if result.get("inferred_turn_intent"):
                    st.session_state.completed_intents.append(result["inferred_turn_intent"])
                
                # Update session primary skill
                if result.get("frequent_agents"):
                    st.session_state.session_primary_skill = result["frequent_agents"][0] if result["frequent_agents"] else ""
                
                # Save to STM
                stm_df = load_stm_data()
                new_stm = short_term_memory_event_log(
                    result,
                    st.session_state.user_id,
                    st.session_state.session_id,
                    st.session_state.completed_intents,
                    st.session_state.session_primary_skill,
                    stm_df
                )
                new_stm.to_csv("./data/STM_Data.csv", index=False)
                
                # Update user context with new LTM state
                st.session_state.user_context['user_journey'] = result.get(
                    'user_journey', 
                    st.session_state.user_context['user_journey']
                )
                st.session_state.user_context['personal_toolkit'] = result.get(
                    'personal_toolkit',
                    st.session_state.user_context['personal_toolkit']
                )
                st.session_state.user_context['guiding_intentions'] = result.get(
                    'guiding_intentions',
                    st.session_state.user_context['guiding_intentions']
                )
                st.session_state.user_context['memory_threads'] = result.get(
                    'memory_thread',
                    st.session_state.user_context['memory_threads']
                )
                
                # Rerun to display new messages
                st.rerun()
                
            except Exception as e:
                st.error(f" Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error. Please try again.",
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                
                # Add empty log for failed responses
                st.session_state.agent_logs.append({
                    "pipeline": "Error occurred",
                    "focus_emotion": "N/A",
                    "session_topic": "N/A",
                    "session_mood": "N/A",
                    "crisis_flag": False,
                    "crisis_level": None
                })
                st.rerun()

# ========== MAIN APP ==========
def main():
    """Main application entry point"""
    init_session_state()
    
    if not st.session_state.logged_in:
        show_login_page()
    else:
        render_sidebar()
        render_chat_interface()

if __name__ == "__main__":
    main()
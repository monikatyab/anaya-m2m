"""
Anaya AI Wellness Assistant - Streamlit App
Main entry point for the web interface
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.utils import get_valid_user_ids, validate_user_id

# Page configuration
st.set_page_config(
    page_title="Anaya AI Wellness Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #555;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.selected_user = None
    st.session_state.conversation_history = []
    st.session_state.session_active = False

def main():
    """Main landing page"""
    
    # Header
    st.markdown('<div class="main-header">🌾 Anaya AI Wellness Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Compassionate Guide to Mental Wellness</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to **Anaya AI**, a sophisticated wellness companion designed to support your mental health journey 
    through empathetic conversation, personalized guidance, and adaptive memory systems.
    """)
    
    # Check for data files
    data_dir = Path("./data")
    if not data_dir.exists():
        st.error("⚠️ Data directory not found! Please create './data' folder with required CSV files.")
        st.info("""
        Required files:
        - `User_Data.csv`
        - `LTM_Data.csv` 
        - `STM_Data.csv`
        """)
        return
    
    # Get valid users
    valid_users = get_valid_user_ids()
    
    if not valid_users:
        st.error("⚠️ No users found in User_Data.csv")
        st.info("Please add user data to ./data/User_Data.csv")
        return
    
    # Display statistics
    st.markdown("### 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(valid_users)}</div>
            <div class="stat-label">Active Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Count total sessions from STM
        try:
            import pandas as pd
            stm_df = pd.read_csv("./data/STM_Data.csv")
            session_count = stm_df['session_id'].nunique() if not stm_df.empty else 0
        except:
            session_count = 0
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{session_count}</div>
            <div class="stat-label">Total Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Count total conversations
        try:
            message_count = len(stm_df) if not stm_df.empty else 0
        except:
            message_count = 0
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{message_count}</div>
            <div class="stat-label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">7</div>
            <div class="stat-label">AI Agents</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🧠 Multi-Agent Architecture</div>
            <div class="feature-desc">
                Specialized AI agents work together to provide comprehensive support:
                <ul>
                    <li><strong>Planner Agent:</strong> Orchestrates conversation flow</li>
                    <li><strong>Wellness Assistant:</strong> Provides therapeutic guidance</li>
                    <li><strong>Reflection Agent:</strong> Offers empathetic validation</li>
                    <li><strong>Crisis Agent:</strong> Handles urgent situations</li>
                    <li><strong>Dialogue Manager:</strong> Ensures smooth transitions</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🎯 Personalized Support</div>
            <div class="feature-desc">
                Every interaction is tailored to your unique journey:
                <ul>
                    <li>Remembers your progress and preferences</li>
                    <li>Adapts to your emotional state</li>
                    <li>Builds on previous conversations</li>
                    <li>Tracks helpful and unhelpful strategies</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">💾 Dual Memory System</div>
            <div class="feature-desc">
                <strong>Short-Term Memory (STM):</strong>
                <ul>
                    <li>Tracks session mood and topics</li>
                    <li>Monitors emotional intensity</li>
                    <li>Records agent interactions</li>
                </ul>
                <strong>Long-Term Memory (LTM):</strong>
                <ul>
                    <li>Chronicles your wellness journey</li>
                    <li>Stores guiding intentions</li>
                    <li>Maintains personal toolkit</li>
                    <li>Builds memory threads by emotion</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🛡️ Safety First</div>
            <div class="feature-desc">
                <ul>
                    <li>Real-time crisis detection</li>
                    <li>Immediate access to emergency resources</li>
                    <li>Confidential and secure</li>
                    <li>Evidence-based therapeutic approaches</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.markdown("### 🔄 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **1️⃣ Analyze**
        
        The system analyzes your message for emotional content, intent, and context.
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Plan**
        
        The Planner Agent creates a thoughtful response strategy using multiple agents.
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Respond**
        
        Specialized agents collaborate to provide personalized, empathetic support.
        """)
    
    with col4:
        st.markdown("""
        **4️⃣ Remember**
        
        Insights are stored in memory systems for continuity and growth tracking.
        """)
    
    st.markdown("---")
    
    # Getting started
    st.markdown("### 🚀 Get Started")
    
    st.info("""
    **Ready to begin?** Choose an option from the sidebar:
    
    - 💬 **Chat** - Start a conversation with Anaya
    - 👤 **Profile** - View your wellness profile and journey
    - 📊 **Analytics** - Explore your session history and insights
    - ⚙️ **Settings** - Configure preferences and manage data
    """)
    
    # Quick start button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💬 Start Chatting", use_container_width=True):
            st.switch_page("pages/1_💬_Chat.py")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Built with ❤️ using LangGraph, LangChain, and Google Gemini</p>
        <p><small>© 2025 Anaya AI Wellness Assistant | Your privacy and wellbeing are our priority</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🌾 Anaya AI")
        st.markdown("---")
        
        st.markdown("### 👥 Available Users")
        for user_id in valid_users:
            st.markdown(f"- `{user_id}`")
        
        st.markdown("---")
        
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [Documentation](#)
        - [Privacy Policy](#)
        - [Terms of Service](#)
        - [Contact Support](#)
        """)
        
        st.markdown("---")
        
        # System status
        st.markdown("### ⚡ System Status")
        st.success("✅ All systems operational")
        
        # Version info
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Model:** Gemini 2.5 Pro")

if __name__ == "__main__":
    main()

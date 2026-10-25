import os
import streamlit as st
from dotenv import load_dotenv
import utils

# Load environment variables from .env
load_dotenv()

st.set_page_config(page_title="MCP POC", page_icon="🤖", layout="wide")

st.title("Model Context Protocol(MCP) - Learning Path Generator")

# Initialize session state for progress
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'last_section' not in st.session_state:
    st.session_state.last_section = ""
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

# Sidebar for API configuration
st.sidebar.header("Configuration")

# API Key input (pre-fills from .env if available)
default_google_key = os.getenv("GEMINI_API_KEY", "")
google_api_key = st.sidebar.text_input("Google API Key", value=default_google_key, type="password")

gemini_model = st.sidebar.selectbox(
    "Gemini model",
    [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
    ],
    index=0,
    key="gemini_model",
    help="Free-tier quotas are per model. If one model returns 429 RESOURCE_EXHAUSTED, pick another.",
)

st.sidebar.subheader("Composio accounts")
st.sidebar.caption("YouTube and Google Drive must be connected before generating a path.")

if "connect_links" not in st.session_state:
    st.session_state.connect_links = {}

try:
    connected_toolkits = utils.get_connected_toolkits()
except Exception as e:
    connected_toolkits = {}
    st.sidebar.error(f"Could not load Composio accounts: {e}")

for toolkit in utils.REQUIRED_TOOLKITS:
    label = utils.TOOLKIT_LABELS[toolkit]
    status = connected_toolkits.get(toolkit, "NOT CONNECTED")
    if status == "ACTIVE":
        st.sidebar.success(f"{label}: connected")
        continue

    st.sidebar.warning(f"{label}: {status.replace('_', ' ').lower()}")
    if st.sidebar.button(f"Connect {label}", key=f"connect_{toolkit}"):
        try:
            st.session_state.connect_links[toolkit] = utils.create_connect_link(toolkit)
        except Exception as e:
            st.sidebar.error(f"Could not start {label} connection: {e}")

    connect_url = st.session_state.connect_links.get(toolkit)
    if connect_url:
        st.sidebar.markdown(f"[Open {label} authorization]({connect_url})")
        st.sidebar.caption("Finish Google sign-in, then refresh this page.")

# Quick guide before goal input
st.info("""
**Quick Guide:**
1. Connect YouTube and Google Drive in the sidebar (Composio OAuth). This is required — tools are not connected automatically.
2. Enter your Google API key (if not already loaded).
3. Enter a clear learning goal, for example:
    - "I want to learn python basics in 3 days"
    - "I want to learn data science basics in 10 days"
""")

# Main content area
st.header("Enter Your Goal")
user_goal = st.text_input("Enter your learning goal:",
                          help="Describe what you want to learn, and we'll generate a structured path using YouTube content and your selected tools.")

# Progress area
progress_container = st.container()
progress_bar = st.empty()

def update_progress(message: str):
    """Update progress in the Streamlit UI"""
    st.session_state.current_step = message
    
    # Determine section and update progress based on the new Composio messages
    if "Setting up agent with Composio" in message:
        section = "Setup"
        st.session_state.progress = 0.1
    elif "Getting available tools from Composio" in message:
        section = "Integration"
        st.session_state.progress = 0.2
    elif "Creating AI agent" in message:
        section = "Setup"
        st.session_state.progress = 0.3
    elif "Generating your learning path" in message:
        section = "Generation"
        st.session_state.progress = 0.5
    elif "Learning path generation complete" in message:
        section = "Complete"
        st.session_state.progress = 1.0
        st.session_state.is_generating = False
    else:
        section = st.session_state.last_section or "Progress"
    
    st.session_state.last_section = section
    
    # Show progress bar
    progress_bar.progress(st.session_state.progress)
    
    # Update progress container with current status
    with progress_container:
        # Show section header if it changed
        if section != st.session_state.last_section and section != "Complete":
            st.write(f"**{section}**")
        
        # Show message with tick for completed steps
        if message == "Learning path generation complete!":
            st.success("All steps completed! 🎉")
        else:
            prefix = "✓" if st.session_state.progress >= 0.5 else "→"
            st.write(f"{prefix} {message}")

# Generate Learning Path button
if st.button("Generate Learning Path", type="primary", disabled=st.session_state.is_generating):
    if not google_api_key:
        st.error("Please enter your Google API key in the sidebar.")
    elif not user_goal:
        st.warning("Please enter your learning goal.")
    else:
        try:
            # Set generating flag
            st.session_state.is_generating = True
            
            # Reset progress
            st.session_state.current_step = ""
            st.session_state.progress = 0
            st.session_state.last_section = ""
            
            # Run the agent using the updated function signature
            result = utils.run_agent_sync(
                google_api_key=google_api_key,
                user_goal=user_goal,
                model=gemini_model,
                progress_callback=update_progress
            )
            
            # Display results
            st.header("Your Learning Path")
            if result and "messages" in result:
                # The final response is usually the last message in the list
                final_message = result["messages"][-1]
                st.markdown(f"📚 {final_message.content}")

            else:
                st.error("No results were generated. Please try again.")
                st.session_state.is_generating = False
        except Exception as e:
            message = str(e)
            st.error(f"An error occurred: {message}")
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                st.warning(
                    "This Gemini model's free daily quota is used up. "
                    "Choose a different model in the sidebar and generate again, "
                    "or wait until the quota resets (about 19 hours for gemini-3.8-flash)."
                )
            else:
                st.error("Please check your API keys and try again.")
            st.session_state.is_generating = False
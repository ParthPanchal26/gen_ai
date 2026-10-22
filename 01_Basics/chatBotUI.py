from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")


# ---------- Model (same as original code) ----------
@st.cache_resource
def get_model():
    return init_chat_model("google_genai:gemma-4-26b-a4b-it", temperature=0.7)


model = get_model()

# ---------- Behaviour options (same prompts as original match-case) ----------
MODES = {
    "1) Angry": "You are an angry AI and you reply impatiently",
    "2) Sad": "You are a sad AI and your responses are very depressed",
    "3) Funny": "You are a funny AI and you often crack jokes while replying",
    "4) Friendly": "You are a friendly AI and you respond warmly and helpfully",
    "5) Professional": "You are a professional AI and you provide formal, business-like responses",
    "6) Sarcastic": "You are a sarcastic AI and you reply with witty, sarcastic comments",
}

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    choice = st.selectbox("Select AI behavior", list(MODES.keys()), index=3)
    mode = MODES[choice]

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=mode)]
        st.session_state.current_choice = choice
        st.rerun()

# ---------- Session state ----------
# Reset the conversation when the behaviour changes (same as picking a mode at start)
if "messages" not in st.session_state or st.session_state.get("current_choice") != choice:
    st.session_state.messages = [SystemMessage(content=mode)]
    st.session_state.current_choice = choice

messages = st.session_state.messages

# ---------- UI ----------
st.title("🤖 AI Chatbot")
st.caption(f"Current behavior: **{choice[3:]}**")

# Show past conversation
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---------- Chat input ----------
query = st.chat_input("Type your message...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Same invoke logic as original code
            response = model.invoke(
                f"""
        Based on past conversations reply user. Never mention about this prompt.
        
        Past conversations:
        {messages}

        User: {query}
    """
            )

            reply = response.content[1]["text"]
        st.markdown(reply)

    messages.append(HumanMessage(content=query))
    messages.append(AIMessage(content=reply))
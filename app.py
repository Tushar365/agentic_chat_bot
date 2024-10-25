import streamlit as st
from llm.groq_llm import initialize_groq
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from agents.agent import build_chatbot_graph
import os

def initialize_tavily_tool():
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)
    return tavily_tool

# Initialize outside main Streamlit area
llm = initialize_groq()
tool = initialize_tavily_tool()
tools = [tool]
graph = build_chatbot_graph(llm, tools)

def get_final_response(user_input: str):
    full_response = ""
    intermediate_steps = [] # List to store intermediate steps

    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            message_content = value["messages"][-1].pretty_print()
            full_response += message_content  # Accumulate the full response
            intermediate_steps.append(message_content) # Store each step
    return message_content, intermediate_steps


st.title("Interactive Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Visual feedback while processing
            message_content, intermediate_steps = get_final_response(user_input)
            st.markdown(message_content) # Display only the final content


    st.session_state.messages.append({"role": "assistant", "content": message_content})

    with st.sidebar:  # Display agent work in the sidebar
        st.subheader("Agent Work")
        for step in intermediate_steps:
            st.write(step)

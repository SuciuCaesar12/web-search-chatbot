import streamlit as st
import asyncio
import json
import uuid

from langfuse.callback import CallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from agents import chat_agent


@st.cache_resource
def create_chatbot_instance():
    return chat_agent

chatbot = create_chatbot_instance()


@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()


async def prompt_ai(messages):
    config = {
        "configurable": {"thread_id": thread_id},
    }

    async for event in chatbot.astream_events(
            {"messages": messages}, config, version="v2", include_names=['chatbot']
        ):
            if event["event"] == "on_chat_model_stream":
                yield event["data"]["chunk"].content            


async def main():
    st.title("Web Search ChatBot")


    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hello! How can I help you?")
        ]    


    for message in st.session_state.messages:
        message_json = json.loads(message.model_dump_json())
        message_type = message_json["type"]
        
        if message_type in ["human", "ai"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])        


    if prompt := st.chat_input("What would you like to do today?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            async for chunk in prompt_ai(st.session_state.messages):
                response_content += chunk
                message_placeholder.markdown(response_content)
        
        st.session_state.messages.append(AIMessage(content=response_content))


if __name__ == "__main__":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

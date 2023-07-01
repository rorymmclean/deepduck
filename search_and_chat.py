from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st

st.set_page_config(page_title="Deep Searching DuckDuckGo", page_icon="🦜", layout="wide")
st.title("🦜 Deep Searching DuckDuckGo")

openai_api_key = st.secrets["openai_api_key"]
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
    search_agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=True,
    )

    # mytemplate=f"""You are a researcher and are helping to answer the question below.
    # Provide a detailed answer that addresses every aspect of the question.

    # Question:
    # {prompt}
    # """

    with st.spinner("Processing..."):
        response = search_agent(prompt) #mytemplate)

    st.session_state.messages.append({"role": "assistant", "content": response['output']})    
    st.chat_message('assistant').write(response['output'])
    with st.expander('Details', expanded=False):
        st.write(response["intermediate_steps"])

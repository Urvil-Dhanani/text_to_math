import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Streamlit App
st.set_page_config(page_title="Text to Math problem solver",
                   page_icon="🧮")
st.title(body="🧮 Text to Math problem solver")

with st.sidebar:
    groq_API=st.text_input(label="Enter Groq API",
                           type="password",
                           value="")

if not groq_API:
    st.info(body="Please enter Groq API")
    st.stop()

llm=ChatGroq(model="Gemma2-9b-It",
             groq_api_key=groq_API)

# Initializing the Tools 

# Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
math_tool=Tool(name="Calculator",               
               func=math_chain.run,
               description="A tools for answering math related questions. Only input mathematical expression need to be provided")

# wikipedia tool
wiki_wrapper=WikipediaAPIWrapper()
wiki_tool=Tool(name="Wikipedia",
               func=wiki_wrapper.run,
               description="A tool for searching the Internet to find required information")

# We use reasoning tool 
# It basically depends on llm model 
# Basically we are using llm model for Reasoning 
# Reasoning: answering logic-based and reasoning questions

# First we create prompts
prompt="""
You are an agent tasked for solving user's mathemtical question. Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below
Question:{question}
Answer:"""

prompt_template=PromptTemplate(input_variables=["question"],
                               template=prompt)

# Lets create llm chain first to initialize reasoning agent
chain=LLMChain(llm=llm,
               prompt=prompt_template)

reasoning_tool=Tool(name="Reasoning",
                    func=chain.run,
                    description="A tool for answering logic-based and reasoning questions")

# Now lets initialize the AgentExecutor -- that will combine tools--llm
agent_assistant=initialize_agent(tools=[math_tool, wiki_tool, reasoning_tool],
                                 llm=llm,
                                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=False,
                                 handle_parsing_error=True)

# lets create chatbot
if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"Assistant", "content":"Hi, I'm a Math chatbot who can answer all your maths questions"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question=st.text_area(label="Please ask any Math question")

if st.button(label="Find the answer"):
    if question:
        with st.spinner(text="Loading"):
            st.session_state.messages.append({"role":"User", "content":question})
            st.chat_message("User").write(question)

            st_cb=StreamlitCallbackHandler(parent_container=st.container(),
                                           expand_new_thoughts=False)
            response=agent_assistant.run(st.session_state, callbacks=[st_cb])

            st.session_state.messages.append({"role":"User", "content":response})
            st.write("ANSWER:")
            st.success(body=response)

    else:
        st.warning(body="Please enter the question")




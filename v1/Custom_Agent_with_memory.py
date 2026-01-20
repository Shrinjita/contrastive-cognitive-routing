from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.agents import AgentExecutor
from langchain.callbacks import get_openai_callback
import gradio as gr

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", 
                 temperature=0, 
                 openai_api_key=os.environ.get("OPENAI_API_KEY"))

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia.run("Highest goals in a single season of La Liga")
tools = [wikipedia]

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
chat_history = []

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an epistemic-aware proxy agent."),
    ("human", "{input}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

with get_openai_callback() as cb:
    out = agent_executor.invoke({"input": "Highest goals in a single season of La Liga", "chat_history": chat_history})
print(out)
print(cb)

chat_history.extend(
    [
        HumanMessage(content="Highest goals in a single season of La Liga"),
        AIMessage(content=out["output"]),
    ]
)

print(chat_history)
agent_executor.invoke({"input": "How many goals he has scored overall in L Liga?", "chat_history": chat_history})

agent_history = []
def call_agent(query, chat_history):
    print("Chat history : ", chat_history)
    output = agent_executor.invoke({"input": query, "chat_history": agent_history})

    agent_history.extend(
    [
        HumanMessage(content="Highest goals in a single season of La Liga"),
        AIMessage(content=out["output"]),
    ]
    )


    chat_history += [
        [
            "<b>Question: </b>" + query,
            "<b>Answer: </b>" + output["output"]
        ]
    ]


    return output["output"], chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label = "QnA with Wikipedia")
    question = gr.Textbox(label = "Ask you query here")

    with gr.Row():
        submit = gr.Button("Submit")
        clear = gr.ClearButton([chatbot, question])

    def user(user_message, history):

        bot_message, history = call_agent(user_message, history)

        return "", history

    question.submit(user, [question, chatbot], [question, chatbot], queue=False)
    submit.click(user, [question, chatbot], [question, chatbot], queue=False)

demo.queue()
demo.launch()
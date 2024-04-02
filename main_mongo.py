import pandas as pd
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
app = FastAPI()
import json


client = MongoClient("mongodb+srv://niteshsinghal9917:bzDCS9Hmf6EOvXxP@cluster0.si0lxp7.mongodb.net/")
db = client['test']
collection = db['timesheetcrons']

cursor = collection.find({})

df = pd.DataFrame(list(cursor))
# print(df.shape)

df.drop(['_id', '__v','createdAt', 'updatedAt'], axis=1, inplace=True)

df.rename(columns={
    'username': 'name',
    'projectname': 'project',
    'taskname': 'task',
    'starttime': 'start time',
    'endtime': 'end time'
}, inplace=True)

# Creating new date column by Extract date component from start time
df['date'] = df['start time'].dt.date

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['total work hours'] = (df['end time'] - df['start time']).dt.total_seconds() / 3600.0

# print(df.dtypes)
# print(df.head())
# print(df.columns)

from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferWindowMemory
import gradio as gr
# from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()


# openai_api_key = "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"

TEMPLATE = """Your name is CatBot, you are here to help Admin with his questions over timesheet data.
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
you are designed to interact with admin, you are expected to question from data base , data base consist of employess login data.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.
You are expected to ask question from the data frame you have trained, use whole data frame to answer questions.
Admin is the one who are going to questions from you to get better insight from the data. So give accurate answer to admin so that he can have better idea about the data.
Since you are talking with admin, address him by calling 'BOSS' or similar words that can be addressed.

You should only take action with tool you have once you make your self clear about what admin is asking about.
if user is 'admin' is asking about any 'employees', 'task' or project, you should make sure you have a clear about about what user is asking about, then only take an action.
For example:
<question> "Tell me about the task 'Ajay' has done last week?" </question>
<logic>Ensure there is data of the name user is asking for, in this case it is 'ajay'.
Check if there is any name in user mentioning in his question, in names column/
If the name is found, proceed with further actions based on the user's query. If not, handle the case gracefully to avoid errors.</logic>

When you are confused about what user is asking about, feel free to ask question back to user so that he can give you better  querry so that you can understand what he meant.
for example:
<question>"what is the work hours of 'kumar' on task integration?</question>
<logic>Check if there is any kumar, if there is no  'kumar' in data frame you can check if there any name in df similar to the name
user asked and ask question back to user that did he meant this, for example did you mean 'rohit kumar?'
only ask questions back to user if you cant find data from the data base and there is a similar content in the data base similar to users querry</logic>
"""
# class PythonInputs(BaseModel):
#     query: str = Field(description="code snippet to run")

df=df
template = TEMPLATE.format(dhead=df.head().to_markdown())
from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(memory_key="chat_history")

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            # MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}")
        ]
    )

repl = PythonAstREPLTool(
            locals={"df": df},
            name="python_repl",
            description="Runs code and returns the output of the final line",
            verbose=True, 
            # handle_tool_error = True,
            # args_schema=PythonInputs(),
        )

# tools = repl
tools = [repl]

agent = create_openai_tools_agent(
            llm=ChatOpenAI(temperature=0.1,
                            openai_api_key = "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",
                            model="gpt-3.5-turbo-1106"), 
                            prompt=prompt, 
                            tools=tools
        )

agent_executor = AgentExecutor(agent=agent, 
                               tools=tools, 
                               max_iterations=10,
                               verbose=True, 
                               early_stopping_method="generate",
                               handle_parsing_errors=True, 
                            #    memory=memory,
                               )

allowed_origins = [
    "*",
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.post("/ask-question")
async def ask_question(question: str):
    response = agent_executor.invoke({"input": question})
    return response["output"]
def get_answer(question, history=[]):
    response = agent_executor.invoke({"input": question})
    return response["output"]
# agent_executor.invoke({"input": 'who is vinayak?'})
gr.ChatInterface(get_answer).launch()
# agent_executor.invoke({"input": "Tell me about the task Ajay has done last week?"})

from fastapi.responses import RedirectResponse

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"

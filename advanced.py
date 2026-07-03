from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool("get_weather", description="Get weather for a given city", return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

@tool('locate_user', description='Check location based on context')
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ789':
            return 'New York'
        case 'LMN456':
            return 'Tokyo'
        case _:
            return 'Unknown'
        
model = init_chat_model(
    model="openai:gpt-4.1-nano",
    temperature=0.1,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant, who is funny.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': '1'}}

response = agent.invoke(
    {
        'messages': [{'role': 'user', 'content': 'What is the weather like?'}]
    },
    config=config,
    context=Context(user_id='ABC123')
)

#print(response)
#print(response["messages"][-1].content)
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool("get_weather", description="Get weather for a given city", return_direct=True)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

agent = create_agent(
    model="openai:gpt-4.1-nano",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant, who is funny."
)

response = agent.invoke({
    'messages': [{'role': 'user', 'content': 'What is the weather in Vienna?'}]
})

print(response)
print(response["messages"][-1].content)

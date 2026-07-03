import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model="openai:gpt-4.1-nano",
    temperature=0.1,
)

conversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a high-level, interpreted programming language known for its readability and versatility."),
    HumanMessage("When was it created?")

]

response = model.invoke(conversation)

print (response.content)

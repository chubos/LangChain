import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="openai:gpt-4.1-nano",
    temperature=0.1,
)

response = model.invoke("What is Python?")

print (response.content)

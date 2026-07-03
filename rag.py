from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

texts = [
    'Apple makes very good computers.',
    'I believe Apple is innovative and a great company.',
    'I love apples',
    'I am a fan od MacBooks.',
    'I enjoy oranges.',
    'I like Lenovo ThinkPads.',
    'I think pears taste very good.'
    'I hate bananas and raspberries.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search("What company makes good computers?"))

retriever = vector_store.as_retriever(search_kwargs={'k': 3})
retriever_tool = create_retriever_tool(retriever, name='kb-search', description='Search knowledge base for relevant information')

agent = create_agent(
    model="openai:gpt-4.1-nano",
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant for questions about Macs, apples and laptops. First call the kb-search tool to find relevant information, then answer the question based on the retrieved information. You may need to call the kb-search tool multiple times to get all relevant information."
)

result = agent.invoke({
    'messages': [{'role': 'user', 'content': 'What three fruit does the person like and dislike?'}]
})

print(result["messages"][-1].content)

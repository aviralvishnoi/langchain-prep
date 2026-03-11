from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from loguru import logger

# Commented to use tavily search tool
# from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     logger.info(f"Searching for query {query}")
#     return tavily.search(query=query)
# tools = [search]

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    logger.info("Hello from langchain course")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Zoetermeer")}
    )
    logger.info(result)


if __name__ == "__main__":
    main()

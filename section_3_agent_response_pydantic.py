from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from loguru import logger


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answers"
    )


llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    logger.info("Hello from langchain course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="what is the capital for France")}
    )
    logger.info(result)


if __name__ == "__main__":
    main()

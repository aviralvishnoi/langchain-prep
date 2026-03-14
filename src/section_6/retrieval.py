import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger
from operator import itemgetter

load_dotenv()


logger.info("Initializing components...")
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
llm = ChatOllama(model="qwen3:1.7b")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question only on the following context
    
    {context} 
    
    Question: {question}
    
    Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    logger.info("=== Retrieving relevant documents ===")
    documents = retriever.invoke(query)

    logger.info("=== Preparing context ===")
    # Step 2: Format documents into context string
    context = format_docs(documents)

    logger.info("=== Formatting prompt message ===")
    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    logger.info("=== Invoking LLM ===")
    # Step 4: Invoke LLM with formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def main():
    logger.info("Retrieving...")

    # Query
    query = "When to use langchain?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    logger.info("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    logger.info("\nAnswer:")
    logger.info(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("IMPLEMENTATION 1: Without LCEL")
    logger.info("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    logger.info("\nAnswer:")
    logger.info(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)


if __name__ == "__main__":
    main()

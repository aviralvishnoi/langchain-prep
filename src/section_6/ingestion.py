import os
from dotenv import load_dotenv
from loguru import logger
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    logger.info("Ingesting...")
    loader = TextLoader(
        "/Users/aviralvishnoi/Documents-local/work/self-improvement/ai_enginner/langchain-course/data/when_to_use_langchain.txt"
    )

    document = loader.load()
    logger.info("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(documents=document)
    logger.info(f"created {len(texts)} chunks ...")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    logger.info("Loading embeddings...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )


if __name__ == "__main__":
    main()

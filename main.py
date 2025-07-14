from fastmcp import FastMCP
from typing import List
import logging
import os
import random

import chromadb
from langchain_core.documents.base import Document
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import PartyRetriever

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "./chroma/"
EMBEDDING_MODEL = "text-embedding-ada-002"


mcp = FastMCP("Hello World")

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

client = chromadb.PersistentClient(
    path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}"))

chroma = Chroma(
    collection_name=f"BTW2025",
    client=client,
    create_collection_if_not_exists=False
)

retriever = PartyRetriever.PartyRetriever(chroma, embedding)

@mcp.tool
def get_party_programs(query: str) -> List[Document]:
    """Retrieve party programs based on the query."""
    results = retriever.invoke(query)
    logger.info(f"Retrieved {len(results)} documents for query: {query}")
    return [doc for doc in results]

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]



# entry point - not needed but suggested
if __name__ == '__main__':
    mcp.run(transport="http",
        host="127.0.0.1",           # Bind to all interfaces
        port=8000,                # Custom port
        stateless_http=True,
        log_level="DEBUG")
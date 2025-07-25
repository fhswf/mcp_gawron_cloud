from fastmcp import FastMCP
from typing import List, Annotated
from pydantic import Field
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

server_instructions = """
This MCP server provides search and document retrieval capabilities 
for deep research. It provides the following tools:
1. `get_party_programs`: Retrieve party programs based on a user query.
2. `roll_dice`: Roll a specified number of 6-sided dice and return the results.
"""

mcp = FastMCP(name = "FH-SWF MCP server")

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

client = chromadb.PersistentClient(
    path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}"))

party_store = Chroma(
    collection_name=f"BTW2025",
    client=client,
    create_collection_if_not_exists=False
)

party_retriever = PartyRetriever.PartyRetriever(party_store, embedding)

@mcp.tool
def get_party_programs(query: Annotated[str, Field(description="The user query")],
                       party: Annotated[str, Field(description="the party the query relates to")] = "") -> List[Document]:
    """Retrieve party programs based on the query."""
    results = party_retriever.invoke(query, party)
    logger.info(f"Retrieved {len(results)} documents for query: {query}")
    return [doc for doc in results]

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]



# entry point - not needed but suggested
if __name__ == '__main__':
    mcp.run(transport="http",
        host="0.0.0.0",           # Bind to all interfaces
        port=8000,                # Custom port
        stateless_http=True,
        log_level="DEBUG")
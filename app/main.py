from fastmcp import FastMCP
from fastapi import FastAPI
from typing import List, Annotated, Dict
from pydantic import Field
import logging
import os
import random

import chromadb
from langchain_core.documents.base import Document
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from . import app, mcp
from .PartyRetriever import PartyRetriever
from .index import router as index_router

app.include_router(index_router)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "./chroma/"
EMBEDDING_MODEL = "text-embedding-ada-002"


embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

client = chromadb.PersistentClient(
    path=os.path.join(DATABASE_PATH, f"{EMBEDDING_MODEL}"))

party_store = Chroma(
    collection_name=f"BTW2025",
    client=client,
    create_collection_if_not_exists=False
)

party_retriever = PartyRetriever(party_store, embedding)


@app.get("/collections", tags=["ChromaDB"])
def get_collections() -> Dict[str, List[str]]:
    """Retrieve all collections in the ChromaDB."""
    collections = client.list_collections()
    logger.info(f"Retrieved {len(collections)} collections: {collections}")
    return {"collections": [c.name for c in collections]}


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

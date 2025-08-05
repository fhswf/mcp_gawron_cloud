import os
import json
import hashlib
import boto3
import chromadb

from pathlib import Path
from typing import Any
from fastapi import APIRouter, UploadFile


from flair.data import Sentence
from flair.models import SequenceTagger

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_docling import DoclingLoader
from langchain_docling.loader import MetaExtractor
from docling_core.transforms.chunker import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from .msword_backend import MsWordDocumentBackend
from docling.document_converter import DocumentConverter, FormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

import typer

from . import logger

# Konfiguration der Zugangsdaten und Endpunkt (für MinIO)
s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_ENDPOINT'),  # MinIO-Endpunkt
    aws_access_key_id=os.getenv('S3_KEY'),
    aws_secret_access_key=os.getenv('S3_SECRET'),
    region_name='us-east-1',  # Region bei MinIO meist egal
)


router = APIRouter()

cli = typer.Typer()

directory = Path(os.getenv("FILE_STORAGE", "files"))
max_characters = 10000
new_after_n_chars = 4000
overlap = 0
combine_text_under_n_chars_multiplier = int(new_after_n_chars*(2/3))

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-large"
MAX_TOKENS = 25000

# Create embeddings
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)

collection = "FH-SWF"

# Create a Chroma collection if it doesn't exist
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection_path = os.path.join(directory, collection)
db = Chroma(
    collection_name=collection,
    embedding_function=embeddings,
    client=chroma_client,
)


class FHSWFMetaExtractor(MetaExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the FLAIR model for NER
        self.tagger = SequenceTagger.load("flair/ner-german-large")

    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        meta = super().extract_chunk_meta(file_path, chunk)
        # Extract NERs from the chunk using spacy
        text = chunk.text
        # generate a SHA256 for the chunk
        hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        meta['sha256'] = hash
        ners = self.extract_ners(text)
        meta['entities'] = ners
        return meta

    def extract_ners(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text."""

        sentence = Sentence(text)
        self.tagger.predict(sentence)
        return [{"text": ent.text, "label": ent.tag} for ent in sentence.get_spans('ner')]


@cli.command(name="list")
def list_documents() -> None:
    """List all documents in the collection."""
    documents = db.get()
    if not documents:
        logger.info("No documents found in the collection.")
        return

    for doc in documents:
        logger.info(f"Document ID: {doc}")


@cli.command()
def index(path: Path) -> str:
    """Index a file into the collection."""
    bucket_name = "public"
    target_path = os.path.join("mcp", path)
    s3.upload_file(path, bucket_name, target_path)

    logger.info(f"Indexing file at {path} into collection {collection}")

    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = False
    format_options = {
        InputFormat.DOCX: FormatOption(
            pipeline_cls=SimplePipeline, backend=MsWordDocumentBackend
        ),
        InputFormat.PDF: FormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pdf_pipeline_options,
            backend=DoclingParseV4DocumentBackend)
    }

    converter = DocumentConverter(format_options=format_options)

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
    )

    meta_extractor = FHSWFMetaExtractor()

    chunks = DoclingLoader(
        file_path=path,
        converter=converter,
        chunker=HybridChunker(tokenizer=tokenizer),
        meta_extractor=meta_extractor,
    ).load()

    logger.info(f"Loaded {len(chunks)} chunks from file: {path}")

    docs = filter_complex_metadata(chunks)
    logger.info(f"Metadata after filtering: {[d.metadata for d in docs]}")

    # Add the document to the collection
    # db.add_documents(docs)

    return f"File '{path}' indexed in collection '{collection}' successfully."


@router.post("/index")
async def index(file: UploadFile, collection: str) -> str:
    """
    Indexing endpoint.
    Upload a file to be indexed in `collection`
    """
    content = await file.read()

    # store the file in the `files` directory
    logger.info(f"Received file: {file.filename} for collection: {collection}")

    file_path = os.path.join(directory, collection, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logger.info(f"Storing file at: {file_path}")
    with open(file_path, "wb") as f:
        f.write(content)

    index(file_path, collection)


if __name__ == "__main__":
    cli()

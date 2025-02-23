import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from typing import List

nest_asyncio.apply()

_ = load_dotenv()

Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="gemini-2.0-flash-001",
    invalidate_cache=True,
    parsing_instruction="",
)

def parse_document(file_path: str) -> List[Document]:
    """Parse the document and return the text content."""
    documents = parser.load_data(file_path)
    return documents

def ingest_documents(file_path: str, persist_dir: str):
    """Ingest the document and return the index"""
    print("pushing the document to the vector index")
    documents = parse_document(file_path)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    index = VectorStoreIndex.from_documents(documents)
    return index



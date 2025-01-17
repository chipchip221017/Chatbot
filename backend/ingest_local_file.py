import logging
import os
import re
from parser import langchain_docs_extractor

from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import chromadb
from chromadb.utils.batch_utils import create_batches
import uuid

import google.generativeai as genai

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is not set. Please set it in your .env file.")
    exit(1)

def get_embeddings_model(type_of_model: str) -> Embeddings:
    if type_of_model == "ollama":
        return OllamaEmbeddings(model='nomic-embed-text')
    elif type_of_model == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def load_data():
    loader = DirectoryLoader('./data', glob="**/*.pdf",use_multithreading = True, show_progress=True)
    return loader.load()


data_dir = "data"
persist_directory = './my_chroma_data'

# Function to list all files in the data directory
def get_all_file_paths(directory, file_extension=".pdf"):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Function to check if the documents are already loaded in Chroma
def documents_already_loaded(vector_store, file_paths):
    existing_metadata = vector_store.get()["metadatas"]
    existing_files = {metadata['source'].replace("/", "\\") for metadata in existing_metadata if 'source' in metadata}
    return all(file.replace("/", "\\") in existing_files for file in file_paths)


def ingest_docs():
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'default_database_host')
    DATABASE_PORT = os.getenv('DATABASE_PORT', 'default_database_port')
    DATABASE_USERNAME = os.getenv('DATABASE_USERNAME', 'default_database_user')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'default_database_password')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'default_database_name')
    RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", 'default_collection_name')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding = get_embeddings_model("gemini")

    # Get the list of files in the data directory
    files_in_data_dir = get_all_file_paths(data_dir)

    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False,
        headers=None,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    client.get_or_create_collection(COLLECTION_NAME)

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist_directory
    )

    # Check if Chroma has already loaded all the files
    if os.path.exists(persist_directory):
        # Load existing Chroma vector store
        print("Loading existing Chroma vector store...")

        # Check if all files are already loaded in Chroma
        if documents_already_loaded(vector_store, files_in_data_dir):
            print("All files are already loaded in Chroma. Skipping loading process.")
        else:
            print("Not all files are loaded in Chroma. Loading remaining files.")
    else:
        # Load all files from the data directory
        print("No existing vector store found. Processing documents...")

    record_manager = SQLRecordManager(
    f"weaviate/{COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    docs_from_documentation = load_data()

    docs_transformed = text_splitter.split_documents(
        docs_from_documentation
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]
    
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vector_store,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")

if __name__ == "__main__":
    ingest_docs()
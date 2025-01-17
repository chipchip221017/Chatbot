# from langchain_core.embeddings import FakeEmbeddings
# from langchain_chroma import Chroma
# import numpy as np

# import chromadb
# from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

# client = chromadb.HttpClient(
#     host="localhost",
#     port=8000,
#     ssl=False,
#     headers=None,
#     settings=Settings(),
#     tenant=DEFAULT_TENANT,
#     database=DEFAULT_DATABASE,
# )


# embeddings = FakeEmbeddings(size=4096)

# vector_store = Chroma(
#     client=client,
#     collection_name="example_collection_from_test.py",
#     embedding_function=embeddings,
#     # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )

# # persistent_client = chromadb.PersistentClient()
# # collection = persistent_client.get_or_create_collection("collection_name")
# # collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

# # vector_store_from_client = Chroma(
# #     client=persistent_client,
# #     collection_name="collection_name",
# #     embedding_function=embeddings,
# # )

# from uuid import uuid4

# from langchain_core.documents import Document

# document_1 = Document(
#     page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
#     metadata={"source": "tweet"},
#     id=1,
# )

# document_2 = Document(
#     page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
#     metadata={"source": "news"},
#     id=2,
# )

# document_3 = Document(
#     page_content="Building an exciting new project with LangChain - come check it out!",
#     metadata={"source": "tweet"},
#     id=3,
# )

# document_4 = Document(
#     page_content="Robbers broke into the city bank and stole $1 million in cash.",
#     metadata={"source": "news"},
#     id=4,
# )

# document_5 = Document(
#     page_content="Wow! That was an amazing movie. I can't wait to see it again.",
#     metadata={"source": "tweet"},
#     id=5,
# )

# document_6 = Document(
#     page_content="Is the new iPhone worth the price? Read this review to find out.",
#     metadata={"source": "website"},
#     id=6,
# )

# document_7 = Document(
#     page_content="The top 10 soccer players in the world right now.",
#     metadata={"source": "website"},
#     id=7,
# )

# document_8 = Document(
#     page_content="LangGraph is the best framework for building stateful, agentic applications!",
#     metadata={"source": "tweet"},
#     id=8,
# )

# document_9 = Document(
#     page_content="The stock market is down 500 points today due to fears of a recession.",
#     metadata={"source": "news"},
#     id=9,
# )

# document_10 = Document(
#     page_content="I have a bad feeling I am going to get deleted :(",
#     metadata={"source": "tweet"},
#     id=10,
# )

# documents = [
#     document_1,
#     document_2,
#     document_3,
#     document_4,
#     document_5,
#     document_6,
#     document_7,
#     document_8,
#     document_9,
#     document_10,
# ]


# uuids = [str(uuid4()) for _ in range(len(documents))]

# vector_store.add_documents(documents=documents, ids=uuids)



# results = vector_store.similarity_search(
#     "LangChain provides abstractions to make working with LLMs easy",
#     k=2,
#     filter={"source": "tweet"},
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

import chromadb

client = chromadb.HttpClient()
print("Number of documents that can be inserted at once: ",client.get_max_batch_size())
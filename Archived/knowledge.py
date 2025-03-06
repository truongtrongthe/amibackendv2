from langchain_text_splitters import CharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = "us-east-1"  # Check Pinecone console for your region
INDEX_NAME = "hitoindex"

index_name = INDEX_NAME
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [i['name'] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize embeddings
embeddings = OpenAIEmbeddings()

index = pc.Index(index_name)

def tobrain(summary, raw_content):
    if not summary and not raw_content:
        print("⚠️ Error: No content to store in Pinecone.")
        return

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="\n\n")
    summary_chunks = text_splitter.split_text(summary or "")
    raw_chunks = text_splitter.split_text(raw_content or "")

    # Convert to Document objects with metadata
    summary_docs = [Document(page_content=chunk, metadata={"type": "summary", "content": chunk}) for chunk in summary_chunks]
    raw_docs = [Document(page_content=chunk, metadata={"type": "raw", "content": chunk}) for chunk in raw_chunks]
    all_docs = summary_docs + raw_docs

    print(f"📌 Number of summary docs: {len(summary_docs)}")
    print(f"📌 Number of raw docs: {len(raw_docs)}")

    if not all_docs:
        print("⚠️ Error: No valid content to store in Pinecone.")
        return

    # Generate embeddings
    vectors = [
        {
            "id": f"doc_{i}",
            "values": embeddings.embed_query(doc.page_content),
            "metadata": doc.metadata  # ✅ Store content inside metadata
        }
        for i, doc in enumerate(all_docs)
    ]

    # Upsert into Pinecone
    index.upsert(vectors)

def save_to_pinecone(user_id, embedding):
    index = pc.Index(index_name)
    index.upsert(vectors)

def retrieve_relevant_info(query, k=1):
    index = pc.Index(index_name)
    query_embedding = embeddings.embed_query(query)

    # Debugging: Ensure embedding is valid
    if not all(isinstance(x, float) for x in query_embedding):
        raise ValueError("Embedding contains invalid (non-float) values!")

    results = index.query(
        vector=query_embedding,  # ✅ Use "vector" instead of "queries"
        top_k=k,
        include_metadata=True
    )

    print("Raw results:", results)

    matches = results.get("matches", [])

    retrieved_docs = [
        {
            "id": match["id"],
            "score": match["score"],
            "content": match["metadata"].get("content", "")
        }
        for match in matches
    ]

    return retrieved_docs

def retrieve_relevant_infov2(query, top_k=5):
    """
    Retrieves relevant information from Pinecone based on a query.
    
    Args:
        query (str): The search query.
        top_k (int): Number of results to retrieve.

    Returns:
        List of relevant documents with scores.
    """
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query)

    # Perform query search in Pinecone
    index = pc.Index(index_name)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extract relevant data
    retrieved_docs = [
        {
            "id": match["id"],
            "score": match["score"],
            "content": match["metadata"].get("content", "No content found")  # ✅ Fix missing content issue
        }
        for match in results["matches"]
    ]

    return retrieved_docs

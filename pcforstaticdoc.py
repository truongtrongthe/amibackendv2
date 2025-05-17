import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger
from datetime import datetime, timedelta
import uuid
import asyncio
import json
from typing import Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
import backoff

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
ent_index_name = os.getenv("ENT", "ent-index")

# Create or connect to index with serverless configuration
def initialize_index():
    try:
        if ent_index_name not in pc.list_indexes().names():
            pc.create_index(
                name=ent_index_name,
                dimension=1536,  # Matches OpenAI embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            logger.info(f"Created Pinecone index: {ent_index_name}")
        return pc.Index(ent_index_name)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index: {e}")
        raise

ent_index = initialize_index()

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def save_knowledge(
    input: str,
    user_id: str,
    bank_name: str = "",
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    categories: Optional[List[str]] = None,
    ttl_days: Optional[int] = None,
    batch_size: int = 500,  # Increased for document chunks
    metadata: Optional[Dict] = None
) -> bool:
    """
    Save knowledge to Pinecone vector database with optimized vector structure for documents.

    Args:
        input: The knowledge content to save (e.g., document chunk text).
        user_id: Identifier for the user.
        bank_name: Namespace for the knowledge (optional).
        thread_id: Document or conversation identifier (e.g., doc_id).
        topic: Core topic of the knowledge (e.g., "Introduction | Subsection").
        categories: List of categories (e.g., ["document", "type_paragraph", "section_Introduction"]).
        ttl_days: Time-to-live in days (optional, for data expiration).
        batch_size: Number of vectors to upsert in one batch.
        metadata: Additional document metadata (e.g., section, subsection, type, doc_id).

    Returns:
        bool: True if saved successfully, False otherwise.

    Raises:
        ValueError: If required parameters are invalid.
    """
    try:
        # Validate inputs
        if not input.strip():
            logger.warning("Empty input provided, skipping save")
            return False
        if not user_id:
            raise ValueError("user_id is required")
        ns = bank_name or "default"
        if not ns.strip():
            raise ValueError("bank_name cannot be empty or whitespace")

        logger.info(f"Saving to index={ent_index_name}, namespace={ns}, user_id={user_id}, thread_id={thread_id}, topic={topic}")

        # Generate embedding
        embedding = await EMBEDDINGS.aembed_query(input)
        if len(embedding) != 1536:
            logger.error(f"Invalid embedding dimension: {len(embedding)}")
            return False

        # Dynamic confidence based on input length
        confidence = min(0.9, 0.5 + len(input) / 1000)

        # Duplicate check with document-specific metadata
        filter_dict = {"user_id": user_id, "raw": {"$eq": input[:2000]}}
        if metadata and metadata.get("doc_id"):
            filter_dict["doc_id"] = metadata["doc_id"]
        if metadata and metadata.get("section"):
            filter_dict["section"] = metadata["section"]

        existing = await asyncio.to_thread(
            ent_index.query,
            vector=embedding,
            top_k=1,
            include_metadata=True,
            namespace=ns,
            filter=filter_dict
        )
        if existing.get("matches", []) and existing["matches"][0]["score"] > 0.95:
            logger.info(f"Skipping near-duplicate: '{input[:50]}...' exists as {existing['matches'][0]['id']}")
            return True

        # Optimized metadata
        pinecone_metadata = {
            "created_at": datetime.now().isoformat(),
            "raw": input[:2000],  # Truncate to save space
            "confidence": float(confidence),
            "source": "document",
            "user_id": user_id,
            "thread_id": thread_id or "",
            "topic": topic or "unknown",
            "categories": categories[:10] if categories else ["general"]
        }
        # Add document-specific metadata
        if metadata:
            pinecone_metadata.update({
                "doc_id": metadata.get("doc_id", ""),
                "section": metadata.get("section", ""),
                "subsection": metadata.get("subsection", ""),
                "type": metadata.get("type", "unknown"),
                "table_index": metadata.get("table_index", None),
                "row_index": metadata.get("row_index", None),
                "page_number": metadata.get("page_number", None),
                "headers": metadata.get("headers", [])
            })
        if ttl_days:
            pinecone_metadata["expires_at"] = (datetime.now() + timedelta(days=ttl_days)).isoformat()

        # Prepare vector for upsert
        convo_id = f"{user_id}_{uuid.uuid4()}"
        vector = [(convo_id, embedding, pinecone_metadata)]

        logger.info(f"Upserting knowledge with convo_id={convo_id}, metadata={json.dumps(pinecone_metadata, ensure_ascii=False)}")
        await asyncio.to_thread(
            ent_index.upsert,
            vectors=vector,
            namespace=ns,
            batch_size=batch_size
        )
        return True
    except Exception as e:
        logger.error(f"Upsert failed after retries: {e}")
        return False

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def query_knowledge(
    query: str,
    bank_name: str = "",
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    doc_id: Optional[str] = None,
    section: Optional[str] = None,
    type: Optional[str] = None,
    page_number: Optional[int] = None,
    table_index: Optional[int] = None,
    top_k: int = 10,
    min_similarity: float = 0.3,
    ef_search: int = 100
) -> List[Dict]:
    """
    Query knowledge from Pinecone with optimized vector structure and document-specific filtering.

    Args:
        query: The search query.
        bank_name: Namespace to query (optional).
        user_id: Filter by user (optional).
        thread_id: Filter by document or conversation identifier (optional).
        topic: Filter by topic (e.g., "Introduction | Subsection").
        doc_id: Filter by document ID (optional).
        section: Filter by document section (optional).
        type: Filter by chunk type (e.g., "paragraph", "table", "pdf_text").
        page_number: Filter by PDF page number (optional).
        table_index: Filter by table index (optional).
        top_k: Maximum number of results.
        min_similarity: Minimum similarity score for matches.
        ef_search: Controls search-time exploration for better recall.

    Returns:
        List of knowledge entries sorted by score, including document metadata.
    """
    try:
        if not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []

        embedding = await EMBEDDINGS.aembed_query(query)
        if len(embedding) != 1536:
            logger.error(f"Invalid query embedding dimension: {len(embedding)}")
            return []

        ns = bank_name or "default"
        if not ns.strip():
            raise ValueError("bank_name cannot be empty or whitespace")
        knowledge = []

        logger.info(f"Querying index={ent_index_name}, namespace={ns}, user_id={user_id}, thread_id={thread_id}, topic={topic}, doc_id={doc_id}, section={section}")

        # Dynamic filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if thread_id:
            filter_dict["thread_id"] = thread_id
        if topic:
            filter_dict["topic"] = topic
        if doc_id:
            filter_dict["doc_id"] = doc_id
        if section:
            filter_dict["section"] = section
        if type:
            filter_dict["type"] = type
        if page_number is not None:
            filter_dict["page_number"] = page_number
        if table_index is not None:
            filter_dict["table_index"] = table_index

        results = await asyncio.to_thread(
            ent_index.query,
            vector=embedding,
            top_k=top_k * 2,
            include_metadata=True,
            namespace=ns,
            filter=filter_dict or None,
            ef_search=ef_search
        )
        matches = results.get("matches", [])
        
        # Current time for post-filtering
        current_time = datetime.now().isoformat()

        # Filter and format results, including expiration check
        knowledge = [
            {
                "id": match["id"],
                "raw": match["metadata"]["raw"],
                "created_at": match["metadata"]["created_at"],
                "expires_at": match["metadata"].get("expires_at"),
                "confidence": match["metadata"]["confidence"],
                "score": match["score"],
                "topic": match["metadata"].get("topic", "unknown"),
                "categories": match["metadata"].get("categories", ["general"]),
                "doc_id": match["metadata"].get("doc_id", ""),
                "section": match["metadata"].get("section", ""),
                "subsection": match["metadata"].get("subsection", ""),
                "type": match["metadata"].get("type", "unknown"),
                "table_index": match["metadata"].get("table_index"),
                "row_index": match["metadata"].get("row_index"),
                "page_number": match["metadata"].get("page_number"),
                "headers": match["metadata"].get("headers", [])
            }
            for match in matches 
            if match["score"] >= min_similarity and
               (not match["metadata"].get("expires_at") or match["metadata"].get("expires_at", "") > current_time)
        ]

        # Sort and limit
        knowledge = sorted(knowledge, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.info(f"Queried {len(knowledge)} knowledge entries for '{query[:50]}...', top score={knowledge[0]['score'] if knowledge else 0}")
        return knowledge
    except Exception as e:
        logger.error(f"Query failed after retries: {e}")
        return []
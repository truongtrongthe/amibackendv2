import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger
from datetime import datetime
import uuid
import asyncio
import json
from typing import Dict, List, Optional
from pinecone import Pinecone
import backoff

# Initialize Pinecone client with proper error handling
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
ent_index_name = os.getenv("ENT", "ent-index")
try:
    ent_index = pc.Index(ent_index_name)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone index: {e}")
    raise

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def save_knowledge(
    input: str,
    user_id: str,
    bank_name: str = "",
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    categories: Optional[List[str]] = None
) -> bool:
    """
    Save knowledge to Pinecone vector database with enhanced metadata.

    Args:
        input: The knowledge content to save.
        user_id: Identifier for the user.
        bank_name: Namespace for the knowledge (optional).
        thread_id: Conversation thread identifier (optional).
        topic: Core topic of the knowledge (e.g., "phân nhóm khách hàng").
        categories: List of categories (e.g., ["health_segmentation"]).

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        embedding = await EMBEDDINGS.aembed_query(input)
        ns = bank_name or "default"
        target_index = ent_index

        logger.info(f"Saving to index={ent_index_name}, namespace={ns}, user_id={user_id}, thread_id={thread_id}")

        # Dynamic confidence based on input length
        confidence = min(0.9, 0.5 + len(input) / 1000)
        
        # Duplicate check with fuzzy matching
        existing = await asyncio.to_thread(
            target_index.query,
            vector=embedding,
            top_k=1,
            include_metadata=True,
            namespace=ns,
            filter={
                "user_id": user_id,
                "raw": {"$eq": input}  # Exact match for raw input
            }
        )
        if existing.get("matches", []) and existing["matches"][0]["score"] > 0.95:
            logger.info(f"Skipping near-duplicate: '{input}' exists as {existing['matches'][0]['id']}")
            return True

        # Enhanced metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "raw": input,
            "confidence": confidence,
            "source": "conversation",
            "user_id": user_id,
            "thread_id": thread_id or "",
            "topic": topic or "unknown",
            "categories": categories or ["general"],
            "categories_special": "document" if categories else "general"
        }
        
        convo_id = f"{user_id}_{uuid.uuid4()}"
        logger.info(f"Upserting knowledge with convo_id={convo_id}, metadata={json.dumps(metadata)}")
        await asyncio.to_thread(
            target_index.upsert,
            vectors=[(convo_id, embedding, metadata)],
            namespace=ns
        )
        return True
    except Exception as e:
        logger.error(f"Upsert failed after retries: {e}")
        return False

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def query_knowledge(
    query: str,
    bank_name: str = "",
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    top_k: int = 10,
    min_similarity: float = 0.3
) -> List[Dict]:
    """
    Query knowledge from Pinecone vector database with context-aware filters.

    Args:
        query: The search query.
        bank_name: Namespace to query (optional).
        user_id: Filter by user (optional).
        thread_id: Filter by conversation thread (optional).
        topic: Filter by topic (e.g., "phân nhóm khách hàng").
        top_k: Maximum number of results.
        min_similarity: Minimum similarity score for matches.

    Returns:
        List of knowledge entries sorted by score.
    """
    try:
        embedding = await EMBEDDINGS.aembed_query(query)
        ns = bank_name or "default"
        knowledge = []

        logger.info(f"Querying index={ent_index_name}, namespace={ns}, user_id={user_id}, thread_id={thread_id}, topic={topic}")

        # Dynamic filter based on provided parameters
        filter_dict = {
            "categories_special": {"$in": ["general", "description", "document", "procedural"]}
        }
        if user_id:
            filter_dict["user_id"] = user_id
        if thread_id:
            filter_dict["thread_id"] = thread_id
        if topic:
            filter_dict["topic"] = topic

        results = await asyncio.to_thread(
            ent_index.query,
            vector=embedding,
            top_k=top_k * 2,  # Oversample to filter later
            include_metadata=True,
            namespace=ns,
            filter=filter_dict
        )
        matches = results.get("matches", [])
        
        # Filter by minimum similarity and extract relevant fields
        knowledge = [
            {
                "id": match["id"],
                "raw": match["metadata"]["raw"],
                "created_at": match["metadata"]["created_at"],
                "confidence": match["metadata"]["confidence"],
                "score": match["score"],
                "topic": match["metadata"].get("topic", "unknown"),
                "categories": match["metadata"].get("categories", ["general"])
            }
            for match in matches if match["score"] >= min_similarity
        ]
        
        # Sort and limit to top_k
        knowledge = sorted(knowledge, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.info(f"Queried {len(knowledge)} knowledge entries for '{query}', top score={knowledge[0]['score'] if knowledge else 0}")
        return knowledge
    except Exception as e:
        logger.error(f"Query failed after retries: {e}")
        return []
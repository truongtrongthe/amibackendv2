import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger
from datetime import datetime, timedelta
import uuid
import asyncio
import json
from typing import Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
import backoff
from supabase import create_client, Client
import traceback

# Load environment variables from .env file
load_dotenv()

spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))

# Global production index
_production_index = None

def get_production_index():
    """
    Get or create the single production Pinecone index.
    
    Returns:
        Pinecone index for production
    """
    global _production_index
    try:
        if _production_index is None:
            index_name = "production"
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # Matches OpenAI embeddings
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
                logger.info(f"Created production Pinecone index: {index_name}")
            _production_index = pc.Index(index_name)
        return _production_index
    except Exception as e:
        logger.error(f"Failed to initialize production Pinecone index: {e}")
        raise

async def get_brain_banks(graph_version_id: str, org_id: str) -> List[Dict[str, str]]:
    """
    Get the bank names for all brains in a version, ensuring the version belongs to the given org_id.

    Args:
        graph_version_id: UUID of the graph version
        org_id: UUID of the organization

    Returns:
        List of dicts containing brain_id and bank_name
    """
    try:
        # Join brain_graph_version with brain_graph to get org_id
        version_response = supabase.table("brain_graph_version")\
            .select("brain_ids,status,brain_graph:graph_id(org_id)")\
            .eq("id", graph_version_id)\
            .execute()

        if not version_response.data:
            logger.error(f"Version {graph_version_id} not found")
            return []

        version_data = version_response.data[0]
        # Check if the linked brain_graph has the correct org_id
        if not version_data.get("brain_graph") or version_data["brain_graph"].get("org_id") != org_id:
            logger.error(f"Version {graph_version_id} does not belong to organization {org_id}")
            return []

        if version_data["status"] != "published":
            logger.warning(f"Version {graph_version_id} is not published")
            return []

        brain_ids = version_data["brain_ids"]
        if not brain_ids:
            logger.warning(f"No brain IDs found for version {graph_version_id}")
            return []
        
        brain_response = supabase.table("brain")\
            .select("id", "brain_id", "bank_name")\
            .in_("id", brain_ids)\
            .eq("org_id", org_id)\
            .execute()

        if not brain_response.data:
            logger.warning(f"No brain data found for brain IDs: {brain_ids}")
            return []

        brain_banks = [
            {
                "id": brain["id"],
                "brain_id": brain["brain_id"],
                "bank_name": brain["bank_name"]
            }
            for brain in brain_response.data
        ]

        logger.info(f"Retrieved brain structure for version {graph_version_id}, {len(brain_banks)} brains")
        return brain_banks
    except Exception as e:
        logger.error(f"Error getting brain banks: {e}")
        logger.error(traceback.format_exc())
        return []

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def save_knowledge(
    input: str,
    user_id: str,
    org_id: str,
    title: str="",
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    categories: Optional[List[str]] = None,
    ttl_days: Optional[int] = None,
    batch_size: int = 100
) -> Dict:
    """
    Save knowledge to production Pinecone index with organization-specific namespace.

    Args:
        input: The knowledge content to save.
        user_id: Identifier for the user.
        org_id: Organization ID.
        title: Title for the knowledge entry.
        thread_id: Conversation thread identifier (optional).
        topic: Core topic of the knowledge.
        categories: List of categories.
        ttl_days: Time-to-live in days (optional).
        batch_size: Number of vectors to upsert in one batch.

    Returns:
        Dict containing success status and vector details.
    """
    try:
        ns = f"ent-{org_id}"
        target_index = get_production_index()

        logger.info(f"Saving to production index, namespace={ns}, user_id={user_id}, thread_id={thread_id}")

        # Generate embedding
        embedding = await EMBEDDINGS.aembed_query(input)
        if len(embedding) != 1536:
            logger.error(f"Invalid embedding dimension: {len(embedding)}")
            return False

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
                "raw": {"$eq": input}  # Remove truncation - let it flow!
            }
        )
        if existing.get("matches", []) and existing["matches"][0]["score"] > 0.95:
            existing_id = existing["matches"][0]["id"]
            logger.info(f"Skipping near-duplicate: '{input[:50]}...' exists as {existing_id}")
            return {
                "success": True,
                "vector_id": existing_id,
                "namespace": ns,
                "created_at": existing["matches"][0]["metadata"]["created_at"],
                "duplicate_detected": True
            }

        # Optimized metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "title": title,
            "raw": input,  # Remove truncation - let it flow!
            "confidence": float(confidence),
            "source": "conversation",
            "user_id": user_id,
            "thread_id": thread_id or "",
            "topic": topic or "unknown",
            "categories": categories if categories else ["general"]  # Remove category limit - let it flow!
        }
        if ttl_days:
            metadata["expires_at"] = (datetime.now() + timedelta(days=ttl_days)).isoformat()

        # Prepare vector for upsert
        convo_id = f"{user_id}_{uuid.uuid4()}"
        vector = [(convo_id, embedding, metadata)]

        logger.info(f"Upserting knowledge with convo_id={convo_id}, metadata={json.dumps(metadata, ensure_ascii=False)}")
        await asyncio.to_thread(
            target_index.upsert,
            vectors=vector,
            namespace=ns,
            batch_size=batch_size
        )
        return {
            "success": True,
            "vector_id": convo_id,
            "namespace": ns,
            "created_at": metadata["created_at"]
        }
    except Exception as e:
        logger.error(f"Upsert failed after retries: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def query_knowledge(
    query: str,
    org_id: str,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    top_k: int = 10,
    min_similarity: float = 0.3,
    ef_search: int = 100
) -> List[Dict]:
    """
    Query knowledge from production Pinecone index with organization-specific namespace.

    Args:
        query: The search query.
        org_id: Organization ID.
        user_id: Filter by user (optional).
        thread_id: Filter by conversation thread (optional).
        topic: Filter by topic.
        top_k: Maximum number of results.
        min_similarity: Minimum similarity score for matches.
        ef_search: Controls search-time exploration.

    Returns:
        List of knowledge entries sorted by score.
    """
    try:
        embedding = await EMBEDDINGS.aembed_query(query)
        if len(embedding) != 1536:
            logger.error(f"Invalid query embedding dimension: {len(embedding)}")
            return []

        ns = f"ent-{org_id}"
        target_index = get_production_index()
        knowledge = []

        logger.info(f"Querying production index, namespace={ns}, user_id={user_id}, thread_id={thread_id}, topic={topic}")

        # Dynamic filter
        filter_dict = {}
        #if user_id:
        #    filter_dict["user_id"] = user_id
        #if thread_id:
        #    filter_dict["thread_id"] = thread_id
        #if topic:
        #    filter_dict["topic"] = topic

        results = await asyncio.to_thread(
            target_index.query,
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
                "categories": match["metadata"].get("categories", ["general"])
            }
            for match in matches 
            if match["score"] >= min_similarity and
               # Filter out expired entries manually
               (not match["metadata"].get("expires_at") or match["metadata"].get("expires_at", "") > current_time)
        ]

        # Sort and limit
        knowledge = sorted(knowledge, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.info(f"Queried {len(knowledge)} knowledge entries for '{query[:50]}...', top score={knowledge[0]['score'] if knowledge else 0}")
        return knowledge
    except Exception as e:
        logger.error(f"Query failed after retries: {e}")
        return []

async def query_knowledge_from_graph(
    query: str,
    graph_version_id: str,
    org_id: str,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    top_k: int = 10,
    min_similarity: float = 0.3,
    ef_search: int = 100,
    exclude_categories: Optional[List[str]] = None,
    include_categories: Optional[List[str]] = None
) -> List[Dict]:
    """
    Query knowledge from production Pinecone index with organization-specific namespace for a graph version.

    Args:
        query: The search query.
        graph_version_id: The graph version ID.
        org_id: Organization ID.
        user_id: The user ID.
        thread_id: The thread ID.
        topic: The topic.
        top_k: The top K.
        min_similarity: Minimum similarity threshold.
        ef_search: Search exploration factor.
        exclude_categories: List of categories to exclude from results.
        include_categories: List of categories to include (only these categories will be returned).
    """
    
    # Since we're using a single namespace per org, we query the org's namespace directly
    knowledge = await query_knowledge(query, org_id, user_id, thread_id, topic, top_k, min_similarity, ef_search)

    # Post-process to filter categories
    if include_categories:
        # Only include entries that have at least one of the included categories
        filtered_knowledge = []
        for entry in knowledge:
            entry_categories = entry.get("categories", [])
            if any(cat in entry_categories for cat in include_categories):
                filtered_knowledge.append(entry)
        logger.info(f"Filtered to include only {len(filtered_knowledge)} entries with included categories: {include_categories}")
        knowledge = filtered_knowledge
    elif exclude_categories:
        # Exclude entries that have any of the excluded categories
        filtered_knowledge = []
        for entry in knowledge:
            entry_categories = entry.get("categories", [])
            if not any(cat in entry_categories for cat in exclude_categories):
                filtered_knowledge.append(entry)
        logger.info(f"Filtered out {len(knowledge) - len(filtered_knowledge)} entries with excluded categories: {exclude_categories}")
        knowledge = filtered_knowledge
    
    logger.info(f"ALL KNOWLEDGE: {knowledge}")
    return knowledge

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def fetch_vector(vector_id: str, org_id: str) -> Dict:
    """
    Fetch a specific vector by its ID from production Pinecone index with organization-specific namespace.

    Args:
        vector_id: The unique vector ID to fetch.
        org_id: Organization ID.

    Returns:
        Dict containing the vector data or error information.
    """
    try:
        ns = f"ent-{org_id}"
        target_index = get_production_index()

        logger.info(f"Fetching vector {vector_id} from production index, namespace={ns}")

        # Fetch the specific vector
        result = await asyncio.to_thread(
            target_index.fetch,
            ids=[vector_id],
            namespace=ns
        )
        
        # Handle Pinecone FetchResponse object correctly
        vectors = result.vectors if hasattr(result, 'vectors') else {}
        if vector_id not in vectors:
            logger.warning(f"Vector {vector_id} not found in namespace {ns}")
            return {
                "success": False,
                "error": f"Vector {vector_id} not found",
                "vector_id": vector_id,
                "namespace": ns
            }

        vector_data = vectors[vector_id]
        metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else {}
        
        # Check if vector has expired
        current_time = datetime.now().isoformat()
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        if metadata_dict.get("expires_at") and metadata_dict.get("expires_at", "") <= current_time:
            logger.warning(f"Vector {vector_id} has expired at {metadata_dict.get('expires_at')}")
            return {
                "success": False,
                "error": f"Vector {vector_id} has expired",
                "vector_id": vector_id,
                "expired_at": metadata_dict.get("expires_at")
            }

        logger.info(f"Successfully fetched vector {vector_id}")
        return {
            "success": True,
            "vector_id": vector_id,
            "namespace": ns,
            "raw": metadata_dict.get("raw", ""),
            "title": metadata_dict.get("title", ""),
            "created_at": metadata_dict.get("created_at", ""),
            "expires_at": metadata_dict.get("expires_at"),
            "confidence": metadata_dict.get("confidence", 0.0),
            "source": metadata_dict.get("source", ""),
            "user_id": metadata_dict.get("user_id", ""),
            "thread_id": metadata_dict.get("thread_id", ""),
            "topic": metadata_dict.get("topic", "unknown"),
            "categories": metadata_dict.get("categories", ["general"]),
            "metadata": metadata_dict
        }
    except Exception as e:
        logger.error(f"Failed to fetch vector {vector_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "vector_id": vector_id
        }

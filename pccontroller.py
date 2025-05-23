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
from supabase import create_client, Client
import traceback

spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)



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


async def get_brain_banks(graph_version_id: str) -> List[Dict[str, str]]:
        """
        Get the bank names for all brains in a version

        Args:
            version_id: UUID of the graph version

        Returns:
            List of dicts containing brain_id and bank_name
        """
        try:
            
            version_response = supabase.table("brain_graph_version")\
                .select("brain_ids", "status")\
                .eq("id", graph_version_id)\
                .execute()

            if not version_response.data:
                logger.error(f"Version {graph_version_id} not found")
                return []

            version_data = version_response.data[0]
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
    bank_name: str = "",
    thread_id: Optional[str] = None,
    topic: Optional[str] = None,
    categories: Optional[List[str]] = None,
    ttl_days: Optional[int] = None,
    batch_size: int = 100
) -> bool:
    """
    Save knowledge to Pinecone vector database with optimized vector structure.

    Args:
        input: The knowledge content to save.
        user_id: Identifier for the user.
        bank_name: Namespace for the knowledge (optional).
        thread_id: Conversation thread identifier (optional).
        topic: Core topic of the knowledge (e.g., "phân nhóm khách hàng").
        categories: List of categories (e.g., ["health_segmentation"]).
        ttl_days: Time-to-live in days (optional, for data expiration).
        batch_size: Number of vectors to upsert in one batch.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        ns = bank_name or "default"
        target_index = ent_index

        logger.info(f"Saving to index={ent_index_name}, namespace={ns}, user_id={user_id}, thread_id={thread_id}")

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
                "raw": {"$eq": input[:2000]}
            }
        )
        if existing.get("matches", []) and existing["matches"][0]["score"] > 0.95:
            logger.info(f"Skipping near-duplicate: '{input[:50]}...' exists as {existing['matches'][0]['id']}")
            return True

        # Optimized metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "raw": input[:2000],  # Truncate to save space
            "confidence": float(confidence),
            "source": "conversation",
            "user_id": user_id,
            "thread_id": thread_id or "",
            "topic": topic or "unknown",
            "categories": categories[:5] if categories else ["general"]
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
    top_k: int = 10,
    min_similarity: float = 0.3,
    ef_search: int = 100
) -> List[Dict]:
    """
    Query knowledge from Pinecone with optimized vector structure and recall tuning.

    Args:
        query: The search query.
        bank_name: Namespace to query (optional).
        user_id: Filter by user (optional).
        thread_id: Filter by conversation thread (optional).
        topic: Filter by topic (e.g., "phân nhóm khách hàng").
        top_k: Maximum number of results.
        min_similarity: Minimum similarity score for matches.
        ef_search: Controls search-time exploration for better recall.

    Returns:
        List of knowledge entries sorted by score.
    """
    try:
        embedding = await EMBEDDINGS.aembed_query(query)
        if len(embedding) != 1536:
            logger.error(f"Invalid query embedding dimension: {len(embedding)}")
            return []

        ns = bank_name or "default"
        knowledge = []

        logger.info(f"Querying index={ent_index_name}, namespace={ns}, user_id={user_id}, thread_id={thread_id}, topic={topic}")

        # Dynamic filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if thread_id:
            filter_dict["thread_id"] = thread_id
        if topic:
            filter_dict["topic"] = topic
        
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
        #logger.info(f"Raw Knowledge vectors: {knowledge}")
        return knowledge
    except Exception as e:
        logger.error(f"Query failed after retries: {e}")
        return []

async def query_knowledge_from_graph(query: str, graph_version_id: str, user_id: Optional[str] = None, thread_id: Optional[str] = None, topic: Optional[str] = None, top_k: int = 10, min_similarity: float = 0.3, ef_search: int = 100, exclude_categories: Optional[List[str]] = None) -> List[Dict]:
    """
    Query knowledge from Pinecone with optimized vector structure and recall tuning.

    Args:
        query: The search query.
        graph_version_id: The graph version ID.
        user_id: The user ID.
        thread_id: The thread ID.
        topic: The topic.
        top_k: The top K.
        min_similarity: Minimum similarity threshold.
        ef_search: Search exploration factor.
        exclude_categories: List of categories to exclude from results.
    """
    
    #brain_banks = await get_brain_banks(graph_version_id)
    #valid_namespaces = [brain["bank_name"] for brain in brain_banks]
    # Add conversation namespace to the list
    valid_namespaces = ["conversation"]
    #logger.info(f"Valid namespaces for graph version {graph_version_id}: {valid_namespaces}")

    # Get all knowledge from all namespaces
    all_knowledge = []
    for namespace in valid_namespaces:
        knowledge = await query_knowledge(query, namespace, user_id, thread_id, topic, top_k, min_similarity, ef_search)
        if knowledge:
            all_knowledge.extend(knowledge)

    # Post-process to filter out excluded categories
    if exclude_categories:
        filtered_knowledge = []
        for entry in all_knowledge:
            # Check if this entry has any of the excluded categories
            entry_categories = entry.get("categories", [])
            if not any(cat in entry_categories for cat in exclude_categories):
                filtered_knowledge.append(entry)
        logger.info(f"Filtered out {len(all_knowledge) - len(filtered_knowledge)} entries with excluded categories: {exclude_categories}")
        all_knowledge = filtered_knowledge

    return all_knowledge

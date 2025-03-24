# [No changes needed—keeping it as the last version for reference]
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger
from datetime import datetime
import uuid

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

ami_index_name = "ami"
ent_index_name = "9well"

llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

if ent_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=ent_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
if ami_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=ami_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

ami_index = pc.Index(ami_index_name)
ent_index = pc.Index(ent_index_name)

def save_pretrain(input: str, user_id: str = "thefusionlab") -> bool:
    logger.info(f"Saving to convo history for user: {user_id}")
    namespace = f"wisdom_{user_id}"
    created_at = datetime.now().isoformat()
    embedding = EMBEDDINGS.embed_query(input)
    convo_id = f"{user_id}_{uuid.uuid4()}"
    metadata = {
        "created_at": created_at,
        "raw": input,
        "confidence": 0.8,
        "source": "preset"
    }
    try:
        upsert_result = ami_index.upsert([(convo_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved to history: {convo_id} - Result: {upsert_result}")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

def load_ami_history(input: str, user_id: str = "thefusionlab", top_k: int = 50) -> str:
    namespace = f"wisdom_{user_id}"
    query_vector = EMBEDDINGS.embed_query(input)
    results = ami_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'preset')})"
    return history if history else "Chưa có lịch sử liên quan."

def load_ami_brain(user_id: str = "thefusionlab") -> str:
    namespace = f"wisdom_{user_id}"
    dummy_vector = EMBEDDINGS.embed_query("fetch all")
    results = ami_index.query(
        vector=dummy_vector,
        top_k=3000,
        include_metadata=True,
        namespace=namespace
    )
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'preset')})"
    return history if history else "Inocent Ami."

def save_to_convo_history(input: str, user_id: str) -> bool:
    logger.info(f"Saving to convo history for user: {user_id}")
    namespace = f"wisdom_{user_id}"
    created_at = datetime.now().isoformat()
    embedding = EMBEDDINGS.embed_query(input)
    convo_id = f"{user_id}_{uuid.uuid4()}"
    metadata = {
        "created_at": created_at,
        "raw": input,
        "confidence": 0.8,
        "source": "enterprise"
    }
    try:
        upsert_result = ent_index.upsert([(convo_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved to history: {convo_id} - Result: {upsert_result}")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

def load_convo_history(input: str, user_id: str, top_k: int = 50) -> str:
    namespace = f"wisdom_{user_id}"
    query_vector = EMBEDDINGS.embed_query(input)
    results = ent_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'enterprise')})"
    return history if history else "Chưa có lịch sử liên quan."

def load_all_convo_history(user_id: str) -> str:
    namespace = f"wisdom_{user_id}"
    dummy_vector = EMBEDDINGS.embed_query("fetch all")
    results = ent_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace=namespace
    )
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'enterprise')})"
    return history if history else "Chưa có gì tui học được cả."

def blend_and_rank_history(input: str, user_id: str, top_k: int = 50) -> str:
    query_vector = EMBEDDINGS.embed_query(input)
    preset_results = ami_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace="wisdom_thefusionlab"
    )
    ent_results = ent_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=f"wisdom_{user_id}"
    )
    from datetime import datetime
    all_matches = preset_results["matches"] + ent_results["matches"]
    ranked = sorted(
        all_matches,
        key=lambda r: (
            r.score * 
            r.metadata.get("confidence", 0.5) *  # Default to 0.5 if missing
            (1 / (datetime.now() - (
                datetime.fromisoformat(r.metadata["created_at"]) 
                if "created_at" in r.metadata else datetime.now()
            )).days + 1)
        ),
        reverse=True
    )
    history = "\n".join(
        f"- {r.metadata['raw']} (from {r.metadata.get('created_at', 'unknown')}, {r.metadata.get('source', 'unknown')})"
        for r in ranked[:top_k]
    )
    return history if history else "Chưa có lịch sử liên quan."
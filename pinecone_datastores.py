# pinecone_datastores.py
# Built by: The Fusion Lab
# Date: March 24, 2025

import os
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger,LLM
from datetime import datetime
import uuid
import asyncio
import json
import re
from typing import Dict


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

ami_index_name = "dev"
ent_index_name = os.getenv("ENT")

llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

ami_index = pc.Index(ami_index_name)
ent_index = pc.Index(ent_index_name)

async def infer_categories(input: str,context : str="") -> list:
    """Infer relevant categories with bilingual labels using LLM."""
    
    category_prompt = f"""
    Conversation Context: '{context}'
    Latest Input: '{input}'
    Task: Identify relevant categories based on meaning.
    For each category, provide a label in Vietnamese (original) and its English translation.
    Examples: 'Học Hỏi' (Learning), 'Kỹ Năng Bán Hàng' (Sales Skills), 'Tính Cách' (character).
    Special Case for 'Tính Cách' (character): Include if:
    1. The input explicitly instructs an AI named Ami (e.g., 'Ami, be curious').
    2. The input implies a personality trait for an AI, especially when building on a definition in context (e.g., 'Người tò mò là...' followed by 'Luôn luôn thể hiện mình là người tò mò...').
    Do NOT tag 'character' for general personality mentions unrelated to an AI.
    If 'character' depends on prior context (e.g., a definition), add 'requires_context': true to its entry.
    Return JSON: {{"categories": [{{"original": "X", "english": "Y", "requires_context": true/false}}, ...]}}.
    Default to 'Chưa Phân Loại' (Uncategorized) if unsure.
    """
    try:
        response = await asyncio.to_thread(llm.invoke, category_prompt)  # Adjust if llm.invoke is async
        raw_content = response.content.strip()

        # Extract JSON from potential markdown blocks
        match = re.search(r"\{.*\}", raw_content, re.DOTALL | re.MULTILINE)
        if match:
            raw_content = match.group(0)

        parsed_response = json.loads(raw_content)
        categories = parsed_response.get("categories", [])

        if not isinstance(categories, list) or not categories:
            raise ValueError("Invalid categories format")

        # Validate each category
        for cat in categories:
            if not isinstance(cat, dict) or "original" not in cat or "english" not in cat:
                raise ValueError("Category missing 'original' or 'english' field")

        return categories

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(
            f"Category inference failed: {e}. Response: {response.content}. Defaulting to 'Chưa Phân Loại'."
        )
        return [{"original": "Chưa Phân Loại", "english": "Uncategorized"}]

async def save_pretrain(input: str, user_id: str = "thefusionlab", context: str = "") -> bool:
    namespace = f"wisdom_{user_id}"
    embedding = EMBEDDINGS.embed_query(input)
    categories = await infer_categories(input, context)
    categories_json = json.dumps(categories)
    has_character_with_context = any(
        cat["english"] == "character" and cat.get("requires_context", False) 
        for cat in categories
    )
    raw_content = f"{context}\nLatest: {input}" if has_character_with_context and context else input
    metadata = {
        "created_at": datetime.now().isoformat(),
        "raw": raw_content,
        "confidence": 0.8,
        "source": "preset",
        "categories": categories_json
    }
    convo_id = f"{user_id}_{uuid.uuid4()}"
    try:
        ami_index.upsert([(convo_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved to Preset Memory: {convo_id} - Categories: {categories}")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

async def save_to_convo_history(input: str, user_id: str,context: str = "") -> bool:
    """Save input to Enterprise Memory with bilingual multi-category tagging."""
    namespace = f"wisdom_{user_id}"
    embedding = EMBEDDINGS.embed_query(input)
    categories = await infer_categories(input, context)
    categories_json = json.dumps(categories)
    has_character_with_context = any(
        cat["english"] == "character" and cat.get("requires_context", False) 
        for cat in categories
    )
    raw_content = f"{context}\nLatest: {input}" if has_character_with_context and context else input
    metadata = {
        "created_at": datetime.now().isoformat(),
        "raw": raw_content,
        "confidence": 0.8,
        "source": "preset",
        "categories": categories_json
    }
    convo_id = f"{user_id}_{uuid.uuid4()}"
    try:
        ent_index.upsert([(convo_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved to Preset Memory: {convo_id} - Categories: {categories}")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

# Update load functions to deserialize categories
def load_ami_history(input: str, user_id: str = "thefusionlab", top_k: int = 50) -> str:
    namespace = f"wisdom_{user_id}"
    query_vector = EMBEDDINGS.embed_query(input)
    results = ami_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    #logger.debug(f"load_ami_history results for {user_id}, input: {input[:50]}...: {results}")
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            # Deserialize categories from JSON string
            categories = json.loads(meta.get("categories", '[{"original": "Chưa Phân Loại", "english": "Uncategorized"}]'))
            categories_str = ", ".join(f"{c['original']} ({c['english']})" for c in categories)
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'preset')}, categories: {categories_str})"
    return history if history else "Chưa có lịch sử liên quan."

def load_convo_history(input: str, user_id: str, top_k: int = 50) -> str:
    namespace = f"wisdom_{user_id}"
    query_vector = EMBEDDINGS.embed_query(input)
    results = ent_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    #logger.debug(f"load_convo_history results for {user_id}, input: {input[:50]}...: {results}")
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            # Deserialize categories from JSON string
            categories = json.loads(meta.get("categories", '[{"original": "Chưa Phân Loại", "english": "Uncategorized"}]'))
            categories_str = ", ".join(f"{c['original']} ({c['english']})" for c in categories)
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'enterprise')}, categories: {categories_str})"
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
    #logger.debug(f"load_ami_brain results for {user_id}: {results}")
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            # Deserialize categories from JSON string
            categories = json.loads(meta.get("categories", '[{"original": "Chưa Phân Loại", "english": "Uncategorized"}]'))
            categories_str = ", ".join(f"{c['original']} ({c['english']})" for c in categories)
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'preset')}, categories: {categories_str})"
    return history if history else "Innocent Smart Ami."

def load_all_convo_history(user_id: str) -> str:
    namespace = f"wisdom_{user_id}"
    dummy_vector = EMBEDDINGS.embed_query("fetch all")
    results = ent_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace=namespace
    )
    #logger.debug(f"load_all_convo_history results for {user_id}: {results}")
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            # Deserialize categories from JSON string
            categories = json.loads(meta.get("categories", '[{"original": "Chưa Phân Loại", "english": "Uncategorized"}]'))
            categories_str = ", ".join(f"{c['original']} ({c['english']})" for c in categories)
            history += f"\n- {raw} (from {timestamp}, {meta.get('source', 'enterprise')}, categories: {categories_str})"
    return history if history else "Chưa có gì tui học được cả."

def load_character_traits(user_id: str) -> str:
        namespace = f"wisdom_{user_id}"
        dummy_vector = EMBEDDINGS.embed_query("fetch all")
        results = ami_index.query(
            vector=dummy_vector, top_k=3000, include_metadata=True, namespace=namespace
        )
        traits = ""
        for r in results["matches"]:
            categories = json.loads(r.metadata.get("categories", "[]"))
            if any(cat["english"] == "character" for cat in categories) and r.metadata["confidence"] >= 0.8:
                traits += f"\n- {r.metadata['raw']} (from {r.metadata['created_at']})"
        return traits if traits else "No character traits yet."

async def blend_and_rank_brain(
    input: str,
    user_id: str = "thefusionlab",
    top_k: int = 50,
    top_n_categories: int = 2,
    weights: dict = {"score": 1, "confidence": 0.9, "recency": 1},
    boost_input: float = 3.0
) -> dict:
    """Retrieve and rank wisdom from Preset and Enterprise Memory, prioritizing character traits."""
    logger.debug(f"Querying blend_and_rank_brain with input: {input[:50]}..., user_id: {user_id}")
    query_vector = EMBEDDINGS.embed_query(input)
    user_namespace = f"wisdom_{user_id}"
    preset_namespace = "wisdom_thefusionlab"

    top_k = int(top_k)  # Ensure integer

    preset_results = ami_index.query(
        vector=query_vector, top_k=top_k, include_metadata=True, namespace=preset_namespace
    )
    ent_results = ent_index.query(
        vector=query_vector, top_k=top_k, include_metadata=True, namespace=user_namespace
    )
    all_matches = preset_results["matches"] + ent_results["matches"]

    if not all_matches:
        logger.debug("No matches found in Pinecone.")
        return {
            "categories": [],
            "wisdoms": [],
            "awareness_categories": [],
            "character_wisdom": []
        }

    ranked_wisdoms = []
    for r in all_matches:
        try:
            meta = r["metadata"]
            created_at = datetime.fromisoformat(meta["created_at"].replace("Z", "+00:00"))
            days_diff = max((datetime.now() - created_at).days + 1, 1)
            confidence = meta.get("confidence", 0.9)
            base_score = (
                (r["score"] * weights["score"]) *
                (confidence * weights["confidence"]) *
                (weights["recency"] / days_diff)
            )
            categories = json.loads(meta["categories"])
            has_character = any(cat["english"].lower() == "character" for cat in categories)
            ranked_wisdoms.append({
                "text": meta["raw"],
                "score": base_score,
                "raw_score": r["score"],
                "source": meta["source"],
                "created_at": created_at,
                "categories": categories,
                "confidence": confidence,
                "is_character": has_character
            })
        except Exception as e:
            logger.warning(f"Failed to rank wisdom {meta.get('raw', 'unknown')}: {e}")
            continue

    # Removed conflict filter: ranked_wisdoms = [w for w in ranked_wisdoms if not w["conflict"]]
    ranked_wisdoms.sort(key=lambda x: x["score"], reverse=True)
    logger.debug(f"Top wisdoms: {[w['text'][:50] + '...' for w in ranked_wisdoms[:top_k]]}")

    character_wisdom = [w for w in ranked_wisdoms if w["is_character"]][:2]
    logger.debug(f"Character wisdom: {[w['text'][:50] + '...' for w in character_wisdom]}")

    category_scores = {}
    for wisdom in ranked_wisdoms[:top_k]:
        for cat in wisdom["categories"]:
            eng_cat = cat["english"].lower()
            category_scores[eng_cat] = category_scores.get(eng_cat, 0) + wisdom["score"]

    input_categories = await infer_categories(input)
    input_cat_names = {cat["english"].lower() for cat in input_categories}
    logger.debug(f"Inferred input categories: {list(input_cat_names)}")
    for cat in input_cat_names:
        if cat in category_scores:
            category_scores[cat] *= boost_input
            logger.debug(f"Boosted {cat} to {category_scores[cat]}")
        else:
            category_scores[cat] = 0.1 * boost_input
            logger.debug(f"Added and boosted {cat} to {category_scores[cat]}")

    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    awareness_categories = sorted_categories[:top_n_categories]
    top_category_names = [cat[0] for cat in awareness_categories if cat[1] > 0]
    logger.debug(f"Selected awareness categories: {top_category_names}")

    return {
        "categories": top_category_names,
        "wisdoms": ranked_wisdoms[:3],
        "awareness_categories": awareness_categories,
        "character_wisdom": [w["text"] for w in character_wisdom]
    }

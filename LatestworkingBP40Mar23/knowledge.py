import json
import uuid
import ast
import datetime
import logging
from utilities import clean_llm_response,logger,LLM,EMBEDDINGS
from pinecone_datastores import index
import unicodedata

def sanitize_vector_id(text):
    """Convert text to ASCII, removing diacritics and replacing spaces with underscores."""
    # Normalize to decompose diacritics (e.g., 'à' → 'a' + combining char)
    normalized = unicodedata.normalize('NFKD', text)
    # Encode to ASCII, ignoring non-ASCII chars, then decode back to string
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    # Replace spaces with underscores and ensure lowercase
    return ascii_text.lower().replace(' ', '_')

def extract_knowledge(state, user_id=None, terms=None):
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    if not terms or not latest_msg.strip():
        logger.info("No terms or message, returning empty list")
        return []

    results = []
    convo_history = " | ".join(m.content for m in messages[:-1]) if len(messages) > 1 else "None"

    for term in terms:
        logger.info(f"Extracting for term: '{term}'")
        attr_prompt = (
            f"Given:\n"
            f"- Latest message: '{latest_msg}'\n"
            f"- Prior messages: '{convo_history}'\n"
            f"- Main term: '{term}'\n"
            "List descriptive properties about '{term}' as a JSON-compatible Python list of dicts. "
            "Examples:\n"
            "- 'HITO is a height booster' → [{\"key\": \"Use\", \"value\": \"height booster\"}]\n"
            "- 'GenX Fast là sản phẩm của công ty mình' → []\n"
            "- 'Nó hỗ trợ Vitamin D' → [] if term is Calcium\n"
            "- 'Nó hỗ trợ Vitamin D' → [{\"key\": \"Use\", \"value\": \"hỗ trợ Calcium\"}] if term is Vitamin D and Calcium is prior\n"
            "Rules:\n"
            "- Include features (e.g., 'Use'), origins (e.g., 'Origin'), benefits, prices (e.g., 'Price') specific to '{term}'.\n"
            "- Use prior message context to infer properties (e.g., 'nó' refers to prior term like 'Calcium').\n"
            "- Assign properties to '{term}' only if it’s the subject or object of the action in the latest message.\n"
            "- Avoid duplicating prior attributes unless restated (e.g., don’t repeat 'giúp xương chắc khỏe' from prior messages).\n"
            "- Return `[]` if nothing fits '{term}' in the latest message.\n"
            "Output ONLY the list, no 'python' or 'json' prefix."
        )
        logger.info("Before attributes extraction")
        attr_response = clean_llm_response(LLM.invoke(attr_prompt).content)
        logger.info(f"Raw attributes response: '{attr_response}'")
        try:
            attributes = json.loads(attr_response)
        except (ValueError, SyntaxError):
            logger.warning(f"JSON parse failed, defaulting to empty: {attr_response}")
            attributes = []

        rel_prompt = (
            f"Given:\n"
            f"- Latest message: '{latest_msg}'\n"
            f"- Prior messages: '{convo_history}'\n"
            f"- Main term: '{term}'\n"
            "List relationships as a JSON-compatible Python list of dicts with node IDs. "
            "Examples:\n"
            "- 'GenX Fast là sản phẩm của công ty mình' → [{\"subject\": \"GenX Fast\", \"relation\": \"Produced By\", \"object\": \"công ty mình\", \"subject_id\": \"node_genx_fast_unknown_user_{user_id}_{uuid.uuid4()}\"}]\n"
            "- 'Nó hỗ trợ Vitamin D' → [{\"subject\": \"Vitamin D\", \"relation\": \"Supports\", \"object\": \"Calcium\", \"subject_id\": \"node_vitamin_d_unknown_user_{user_id}_{uuid.uuid4()}\"}] if term is Vitamin D and Calcium is prior\n"
            "- 'Nó hỗ trợ Vitamin D' → [] if term is Calcium\n"
            "Rules:\n"
            "- Identify external entities (e.g., companies) or prior terms connected via verbs/prepositions (e.g., 'của', 'hỗ trợ') involving '{term}'.\n"
            "- Generate unique node IDs: `node_<subject>_<category>_user_{user_id}_<uuid>`.\n"
            "- Return `[]` if no clear relationships for '{term}' in the latest message.\n"
            "Output ONLY the list, no 'python' or 'json' prefix."
        )
        logger.info("Before relationships extraction")
        rel_response = clean_llm_response(LLM.invoke(rel_prompt).content)
        logger.info(f"Raw relationships response: '{rel_response}'")
        try:
            relationships = json.loads(rel_response)
        except (ValueError, SyntaxError):
            logger.warning(f"JSON parse failed, defaulting to empty: {rel_response}")
            relationships = []

        for rel in relationships:
            if "subject_id" not in rel:
                category = "Unknown"
                rel["subject_id"] = f"node_{rel['subject'].lower().replace(' ', '_')}_{category.lower()}_user_{user_id}_{uuid.uuid4()}"

        results.append({
            "term": term,
            "attributes": attributes,
            "relationships": relationships,
            "confidence": 0.9
        })

    logger.info(f"Extracted: {results}")
    return results

def save_knowledge(state, user_id, pending=None):
    if pending is None:
        pending = state.get("pending_knowledge", {})
    
    if not pending:
        logger.debug("No pending knowledge to save")
        return False
    
    logger.info(f"Entering save_knowledge for user_id: {user_id}, pending: {pending}")
    
    term_id = sanitize_vector_id(pending["term_id"])
    category = pending["category"]
    namespace = f"enterprise_knowledge_tree_{user_id}"
    parent_id = sanitize_vector_id(pending["parent_id"])
    
    vibe_score = pending["vibe_score"]
    if pending["name"] in state["active_terms"]:
        vibe_score = max(vibe_score, state["active_terms"][pending["name"]]["vibe_score"])
    logger.debug(f"Vibe score for '{pending['name']}': {vibe_score}")

    # Check existing node
    query_embedding = EMBEDDINGS.embed_query(pending["name"])
    existing = index.query(
        vector=query_embedding,
        top_k=1,
        include_metadata=True,
        namespace=namespace,
        filter={"name": pending["name"], "category": category}
    )
    logger.debug(f"Query for existing node '{pending['name']}': {existing}")
    if existing["matches"]:
        term_id = existing["matches"][0]["id"]
        logger.info(f"Found existing node: {term_id}, updating...")
    else:
        logger.info(f"Creating new node: {term_id}")

    # Root node
    root_metadata = {
        "name": category,
        "category": category,
        "vibe_score": 1.0,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    root_embedding = EMBEDDINGS.embed_query(category)
    try:
        if not index.fetch([parent_id], namespace=namespace).vectors.get(parent_id):
            index.upsert([(parent_id, root_embedding, root_metadata)], namespace=namespace)
            logger.info(f"Saved root node: {parent_id}")
    except Exception as e:
        logger.error(f"Root node upsert failed: {e}")
        return False

    # Child node
    attributes = pending["attributes"]
    relationships = pending["relationships"]
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    embedding_text = f"{pending['name']} " + " ".join([f"{a['key']}:{a['value']}" for a in attributes])
    embedding = EMBEDDINGS.embed_query(embedding_text)
    metadata = {
        "name": pending["name"],
        "category": category,
        "parent_id": parent_id,
        "attributes": json.dumps(attributes, ensure_ascii=False),
        "relationships": json.dumps(relationships, ensure_ascii=False),
        "vibe_score": vibe_score,
        "created_at": created_at
    }
    
    try:
        upsert_result = index.upsert([(term_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved child node: {term_id} - Result: {upsert_result}")
        state["active_terms"][pending["name"]] = {
            "term_id": term_id,
            "vibe_score": vibe_score,
            "attributes": attributes,
            "last_mentioned": created_at,
            "category": category
        }
    except Exception as e:
        logger.error(f"Child node upsert failed: {e}")
        return False
    
    logger.info(f"Exiting save_knowledge - Active Terms: {state['active_terms']}")
    return True

def recall_knowledge(message, state, user_id=None, fetch_all=False):
    namespace = f"enterprise_knowledge_tree_{user_id}"
    active_terms = state.get("active_terms", {})
    
    query_text = message if message != f"user_profile_{user_id}" else " ".join(active_terms.keys()) if active_terms else "default_query"
    logger.info(f"Recalling knowledge for query: '{query_text}' in namespace: {namespace}")
    
    nodes = []
    now = datetime.datetime.now(datetime.timezone.utc)
    
    if fetch_all:
        results = index.query(
            vector=EMBEDDINGS.embed_query(query_text),
            top_k=1000,
            include_metadata=True,
            namespace=namespace,
            filter={"parent_id": {"$exists": True}}
        )
    else:
        results = index.query(
            vector=EMBEDDINGS.embed_query(query_text),
            top_k=10,
            include_metadata=True,
            namespace=namespace,
            filter={"parent_id": {"$exists": True}}
        )
    logger.info(f"Query found {len(results['matches'])} matches for '{query_text}'")
    
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            attributes = json.loads(meta["attributes"]) if isinstance(meta.get("attributes"), str) else meta.get("attributes", [])
            relationships = json.loads(meta["relationships"]) if isinstance(meta.get("relationships"), str) else meta.get("relationships", [])
            meta["attributes"] = attributes
            meta["relationships"] = relationships
            days_since = (now - datetime.datetime.fromisoformat(meta["created_at"])).days
            vibe_score = meta["vibe_score"] - (0.05 * (days_since // 30))
            vibe_score = min(2.2, max(0.1, vibe_score))
            nodes.append({"id": r.id, "meta": meta, "score": r.score})
    
    if not nodes and active_terms:
        logger.info("No matches found, preserving active terms")
        nodes = [
            {"id": v["term_id"], "meta": {
                "name": k, "vibe_score": v["vibe_score"], "attributes": v["attributes"],
                "relationships": [], "created_at": v["last_mentioned"], "category": v["category"]
            }, "score": 1.0}
            for k, v in active_terms.items()
        ]
    
    if not nodes:
        preset_results = index.query(
            vector=EMBEDDINGS.embed_query("preset_profile"),
            top_k=5,
            include_metadata=True,
            namespace="preset_knowledge_tree"
        )
        nodes = [{"id": r.id, "meta": r.metadata, "score": r.score} for r in preset_results["matches"]]
        logger.info(f"Fallback to preset_knowledge_tree: {len(nodes)} nodes")
    
    nodes = sorted(nodes, key=lambda x: (x["score"], x["meta"]["created_at"]), reverse=True)
    
    knowledge = [
        {
            "name": n["meta"]["name"],
            "vibe_score": n["meta"]["vibe_score"],
            "attributes": n["meta"]["attributes"],
            "relationships": n["meta"]["relationships"],
            "term_id": n["id"],
            "last_mentioned": n["meta"]["created_at"],
            "category": n["meta"]["category"]
        }
        for n in nodes
    ]
    
    # Preserve existing active_terms, update with new ones
    new_active_terms = {f"{n['meta']['name']}_{n['id']}": {
        "term_id": n["id"],
        "vibe_score": n["meta"]["vibe_score"],
        "attributes": n["meta"]["attributes"],
        "last_mentioned": n["meta"]["created_at"],
        "category": n["meta"]["category"]
    } for n in nodes}
    state["active_terms"] = {**active_terms, **{k.split('_')[0]: v for k, v in new_active_terms.items()}}
    
    logger.info(f"Recalled active_terms: {state['active_terms']}")
    return {"knowledge": knowledge, "terms": {f"{k['name']}_{k['term_id']}": k for k in knowledge}}


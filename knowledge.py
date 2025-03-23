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
    
    logger.info(f"Entering save_knowledge for user_id: {user_id}, confirmed: True")
    
    term_id = sanitize_vector_id(pending["term_id"])
    category = pending["category"]
    namespace = f"enterprise_knowledge_{user_id}"
    parent_id = sanitize_vector_id(pending["parent_id"])
    logger.info(f"Pending knowledge: {pending}")

    vibe_score = pending["vibe_score"]
    if pending["name"] in state["active_terms"]:
        existing_vibe = state["active_terms"][pending["name"]]["vibe_score"]
        vibe_score = max(vibe_score, existing_vibe)
        logger.info(f"Preserving vibe_score from active_terms: {vibe_score} (was {pending['vibe_score']})")
    else:
        logger.info(f"Using pending vibe_score: {vibe_score}")

    query_embedding = EMBEDDINGS.embed_query(pending["name"])
    logger.info(f"Querying Pinecone for existing node: {pending['name']}")
    existing = index.query(
        vector=query_embedding,
        top_k=1,
        include_metadata=True,
        namespace=namespace,
        filter={"name": pending["name"], "category": category}
    )
    logger.info(f"Query result: {existing}")

    if existing["matches"]:
        term_id = existing["matches"][0]["id"]
        logger.info(f"Found existing node: {term_id}, updating...")
    else:
        term_id = sanitize_vector_id(pending["term_id"])
        logger.info(f"No existing node found for '{pending['name']}', creating new: {term_id}")

    root_metadata = {
        "name": category,
        "category": category,
        "vibe_score": 1.0,
        "created_at": datetime.datetime.now().isoformat()  # Corrected
    }
    root_embedding = EMBEDDINGS.embed_query(category)
    try:
        if not index.fetch([parent_id], namespace=namespace).vectors.get(parent_id):
            logger.info(f"Upserting root node: {parent_id}")
            index.upsert([(parent_id, root_embedding, root_metadata)], namespace=namespace)
            logger.info(f"Created root node: {parent_id} in {namespace}")
    except Exception as e:
        logger.error(f"Root node upsert failed: {e}")
        return False

    existing_node = index.fetch([term_id], namespace=namespace).vectors.get(term_id, None)
    if existing_node:
        old_meta = existing_node.metadata
        old_attributes = json.loads(old_meta.get("attributes", "[]"))
        old_relationships = json.loads(old_meta.get("relationships", "[]"))
        attributes = list({(a["key"], a["value"]): a for a in old_attributes + pending["attributes"]}.values())
        relationships = list({(r["subject"], r["relation"], r["object"]): r for r in old_relationships + pending["relationships"]}.values())
        created_at = old_meta["created_at"]
        logger.info(f"Merging with existing node - Attributes: {len(attributes)}, Relationships: {len(relationships)}")
    else:
        attributes = pending["attributes"]
        relationships = pending["relationships"]
        created_at = datetime.datetime.now().isoformat()  # Corrected
        logger.info("Creating new node - No existing data to merge")

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
    logger.info(f"Saving metadata: {metadata}")

    try:
        upsert_result = index.upsert([(term_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved/Updated child node: {term_id} to {namespace} - Result: {upsert_result}")
        state["active_terms"][pending["name"]] = {
            "term_id": term_id,
            "vibe_score": vibe_score,
            "attributes": pending["attributes"]
        }
    except Exception as e:
        logger.error(f"Child node upsert failed: {e}")
        return False

    convo_id = state.get("convo_id", "default_convo")
    convo_meta_id = f"convo_{convo_id}_{uuid.uuid4()}"
    convo_embedding = EMBEDDINGS.embed_query(" ".join([m.content for m in state["messages"][-3:]]))
    convo_metadata = {
        "state": json.dumps(state, default=str, ensure_ascii=False),
        "last_updated": datetime.datetime.now().isoformat()  # Corrected
    }
    try:
        index.upsert([(convo_meta_id, convo_embedding, convo_metadata)], namespace="convo_metadata")
        logger.info(f"Saved convo metadata: {convo_meta_id} to convo_metadata")
    except Exception as e:
        logger.error(f"Convo metadata upsert failed: {e}")

    logger.info(f"Exiting save_knowledge - Active Terms: {state['active_terms']}")
    return True
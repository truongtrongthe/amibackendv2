# utilities.py (Ami_Blue_Print_3_4 Mark 3.4 - AI Brain, locked March 16, 2025, Optimized March 18, 2025)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import json
import pinecone
from datetime import datetime
from pinecone_datastores import index
import uuid
import logging
from fuzzywuzzy import fuzz
import unidecode

# Setup logging (minimal)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
LLM = ChatOpenAI(model="gpt-4o", streaming=True)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

CATEGORIES = ["Skills", "Guidelines", "Lessons", "Products and Services", "Customer Personas and Behavior",
              "Objections and Responses", "Sales Scenarios and Context", "Feedback and Outcomes",
              "Ethical and Compliance Guidelines", "Industry and Market Trends", "Emotional and Psychological Insights",
              "Personalization and Customer History", "Metrics and Performance Tracking", "Team and Collaboration Dynamics",
              "Creative Workarounds", "Tools and Tech", "External Influences", "Miscellaneous"]

INTENTS = ["greeting", "question", "casual", "teaching", "request", "exit", "humor", "challenge", "confusion"]

CATEGORY_KEYWORDS = {...}  # Unchanged from your version

PRESET_KNOWLEDGE = {
    "Skills": {"text": "Đặt câu hỏi mở để hiểu khách hàng", "confidence": 0.9},
    "External Influences": {"text": "Thời tiết ảnh hưởng tâm trạng", "confidence": 0.85},
}

def sanitize_vector_id(text):
    return unidecode.unidecode(text).replace(" ", "_").lower()

def clean_llm_response(response):
    response = response.strip()
    if response.startswith("```json") and response.endswith("```"):
        return response[7:-3].strip()
    elif response.startswith("```") and response.endswith("```"):
        return response[3:-3].strip()
    while response.endswith("}"):
        response = response.rstrip("}").rstrip() + "}"
        try:
            json.loads(response)
            break
        except json.JSONDecodeError:
            continue
    return response

def initialize_preset_brain():
    for category, data in PRESET_KNOWLEDGE.items():
        vector_id = f"preset_{category.lower().replace(' ', '_')}"
        embedding = EMBEDDINGS.embed_query(data["text"])
        metadata = {"category": category, "text": data["text"], "confidence": data["confidence"], "created_at": datetime.now().isoformat()}
        try:
            index.upsert([(vector_id, embedding, metadata)], namespace="Preset")
        except Exception as e:
            logger.error(f"Preset upsert failed: {e}")

initialize_preset_brain()

def detect_intent(state):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    last_ami_msg = state.get("prompt_str", "")
    latest_msg = messages[-1].content if messages else ""
    
    prompt = f"""You’re Ami, detecting human intent. Given:
    - Latest message (70% weight): '{latest_msg}'
    - Last 3 messages (20% weight): '{convo_history}'
    - Last Ami message (10% weight): '{last_ami_msg}'
    Pick ONE intent from: {', '.join(f'"{i}"' for i in INTENTS)}. 
    Return ONLY a raw JSON object like: {{"intent": "teaching", "confidence": 0.9}}.
    Rules: Latest message drives it. Confidence <0.7? Pick 'confusion'. Output MUST be valid JSON."""
    
    response = ""
    for chunk in LLM.invoke(prompt):  # Streaming LLM
        if isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == 'content':
            response += chunk[1]  # Only append the 'content' value
        # Ignore other tuples (e.g., 'additional_kwargs', 'response_metadata')
        else:
            logger.debug(f"Ignoring non-content chunk in detect_intent: {chunk}")
    
    response = clean_llm_response(response)
    try:
        result = json.loads(response)
        intent, confidence = result["intent"], result["confidence"]
        if confidence < 0.7:
            state["prompt_str"] = f"Này, bạn đang muốn {INTENTS[3]} hay {INTENTS[2]} vậy?"
            return "confusion"
        return intent
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Intent parse error: {e}. Raw: '{response}'")
        return "casual"

def identify_topic(state, max_context=1000):
    messages = state["messages"] if state["messages"] else []
    context_size = min(max_context, len(messages))
    convo_history = " | ".join([m.content for m in messages[-context_size:]]) if messages else ""
    latest_msg = messages[-1].content if messages else ""
    
    prompt = f"""You’re Ami, identifying topics per Ami_Blue_Print_3_4 Mark 3.4. Given:
    - Latest message: '{latest_msg}'
    - Context (last {context_size} messages): '{convo_history}'
    Pick 1-3 topics from: {', '.join(CATEGORIES)}. Return raw JSON as a LIST like:
    [{{"name": "Products and Services", "confidence": 0.9}}]
    Rules: Latest message priority. Confidence <0.7? Best guess. Output MUST be a valid JSON LIST."""
    
    response = ""
    for chunk in LLM.invoke(prompt):
        if isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == 'content':
            response += chunk[1]
        else:
            logger.debug(f"Ignoring non-content chunk in identify_topic: {chunk}")
    
    response = clean_llm_response(response)
    try:
        topics = json.loads(response)
        if not isinstance(topics, list) or not all(isinstance(t, dict) and "name" in t and "confidence" in t for t in topics):
            raise ValueError("Invalid topic format")
        topics = [t for t in topics if t["name"] in CATEGORIES][:3]
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Topic parse error: {e}. Raw: '{response}'")
        return [{"name": "Miscellaneous", "confidence": 0.5}]

    if not topics or max(t["confidence"] for t in topics) < 0.7:
        state["prompt_str"] = f"Này, bạn đang nói về {CATEGORIES[3]} hay {CATEGORIES[10]} vậy?"
        return [{"name": "Miscellaneous", "confidence": 0.5}]
    return sorted(topics, key=lambda x: x["confidence"], reverse=True)

def extract_knowledge(state, user_id=None, intent=None):
    intent = intent or detect_intent(state)
    topics = identify_topic(state)
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    convo_history = " ".join(m.content for m in messages)
    convo_id = state.get("convo_id", str(uuid.uuid4()))
    
    prompt_GOOD = f"""You’re Ami, extracting for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}'
    - Convo History (last 5): '{convo_history}'
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"], "aliases": ["<variant1>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules: Extract ALL distinct noun phrases/entities/concepts. "knowledge" must be concise facts. Output MUST be valid JSON."""
    
    prompt = f"""You’re Ami, extracting for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}' (in its original language, Vietnamese - KEEP THE ORIGINAL LANGUAGE, DO NOT TRANSLATE)
    - Convo History (last 5): '{convo_history}' (in its original language, Vietnamese - KEEP THE ORIGINAL LANGUAGE, DO NOT TRANSLATE)
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"], "aliases": ["<variant1>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules: 
    - Extract ALL distinct noun phrases/entities/concepts from the input in its ORIGINAL LANGUAGE (Vietnamese), without translation or modification.
    - "knowledge" must be concise facts extracted directly from the input, in the ORIGINAL LANGUAGE (Vietnamese).
    - Output MUST be valid JSON with proper UTF-8 encoding to preserve Vietnamese characters.
    - Under NO circumstances translate any part of the input or output into another language."""

    response = ""
    for chunk in LLM.invoke(prompt):  # Streaming LLM
        if isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == 'content':
            response += chunk[1]  # Only append the 'content' value
        else:
            logger.debug(f"Ignoring non-content chunk in extract_knowledge: {chunk}")
    
    response = clean_llm_response(response)
    try:
        result = json.loads(response)
        terms = result["terms"]
        piece = result["piece"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Extract parse error: {e}. Raw: '{response}'")
        terms = {latest_msg.split()[0]: {"knowledge": [latest_msg], "aliases": []}}
        piece = {"intent": intent, "topic": topics[0], "raw_input": latest_msg}
    
    piece["piece_id"] = f"piece_{uuid.uuid4()}"
    piece["meaningfulness_score"] = 0.9
    piece["needs_clarification"] = False
    piece["term_refs"] = []
    
    for term_name, data in terms.items():
        canonical_name = term_name
        for existing_name in active_terms:
            if fuzz.ratio(term_name.lower(), existing_name.lower()) > 80:
                canonical_name = existing_name
                break
        term_id = active_terms.get(canonical_name, {}).get("term_id", f"term_{sanitize_vector_id(canonical_name)}_{convo_id}")
        active_terms[canonical_name] = {"term_id": term_id, "vibe_score": active_terms.get(canonical_name, {}).get("vibe_score", 1.0)}
        if term_id not in piece["term_refs"]:
            piece["term_refs"].append(term_id)
        state.setdefault("pending_knowledge", {}).setdefault(term_id, []).extend(
            [{"text": chunk, "confidence": 0.9, "source_piece_id": piece["piece_id"], "created_at": datetime.now().isoformat(), "aliases": data["aliases"]} 
             for chunk in data["knowledge"]]
        )
    
    if not piece["term_refs"] and "nó" in latest_msg.lower() and active_terms:
        top_term = max(active_terms.items(), key=lambda x: x[1]["vibe_score"])[1]["term_id"]
        if top_term not in piece["term_refs"]:
            piece["term_refs"].append(top_term)
    
    pending_node = state.get("pending_node", {"pieces": [], "primary_topic": topics[0]["name"]})
    pending_node["pieces"].append(piece)
    state["pending_node"] = pending_node
    state["active_terms"] = active_terms
    
    unclear_pieces = [p["raw_input"] for p in pending_node["pieces"] if p["meaningfulness_score"] < 0.8]
    if unclear_pieces:
        state["prompt_str"] = f"Ami thấy mấy ý—{', '.join(f'‘{t}’' for t in unclear_pieces)}—nói rõ cái nào đi bro!"
        for p in pending_node["pieces"]:
            if p["meaningfulness_score"] < 0.8:
                p["needs_clarification"] = True
    
    return {"terms": terms, "piece": piece}

def upsert_terms(terms):
    vectors = []
    now = datetime.now().isoformat()
    embedding_texts = []
    for term_id, knowledge in terms.items():
        term_name = term_id.split("term_")[1].rsplit("_", 1)[0] if "term_" in term_id else term_id
        term_chunks = [chunk["text"] for chunk in knowledge if isinstance(chunk, dict) and "text" in chunk]
        embedding_texts.append(term_name + " " + " ".join(term_chunks))
    embeddings = EMBEDDINGS.embed_documents(embedding_texts)  # Batch embed
    
    for i, (term_id, knowledge) in enumerate(terms.items()):
        term_name = term_id.split("term_")[1].rsplit("_", 1)[0] if "term_" in term_id else term_id
        term_chunks = [{"text": chunk["text"], "confidence": chunk.get("confidence", 0.9), 
                        "source_piece_id": chunk.get("source_piece_id", ""), "created_at": chunk.get("created_at", now), 
                        "aliases": chunk.get("aliases", [])} for chunk in knowledge if isinstance(chunk, dict) and "text" in chunk]
        vectors.append({
            "id": term_id,
            "values": embeddings[i],
            "metadata": {
                "term_name": term_name,
                "knowledge": json.dumps(term_chunks, ensure_ascii=False),
                "vibe_score": 1.0,
                "access_count": 0,
                "last_updated": now,
                "convo_id": term_id.rsplit("_", 1)[-1] if "_" in term_id else str(uuid.uuid4())
            }
        })
    
    if vectors:
        try:
            index.upsert(vectors=vectors, namespace="term_memory")
        except Exception as e:
            logger.error(f"Upsert terms failed: {e}")

def store_convo_node(node, user_id):
    vector_id = f"{node['node_id']}_{node['primary_topic']}_{node['created_at']}"
    embedding_text = " ".join(p["raw_input"] for p in node["pieces"])
    embedding = EMBEDDINGS.embed_query(embedding_text)
    metadata = {
        "node_id": node["node_id"],
        "convo_id": node["convo_id"],
        "pieces": json.dumps(node["pieces"], ensure_ascii=False),
        "primary_topic": node["primary_topic"],
        "confidence": node["confidence"],
        "confirmed_by": node["confirmed_by"],
        "last_accessed": node["last_accessed"],
        "access_count": node["access_count"],
        "created_at": node["created_at"]
    }
    try:
        index.upsert([(vector_id, embedding, metadata)], namespace="convo_nodes")
    except Exception as e:
        logger.error(f"Convo upsert failed: {e}")

def recall_knowledge(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent_result = detect_intent(state)
    intent = intent_result[0] if isinstance(intent_result, tuple) else intent_result
    
    query_embedding = EMBEDDINGS.embed_query(message)
    preset_results = index.query(vector=query_embedding, top_k=2, include_metadata=True, namespace="Preset")
    for r in preset_results["matches"]:
        if r.score > 0.8:
            return {"knowledge": [{"text": r.metadata['text'], "source": "Preset", "score": r.score}], "mode": "Co-Pilot", "intent": intent}

    convo_results = index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace="convo_nodes")
    now = datetime.now()
    nodes = []
    for r in convo_results["matches"]:
        meta = r.metadata
        meta["pieces"] = json.loads(meta["pieces"])
        meta["last_accessed"] = meta.get("last_accessed", now.isoformat())
        meta["access_count"] = meta.get("access_count", 0)
        days_since = (now - datetime.fromisoformat(meta["last_accessed"])).days
        vibe_score = (0.5 * r.score) + (0.3 * max(0, 1 - 0.05 * days_since)) + (0.2 * min(1, meta["access_count"] / 10))
        nodes.append({"meta": meta, "vibe_score": vibe_score})
    
    filtered_nodes = [n for n in nodes if n["meta"]["pieces"]] or nodes[:2]
    if not filtered_nodes:
        return {"knowledge": [], "mode": "Co-Pilot", "intent": intent}
    
    filtered_nodes.sort(key=lambda x: (x["vibe_score"], datetime.fromisoformat(x["meta"]["last_accessed"])), reverse=True)
    top_nodes = filtered_nodes[:3]
    term_ids = set(t for n in top_nodes for p in n["meta"]["pieces"] for t in p["term_refs"])
    
    fetch_response = index.fetch(list(term_ids), namespace="term_memory")
    term_nodes = getattr(fetch_response, "vectors", {})
    
    terms_to_update = {}
    for term_id in term_ids:
        if term_id in term_nodes:
            meta = term_nodes[term_id]["metadata"]
            meta["vibe_score"] += 0.1
            meta["access_count"] += 1
            meta["last_updated"] = now.isoformat()
            knowledge = json.loads(meta["knowledge"])
            terms_to_update[term_id] = [{"text": k["text"], "confidence": k.get("confidence", 0.9), 
                                        "source_piece_id": k.get("source_piece_id", ""), "created_at": k.get("created_at", now), 
                                        "aliases": k.get("aliases", [])} for k in knowledge]
    
    if terms_to_update:
        upsert_terms(terms_to_update)
    
    return {
        "knowledge": [{"meta": n["meta"], "vibe_score": n["vibe_score"]} for n in top_nodes],
        "terms": {tid: tn["metadata"] for tid, tn in term_nodes.items()},
        "mode": "Autopilot" if intent == "request" else "Co-Pilot",
        "intent": intent
    }
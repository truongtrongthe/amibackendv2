# utilities.py (Ami_Blue_Print_3_4 Mark 3.4 - AI Brain, locked March 16, 2025)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import json
import pinecone
from datetime import datetime
from pinecone_datastores import index  # Assumed Pinecone index import
import uuid
import os
import logging
import time
from fuzzywuzzy import fuzz  # Added for variant merging

import unidecode

def sanitize_vector_id(text):
    """Convert non-ASCII text to ASCII-safe string for Pinecone vector IDs."""
    return unidecode.unidecode(text).replace(" ", "_").lower()

def clean_llm_response(response):
    response = response.strip()
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()
    elif response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()
    # Strip trailing } if it’s malformed
    while response.endswith("}"):
        response = response.rstrip("}").rstrip() + "}"
        try:
            json.loads(response)
            break
        except json.JSONDecodeError:
            continue
    return response

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Config - Dynamic, no hardcodes
LLM = ChatOpenAI(model="gpt-4o", streaming=True)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

CATEGORIES = [
    "Skills", "Guidelines", "Lessons", "Products and Services", "Customer Personas and Behavior",
    "Objections and Responses", "Sales Scenarios and Context", "Feedback and Outcomes",
    "Ethical and Compliance Guidelines", "Industry and Market Trends", "Emotional and Psychological Insights",
    "Personalization and Customer History", "Metrics and Performance Tracking", "Team and Collaboration Dynamics",
    "Creative Workarounds", "Tools and Tech", "External Influences", "Miscellaneous"
]

INTENTS = ["greeting", "question", "casual", "teaching", "request", "exit", "humor", "challenge", "confusion"]

CATEGORY_KEYWORDS = {
    "Skills": {"keywords": ["kỹ năng", "mẹo", "cách", "chia sẻ"], "aliases": []},
    "Guidelines": {"keywords": ["hướng dẫn", "quy tắc", "luật"], "aliases": []},
    "Lessons": {"keywords": ["bài học", "kinh nghiệm", "học được"], "aliases": []},
    "Products and Services": {"keywords": ["sản phẩm", "có", "chứa", "glucosamine", "iphone", "galaxy", "tesla", "coffee"], "aliases": ["HITO Cốm", "HITO Com", "HITO"]},
    "Customer Personas and Behavior": {"keywords": ["khách hàng", "hành vi", "nhu cầu"], "aliases": []},
    "Objections and Responses": {"keywords": ["phản đối", "trả lời", "giải thích"], "aliases": []},
    "Sales Scenarios and Context": {"keywords": ["tình huống", "bán hàng", "kịch bản"], "aliases": []},
    "Feedback and Outcomes": {"keywords": ["phản hồi", "kết quả", "đánh giá"], "aliases": []},
    "Ethical and Compliance Guidelines": {"keywords": ["đạo đức", "tuân thủ", "quy định"], "aliases": []},
    "Industry and Market Trends": {"keywords": ["ngành", "xu hướng", "thị trường"], "aliases": []},
    "Emotional and Psychological Insights": {"keywords": ["tâm trạng", "cảm xúc", "tăng chiều cao"], "aliases": []},
    "Personalization and Customer History": {"keywords": ["cá nhân hóa", "lịch sử", "khách cũ"], "aliases": []},
    "Metrics and Performance Tracking": {"keywords": ["số liệu", "hiệu suất", "theo dõi"], "aliases": []},
    "Team and Collaboration Dynamics": {"keywords": ["đội nhóm", "hợp tác", "không khí"], "aliases": []},
    "Creative Workarounds": {"keywords": ["sáng tạo", "giải pháp", "linh hoạt"], "aliases": []},
    "Tools and Tech": {"keywords": ["crm", "công cụ", "đồng bộ"], "aliases": []},
    "External Influences": {"keywords": ["thời tiết", "môi trường", "bên ngoài"], "aliases": []},
    "Miscellaneous": {"keywords": ["khác", "tùy", "chưa rõ"], "aliases": []}
}

# Preset Knowledge - Moved to Pinecone setup function
PRESET_KNOWLEDGE = {
    "Skills": {"text": "Đặt câu hỏi mở để hiểu khách hàng", "confidence": 0.9},
    "External Influences": {"text": "Thời tiết ảnh hưởng tâm trạng", "confidence": 0.85},
}

def initialize_preset_brain():
    """Load PRESET_KNOWLEDGE into Pinecone 'Preset' namespace on startup."""
    for category, data in PRESET_KNOWLEDGE.items():
        vector_id = f"preset_{category.lower().replace(' ', '_')}"
        embedding_text = data["text"]
        embedding = EMBEDDINGS.embed_query(embedding_text)
        metadata = {
            "category": category,
            "text": data["text"],
            "confidence": data["confidence"],
            "created_at": datetime.now().isoformat()
        }
        try:
            index.upsert([(vector_id, embedding, metadata)], namespace="Preset")
            logger.info(f"Upserted preset: {vector_id}")
        except Exception as e:
            logger.error(f"Preset upsert failed: {e}")

# Run on import
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
    Rules:
    - Latest message drives it—judge its raw vibe first.
    - Context adjusts only if tied.
    - Confidence <0.7? Pick 'confusion' and flag for clarification.
    - Output MUST be valid JSON, no markdown."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    try:
        result = json.loads(response)
        intent, confidence = result["intent"], result["confidence"]
        if confidence < 0.7:
            state["prompt_str"] = f"Này, bạn đang muốn {INTENTS[3]} hay {INTENTS[2]} vậy?"
            return "confusion"
        logger.info(f"Intent for '{latest_msg}': {intent} ({confidence})")
        return intent
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Intent parse error: {e}. Raw response: '{response}'. Falling back to 'casual'")
        return "casual"

def identify_topic(state, max_context=1000):
    messages = state["messages"] if state["messages"] else []
    context_size = min(max_context, len(messages))
    convo_history = " | ".join([m.content for m in messages[-context_size:]]) if messages else ""
    latest_msg = messages[-1].content if messages else ""
    
    prompt = (
        f"You’re Ami, identifying topics per Ami_Blue_Print_3_4 Mark 3.4. Given:\n"
        f"- Latest message: '{latest_msg}'\n"
        f"- Context (last {context_size} messages): '{convo_history}'\n"
        f"Pick 1-3 topics from: {', '.join(CATEGORIES)}. Return raw JSON as a LIST like:\n"
        f'[{{"name": "Products and Services", "confidence": 0.9}}]\n'
        f"Rules:\n"
        f"- Latest message is priority—judge its vibe first.\n"
        f"- Context adds flow—check continuity.\n"
        f"- Use NER, keywords, and aliases from {json.dumps({k: v['keywords'] + v['aliases'] for k, v in CATEGORY_KEYWORDS.items()}, ensure_ascii=False)}, and vibe.\n"
        f"- Confidence: 0.9+ exact match, 0.7-0.8 likely, <0.7 stretch.\n"
        f"- Max 3 topics, primary first (highest confidence).\n"
        f"- Ambiguous (<70%)? Flag for clarification, best guess now.\n"
        f"- Output MUST be a valid JSON LIST, e.g., '[{{}}]', no extra brackets or malformed syntax.\n"
        f"- Example: 'HITO Cốm là thuốc bổ sung canxi' → '[{{'name': 'Products and Services', 'confidence': 0.9}}]'.\n"
        f"- NO markdown, just raw JSON."
    )
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.info(f"Raw LLM topic response: '{response}'")
    try:
        topics = json.loads(response)
        if not isinstance(topics, list) or not all(isinstance(t, dict) and "name" in t and "confidence" in t for t in topics):
            raise ValueError("Invalid topic format—must be a list of dicts")
        topics = [t for t in topics if t["name"] in CATEGORIES][:3]
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Topic parse error: {e}. Raw response: '{response}'. Falling back to Miscellaneous")
        return [{"name": "Miscellaneous", "confidence": 0.5}]

    if not topics or max(t["confidence"] for t in topics) < 0.7:
        state["prompt_str"] = f"Này, bạn đang nói về {CATEGORIES[3]} hay {CATEGORIES[10]} vậy?"
        return [{"name": "Miscellaneous", "confidence": 0.5}]

    topics = sorted(topics, key=lambda x: x["confidence"], reverse=True)
    logger.info(f"Parsed topics: {json.dumps(topics, ensure_ascii=False)}")
    return topics

# utilities.py
def extract_knowledge(state, user_id=None, intent=None):
    intent = intent or detect_intent(state)
    topics = identify_topic(state)
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    convo_history = " ".join(m.content for m in messages)
    convo_id = state.get("convo_id", str(uuid.uuid4()))
    
    prompt_old = f"""You’re Ami, extracting for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}'
    - Convo History (last 5): '{convo_history}'
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"], "aliases": ["<variant1>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules:
    - "terms": Extract ALL distinct noun phrases, entities, or concepts (e.g., "Aquamin F", "canxi hữu cơ", "hệ xương") via NER or context. Include both specific entities and key concepts.
    - "knowledge": Split the input into concise, standalone chunks tied to each term. Each chunk MUST be a specific fact or attribute (e.g., "32% canxi" or "dễ hấp thu"), max 1-2 short phrases. Break sentences into separate facts where possible.
    - "aliases": Leave empty unless variants detected.
    - NO translation—keep exact text from input.
    - Example: Input 'Aquamin F( 32% canxi , canxi từ tảo biển đỏ) bổ sung canxi hữu cơ' → {{"terms": {{"Aquamin F": {{"knowledge": ["32% canxi", "canxi từ tảo biển đỏ", "bổ sung canxi hữu cơ"], "aliases": []}}, "canxi hữu cơ": {{"knowledge": ["bổ sung canxi hữu cơ"], "aliases": []}}}}, "piece": ...}}
    Output MUST be valid JSON."""
    prompt = f"""You’re Ami, extracting for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}'
    - Convo History (last 5): '{convo_history}'
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"], "aliases": ["<variant1>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules:
    - "terms": Extract ALL distinct noun phrases, entities, or concepts (e.g., "khách hàng", "giá", "ROI") via NER or context.
    - "knowledge": Split the input into concise, standalone chunks tied to each term. Each chunk MUST be a specific fact, attribute, or action (e.g., "hỏi về giá", "phải nhấn mạnh"). For instructions, break into actionable pieces.
    - "aliases": Leave empty unless variants detected.
    - NO translation—keep exact text from input.
    - Example: Input 'Khi khách hàng hỏi về giá, hãy nhấn mạnh vào ROI' → {{"terms": {{"khách hàng": {{"knowledge": ["hỏi về giá"], "aliases": []}}, "giá": {{"knowledge": ["được hỏi bởi khách hàng"], "aliases": []}}, "ROI": {{"knowledge": ["phải nhấn mạnh"], "aliases": []}}}}, "piece": ...}}
    Output MUST be valid JSON."""
    
    LLM_NO_STREAM = ChatOpenAI(model="gpt-4o", streaming=False)
    try:
        response = clean_llm_response(LLM_NO_STREAM.invoke(prompt, timeout=30).content)
        logger.info(f"Raw LLM response from extract_knowledge: '{response}'")
    except Exception as e:
        logger.error(f"LLM invoke failed: {e}. Falling back")
        response = '{"terms": {"fallback": {"knowledge": ["' + latest_msg + '"], "aliases": []}}, "piece": {"intent": "' + intent + '", "topic": ' + json.dumps(topics[0]) + ', "raw_input": "' + latest_msg + '"}}'
    
    try:
        result = json.loads(response)
        terms = result["terms"]
        piece = result["piece"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Extract parse error: {e}. Raw: '{response}'. Falling back")
        terms = {latest_msg.split()[0]: {"knowledge": [latest_msg], "aliases": []}}
        piece = {"intent": intent, "topic": topics[0], "raw_input": latest_msg}
    
    # Add metadata to piece
    piece["piece_id"] = f"piece_{uuid.uuid4()}"
    piece["meaningfulness_score"] = 0.9
    piece["needs_clarification"] = False
    piece["term_refs"] = []
    
    # Process terms and update state
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
    
    # Handle "nó" reference
    if not piece["term_refs"] and "nó" in latest_msg.lower() and active_terms:
        top_term = max(active_terms.items(), key=lambda x: x[1]["vibe_score"])[1]["term_id"]
        if top_term not in piece["term_refs"]:
            piece["term_refs"].append(top_term)
    
    # Build pending_node
    pending_node = state.get("pending_node", {"pieces": [], "primary_topic": topics[0]["name"]})
    pending_node["pieces"].append(piece)
    state["pending_node"] = pending_node
    state["active_terms"] = active_terms
    
    # Check for unclear pieces
    unclear_pieces = [p["raw_input"] for p in pending_node["pieces"] if p["meaningfulness_score"] < 0.8]
    if unclear_pieces:
        state["prompt_str"] = f"Ami thấy mấy ý—{', '.join(f'‘{t}’' for t in unclear_pieces)}—nói rõ cái nào đi bro!"
        for p in pending_node["pieces"]:
            if p["meaningfulness_score"] < 0.8:
                p["needs_clarification"] = True
    
    logger.info(f"Extracted pieces: {json.dumps(pending_node['pieces'], ensure_ascii=False)}")
    
    # Return raw LLM structure with terms
    return {"terms": terms, "piece": piece}

def upsert_terms(terms):
    vectors = []
    now = datetime.now().isoformat()
    for term_id, knowledge in terms.items():
        try:
            if "term_" not in term_id:
                term_name = term_id
                term_id = f"term_{sanitize_vector_id(term_name)}_{uuid.uuid4()}"
            else:
                term_name = term_id.split("term_")[1].rsplit("_", 1)[0]
        except IndexError:
            logger.error(f"Malformed term_id: {term_id}. Rebuilding.")
            term_name = term_id
            term_id = f"term_{sanitize_vector_id(term_name)}_{uuid.uuid4()}"

        term_chunks = []
        for chunk in knowledge:
            if isinstance(chunk, str):
                # Handle raw string chunks (backward compatibility or error case)
                term_chunks.append({
                    "text": chunk,
                    "confidence": 0.9,
                    "source_piece_id": "",
                    "created_at": now,
                    "aliases": []
                })
            elif isinstance(chunk, dict) and "text" in chunk:
                # Handle expected dict format
                term_chunks.append({
                    "text": chunk["text"],
                    "confidence": chunk.get("confidence", 0.9),
                    "source_piece_id": chunk.get("source_piece_id", ""),
                    "created_at": chunk.get("created_at", now),
                    "aliases": chunk.get("aliases", [])
                })
            else:
                logger.warning(f"Unexpected chunk format: {chunk}. Skipping.")
                continue

        vector = {
            "id": term_id,
            "values": EMBEDDINGS.embed_query(term_name + " " + " ".join(c["text"] for c in term_chunks)),
            "metadata": {
                "term_name": term_name,
                "knowledge": json.dumps(term_chunks, ensure_ascii=False),
                "vibe_score": 1.0,
                "access_count": 0,
                "last_updated": now,
                "convo_id": term_id.rsplit("_", 1)[-1] if "_" in term_id else str(uuid.uuid4())
            }
        }
        vectors.append(vector)
    
    if vectors:
        index.upsert(vectors=vectors, namespace="term_memory")
        logger.info(f"Batch upserted {len(vectors)} terms: {', '.join(v['id'] for v in vectors)}")
def upsert_terms_old(terms_data, namespace="term_memory"):
    term_upserts = []
    now = datetime.now().isoformat()
    
    for term_id, new_knowledge in terms_data.items():
        #term_name_parts = term_id.split("term_")[1].split("_")
        #term_name = "_".join(term_name_parts[:-1])  # Fix: Take all parts except convo_id, e.g., "canxi_huu_co"
        term_name = term_id.split("term_")[1].rsplit("_", 1)[0]
        
        fetch_response = index.fetch([term_id], namespace=namespace)
        existing_node = fetch_response.vectors.get(term_id) if fetch_response.vectors else None
        
        aliases = list(set(sum([k["aliases"] for k in new_knowledge if "aliases" in k], [])))
        if existing_node:
            metadata = existing_node["metadata"]
            combined_knowledge = json.loads(metadata["knowledge"]) + new_knowledge
            vibe_score = metadata["vibe_score"]
            aliases = list(set(json.loads(metadata.get("aliases", "[]")) + aliases))
            access_count = metadata["access_count"]
            created_at = metadata["created_at"]
        else:
            combined_knowledge = new_knowledge
            vibe_score = 1.0
            aliases = aliases or []
            access_count = 0
            created_at = now
        
        metadata = {
            "term_id": term_id,
            "term_name": term_name,
            "knowledge": json.dumps(combined_knowledge, ensure_ascii=False),  # Fix: Keep all chunks
            "aliases": json.dumps(aliases, ensure_ascii=False),
            "vibe_score": vibe_score,
            "last_updated": now,
            "access_count": access_count,
            "created_at": created_at
        }
        
        embedding_text = f"{term_name} {' '.join(k['text'] for k in combined_knowledge)}"
        embedding = EMBEDDINGS.embed_query(embedding_text)
        term_upserts.append((term_id, embedding, metadata))
    
    if term_upserts:
        try:
            index.upsert(term_upserts, namespace=namespace)
            logger.info(f"Batch upserted {len(term_upserts)} terms: {', '.join(t[0] for t in term_upserts)}")
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}. Terms: {len(term_upserts)}")

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
        logger.info(f"Stored convo node: {vector_id}")
    except Exception as e:
        logger.error(f"Convo upsert failed: {e}")
        return {"response": f"Ami đây! Chưa rõ lắm, bro nói thêm đi!", "mode": "Co-Pilot", "source": "Enterprise"}


# utilities.py
def recall_knowledge(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent_result = detect_intent(state)
    intent = intent_result[0] if isinstance(intent_result, tuple) else intent_result
    
    preset_results = index.query(vector=EMBEDDINGS.embed_query(message), top_k=2, include_metadata=True, namespace="Preset")
    for r in preset_results["matches"]:
        if r.score > 0.8:
            return {"knowledge": [{"text": r.metadata['text'], "source": "Preset", "score": r.score}], "mode": "Co-Pilot", "intent": intent}

    query_embedding = EMBEDDINGS.embed_query(message)
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
    
    # Batch update terms
    terms_to_update = {}
    for term_id in term_ids:
        if term_id in term_nodes:
            meta = term_nodes[term_id]["metadata"]
            meta["vibe_score"] += 0.1
            meta["access_count"] += 1
            meta["last_updated"] = now.isoformat()
            knowledge = json.loads(meta["knowledge"])
            terms_to_update[term_id] = [{"text": k["text"], "confidence": k.get("confidence", 0.9), "source_piece_id": k.get("source_piece_id", ""), "created_at": k.get("created_at", now), "aliases": k.get("aliases", [])} for k in knowledge]
    
    if terms_to_update:
        upsert_terms(terms_to_update)
    
    return {
        "knowledge": [{"meta": n["meta"], "vibe_score": n["vibe_score"]} for n in top_nodes],
        "terms": {tid: tn["metadata"] for tid, tn in term_nodes.items()},
        "mode": "Autopilot" if intent == "request" else "Co-Pilot",
        "intent": intent
    }
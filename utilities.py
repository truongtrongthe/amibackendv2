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
logging.basicConfig(level=logging.INFO)
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

def extract_knowledge(state, user_id=None, intent =None):
    intent = intent or detect_intent(state) 
    topics = identify_topic(state)
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    convo_history = " ".join(m.content for m in messages)
    convo_id = state.get("convo_id", str(uuid.uuid4()))
    
    
    prompt = f"""You’re Ami, extracting for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}'
    - Convo History (last 5): '{convo_history}'
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"], "aliases": ["<variant1>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules:
    - "terms": Extract ALL full noun phrases, roles, or titles (e.g., "HITO Cốm", "bác sĩ dinh dưỡng") via NER or context. Prioritize specific entities over generics.
    - "knowledge": Tie chunks to terms—e.g., "giới thiệu mình là bác sĩ dinh dưỡng" for "bác sĩ dinh dưỡng".
    - "aliases": Leave empty unless variants detected.
    - NO translation—keep it exact.
    - Output MUST be valid JSON."""
    # Debug: Disable streaming, add timeout
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
    
    logger.info(f"Extracted pieces: {json.dumps(pending_node['pieces'], ensure_ascii=False)}")
    return pending_node



def upsert_term_node(term_id, convo_id, new_knowledge):
    term_name = term_id.split("term_")[1].split(f"_{convo_id}")[0]
    vector_id = term_id  # Already term_<name>_<convo_id>
    existing_node = index.fetch([vector_id], namespace="term_memory").get("vectors", {}).get(vector_id, None)
    
    aliases = []
    if new_knowledge and "aliases" in new_knowledge[0]:
        aliases = list(set(sum([k["aliases"] for k in new_knowledge], [])))  # Flatten and dedupe aliases
    
    if existing_node:
        metadata = existing_node["metadata"]
        knowledge = json.loads(metadata["knowledge"]) + new_knowledge
        embedding_text = f"{term_name} {' '.join(k['text'] for k in knowledge)}"
        vibe_score = metadata["vibe_score"]
        aliases = list(set(json.loads(metadata.get("aliases", "[]")) + aliases))  # Merge with existing
    else:
        knowledge = new_knowledge
        embedding_text = f"{term_name} {' '.join(k['text'] for k in knowledge)}"
        vibe_score = 1.0  # 3.4: Start at 1.0
        aliases = aliases or []
    
    metadata = {
        "term_id": term_id,
        "term_name": term_name,
        "knowledge": json.dumps([k for k in knowledge if "aliases" not in k], ensure_ascii=False),  # Strip aliases from knowledge
        "aliases": json.dumps(aliases, ensure_ascii=False),
        "vibe_score": vibe_score,
        "last_updated": datetime.now().isoformat(),
        "access_count": existing_node["metadata"]["access_count"] if existing_node else 0,
        "created_at": existing_node["metadata"]["created_at"] if existing_node else datetime.now().isoformat()
    }
    embedding = EMBEDDINGS.embed_query(embedding_text)
    try:
        index.upsert([(vector_id, embedding, metadata)], namespace="term_memory")
        logger.info(f"Upserted term node: {vector_id}")
    except Exception as e:
        logger.error(f"Term upsert failed: {e}")

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

def recall_knowledge_runOK(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent = detect_intent(state)
    
    message_lower = message.lower()
    preset_results = index.query(vector=EMBEDDINGS.embed_query(message), top_k=2, include_metadata=True, namespace="Preset")
    for r in preset_results["matches"]:
        if r.score > 0.8:
            return {"response": f"Ami đây! {r.metadata['text']}—thử không bro?", "mode": "Co-Pilot", "source": "Preset"}

    query_embedding = EMBEDDINGS.embed_query(message)
    convo_results = index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace="convo_nodes")
    logger.info(f"Recall convo_results: {json.dumps(convo_results, default=str)}")
    
    now = datetime.now()
    nodes = []
    for r in convo_results["matches"]:
        meta = r.metadata
        meta["pieces"] = json.loads(meta["pieces"])
        meta["last_accessed"] = meta.get("last_accessed", now.isoformat())
        meta["access_count"] = meta.get("access_count", 0)
        days_since = (now - datetime.fromisoformat(meta["last_accessed"])).days
        relevance = r.score
        recency = max(0, 1 - 0.05 * days_since)
        usage = min(1, meta["access_count"] / 10)
        vibe_score = (0.5 * relevance) + (0.3 * recency) + (0.2 * usage)
        nodes.append({"meta": meta, "vibe_score": vibe_score})
    
    filtered_nodes = [n for n in nodes if n["meta"]["pieces"]] or nodes[:2]
    if not filtered_nodes:
        return {"response": f"Ami đây! Chưa đủ info, bro thêm tí nha!", "mode": "Co-Pilot", "source": "Enterprise"}
    
    filtered_nodes.sort(key=lambda x: (x["vibe_score"], datetime.fromisoformat(x["meta"]["last_accessed"])), reverse=True)
    top_nodes = filtered_nodes[:3]
    term_ids = set(t for n in top_nodes for p in n["meta"]["pieces"] for t in p["term_refs"])
    
    fetch_response = index.fetch(list(term_ids), namespace="term_memory")
    term_nodes = getattr(fetch_response, "vectors", {})
    for term_id in term_ids:
        if term_id in term_nodes:
            meta = term_nodes[term_id]["metadata"]
            meta["vibe_score"] += 0.1
            meta["access_count"] += 1
            meta["last_updated"] = now.isoformat()
            embedding_text = f"{meta['term_name']} {' '.join(k['text'] for k in json.loads(meta['knowledge']))}"
            embedding = EMBEDDINGS.embed_query(embedding_text)
            index.upsert([(term_id, embedding, meta)], namespace="term_memory")
            time.sleep(1)
    
    prompt = f"""You’re Ami, pitching for AI Brain Mark 3.4. Given:
    - Input: '{message}'
    - Intent: '{intent}'
    - Convo Nodes: {json.dumps([n['meta'] for n in top_nodes], ensure_ascii=False)}
    - Term Nodes: {json.dumps({tid: tn['metadata'] for tid, tn in term_nodes.items()}, ensure_ascii=False)}
    Return raw JSON: {{"response": "<response>", "mode": "<mode>", "source": "Enterprise"}}
    Rules:
    - "response": Vietnamese, casual, sales-y—use all three top_nodes for fullest context, check aliases. For 'request', prioritize sales steps with specifics (e.g., "#combo 1"). For 'question', blend all top_nodes; include all key details from highest-scoring node (e.g., all benefits), use exact key phrases where possible (e.g., "ổn định hấp thụ xương"), and make sales hooks explicit (e.g., "mua cho con"). If asking about a component (e.g., "Aquamin F"), link it to "HITO Cốm" explicitly. Predict objections (e.g., age, cost) and address them.
    - "mode": "Autopilot" if "request", else "Co-Pilot".
    - "source": "Enterprise".
    - Short, actionable, charming—use highest vibe_score terms first.
    - Example: "Tôi 35 tuổi, HITO Cốm có tác dụng không?" → {{"response": "Bro 35 tuổi thì HITO Cốm giúp ổn định hấp thụ xương, không tăng chiều cao nhiều—mua cho con nhỏ thì đỉnh hơn!", "mode": "Co-Pilot", "source": "Enterprise"}}
    - Output MUST be valid JSON, no markdown."""

    response = clean_llm_response(LLM.invoke(prompt).content)
    try:
        result = json.loads(response)
        logger.info(f"Recalled: {json.dumps(result, ensure_ascii=False)}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Recall parse error: {e}. Raw: '{response}'. Falling back")
        return {"response": f"Ami đây! Chưa rõ lắm, bro nói thêm đi!", "mode": "Co-Pilot", "source": "Enterprise"}
# utilities.py (snippet)
# utilities.py (snippet)
def recall_knowledge(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    # Safely handle detect_intent output—string or tuple
    intent_result = detect_intent(state)
    if isinstance(intent_result, tuple):
        intent, _ = intent_result  # Unpack if tuple (e.g., "request", 0.9)
    else:
        intent = intent_result  # Use as-is if string (e.g., "question")
    
    message_lower = message.lower()
    preset_results = index.query(vector=EMBEDDINGS.embed_query(message), top_k=2, include_metadata=True, namespace="Preset")
    for r in preset_results["matches"]:
        if r.score > 0.8:
            return {
                "knowledge": [{"text": r.metadata['text'], "source": "Preset", "score": r.score}],
                "mode": "Co-Pilot",
                "intent": intent
            }

    query_embedding = EMBEDDINGS.embed_query(message)
    convo_results = index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace="convo_nodes")
    logger.info(f"Recall convo_results: {json.dumps(convo_results, default=str)}")
    
    now = datetime.now()
    nodes = []
    for r in convo_results["matches"]:
        meta = r.metadata
        meta["pieces"] = json.loads(meta["pieces"])
        meta["last_accessed"] = meta.get("last_accessed", now.isoformat())
        meta["access_count"] = meta.get("access_count", 0)
        days_since = (now - datetime.fromisoformat(meta["last_accessed"])).days
        relevance = r.score
        recency = max(0, 1 - 0.05 * days_since)
        usage = min(1, meta["access_count"] / 10)
        vibe_score = (0.5 * relevance) + (0.3 * recency) + (0.2 * usage)
        nodes.append({"meta": meta, "vibe_score": vibe_score})
    
    filtered_nodes = [n for n in nodes if n["meta"]["pieces"]] or nodes[:2]
    if not filtered_nodes:
        return {"knowledge": [], "mode": "Co-Pilot", "intent": intent}
    
    filtered_nodes.sort(key=lambda x: (x["vibe_score"], datetime.fromisoformat(x["meta"]["last_accessed"])), reverse=True)
    top_nodes = filtered_nodes[:3]
    term_ids = set(t for n in top_nodes for p in n["meta"]["pieces"] for t in p["term_refs"])
    
    fetch_response = index.fetch(list(term_ids), namespace="term_memory")
    term_nodes = getattr(fetch_response, "vectors", {})
    for term_id in term_ids:
        if term_id in term_nodes:
            meta = term_nodes[term_id]["metadata"]
            meta["vibe_score"] += 0.1
            meta["access_count"] += 1
            meta["last_updated"] = now.isoformat()
            embedding_text = f"{meta['term_name']} {' '.join(k['text'] for k in json.loads(meta['knowledge']))}"
            embedding = EMBEDDINGS.embed_query(embedding_text)
            index.upsert([(term_id, embedding, meta)], namespace="term_memory")
            time.sleep(1)
    
    return {
        "knowledge": [{"meta": n["meta"], "vibe_score": n["vibe_score"]} for n in top_nodes],
        "terms": {tid: tn["metadata"] for tid, tn in term_nodes.items()},
        "mode": "Autopilot" if intent == "request" else "Co-Pilot",
        "intent": intent
    }
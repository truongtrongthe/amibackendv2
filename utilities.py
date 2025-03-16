# utilities.py (Ami_Blue_Print_3_3 Mark 3 - AI Brain, locked March 15, 2025)
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
    "Skills": ["kỹ năng", "mẹo", "cách", "chia sẻ"],
    "Guidelines": ["hướng dẫn", "quy tắc", "luật"],
    "Lessons": ["bài học", "kinh nghiệm", "học được"],
    "Products and Services": ["sản phẩm", "có", "chứa", "glucosamine", "iphone", "galaxy", "tesla", "coffee"],
    "Customer Personas and Behavior": ["khách hàng", "hành vi", "nhu cầu"],
    "Objections and Responses": ["phản đối", "trả lời", "giải thích"],
    "Sales Scenarios and Context": ["tình huống", "bán hàng", "kịch bản"],
    "Feedback and Outcomes": ["phản hồi", "kết quả", "đánh giá"],
    "Ethical and Compliance Guidelines": ["đạo đức", "tuân thủ", "quy định"],
    "Industry and Market Trends": ["ngành", "xu hướng", "thị trường"],
    "Emotional and Psychological Insights": ["tâm trạng", "cảm xúc", "tăng chiều cao"],
    "Personalization and Customer History": ["cá nhân hóa", "lịch sử", "khách cũ"],
    "Metrics and Performance Tracking": ["số liệu", "hiệu suất", "theo dõi"],
    "Team and Collaboration Dynamics": ["đội nhóm", "hợp tác", "không khí"],
    "Creative Workarounds": ["sáng tạo", "giải pháp", "linh hoạt"],
    "Tools and Tech": ["crm", "công cụ", "đồng bộ"],
    "External Influences": ["thời tiết", "môi trường", "bên ngoài"],
    "Miscellaneous": ["khác", "tùy", "chưa rõ"]
}

PRESET_KNOWLEDGE = {
    "Skills": {"text": "Đặt câu hỏi mở để hiểu khách hàng", "confidence": 0.9},
    "External Influences": {"text": "Thời tiết ảnh hưởng tâm trạng", "confidence": 0.85},
}

# Helper to strip markdown
def clean_llm_response(response):
    response = response.strip()
    if response.startswith("```json") and response.endswith("```"):
        return response[7:-3].strip()
    elif response.startswith("```") and response.endswith("```"):
        return response[3:-3].strip()
    return response

# Detect Intent
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
        f"You’re Ami, identifying topics per Ami_Blue_Print_3_3 Mark 3. Given:\n"
        f"- Latest message: '{latest_msg}'\n"
        f"- Context (last {context_size} messages): '{convo_history}'\n"
        f"Pick 1-3 topics from: {', '.join(CATEGORIES)}. Return raw JSON as a LIST like:\n"
        f'[{{"name": "Products and Services", "confidence": 0.9}}]\n'
        f"Rules:\n"
        f"- Latest message is priority—judge its vibe first.\n"
        f"- Context adds flow—check continuity.\n"
        f"- Use NER, keywords from {json.dumps(CATEGORY_KEYWORDS, ensure_ascii=False)}, and vibe.\n"
        f"- Confidence: 0.9+ exact match, 0.7-0.8 likely, <0.7 stretch.\n"
        f"- Max 3 topics, primary first (highest confidence).\n"
        f"- Ambiguous (<70%)? Flag for clarification, best guess now.\n"
        f"- Output MUST be a valid JSON LIST, e.g., '[{{}}]', no extra brackets or malformed syntax.\n"
        f"- Example: 'HITO là thuốc bổ sung canxi' → '[{{'name': 'Products and Services', 'confidence': 0.9}}]'.\n"
        f"- NO markdown, just raw JSON."
    )
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.info(f"Raw LLM topic response: '{response}'")  # Debug raw output
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

# Extract Knowledge (Training Path)
def extract_knowledge(state, user_id=None):
    intent = detect_intent(state)
    topics = identify_topic(state)
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    convo_history = " ".join(m.content for m in messages)
    convo_id = state.get("convo_id", str(uuid.uuid4()))
    
    prompt = f"""You’re Ami, extracting for AI Brain Mark 3. Given:
    - Latest: '{latest_msg}'
    - Convo History (last 5): '{convo_history}'
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules:
    - "terms": Extract ALL full noun phrases or product names (e.g., "Bột xương cá tuyết", "HITO") via NER or context—grab every distinct entity you spot.
    - "knowledge": Chunk the input into full, meaningful phrases tied to each term based on context—don’t force a pattern, just split naturally. "Nó"/"it" refers to the last active term.
    - "piece": Single object with intent, primary topic, raw_input.
    - NO translation—keep input language EXACTLY as provided (e.g., Vietnamese stays Vietnamese).
    - Example: "HITO giúp xương chắc khỏe" → {{"terms": {{"HITO": {{"knowledge": ["giúp xương chắc khỏe"]}}}}, "piece": ...}}
    - Output MUST be valid JSON, no markdown."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.info(f"Raw LLM response from extract_knowledge: '{response}'")
    try:
        result = json.loads(response)
        terms = result["terms"]
        piece = result["piece"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Extract parse error: {e}. Raw: '{response}'. Falling back")
        terms = {latest_msg.split()[0]: {"knowledge": [latest_msg]}}  # Fallback: first word as term
        piece = {"intent": intent, "topic": topics[0], "raw_input": latest_msg}
    
    piece["piece_id"] = f"piece_{uuid.uuid4()}"
    piece["meaningfulness_score"] = 0.9
    piece["needs_clarification"] = False
    piece["term_refs"] = []
    
    for term_name, data in terms.items():
        term_id = active_terms.get(term_name, f"term_{uuid.uuid4()}")
        active_terms[term_name] = term_id
        piece["term_refs"].append(term_id)  # Link all terms in piece
        state.setdefault("pending_knowledge", {}).setdefault(term_id, []).extend(
            [{"text": chunk, "confidence": 0.9, "source_piece_id": piece["piece_id"], "created_at": datetime.now().isoformat()} 
             for chunk in data["knowledge"]]
        )
    
    if not piece["term_refs"] and "nó" in latest_msg.lower() and active_terms:
        last_term_id = list(active_terms.values())[-1]
        piece["term_refs"].append(last_term_id)
    
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


def confirm_knowledge(state, user_id, confirm_callback=None):
    node = state.get("pending_node", {"pieces": [], "primary_topic": "Miscellaneous"})
    if not node["pieces"]:
        return None
    
    confirm_callback = confirm_callback or (lambda x: input(x + " (yes/no): ").lower())
    for piece in node["pieces"]:
        if piece["needs_clarification"]:
            state["prompt_str"] = f"Ami hiểu là {piece['raw_input']}—đúng không?"
            response = confirm_callback(state["prompt_str"])
            state["last_response"] = response
            if response == "yes":
                piece["needs_clarification"] = False
                piece["meaningfulness_score"] = max(piece["meaningfulness_score"], 0.8)
            elif response == "no":
                piece["meaningfulness_score"] = 0.5
    
    state["prompt_str"] = "Ami lưu cả mớ này nhé?"
    response = confirm_callback(state["prompt_str"])
    state["last_response"] = response
    if response == "yes":
        node["node_id"] = f"node_{uuid.uuid4()}"
        node["convo_id"] = state.get("convo_id", str(uuid.uuid4()))
        node["confidence"] = sum(p["meaningfulness_score"] for p in node["pieces"]) / len(node["pieces"])
        node["created_at"] = datetime.now().isoformat()
        node["last_accessed"] = node["created_at"]
        node["access_count"] = 0
        node["confirmed_by"] = user_id or "user123"
        node["primary_topic"] = node["pieces"][0]["topic"]["name"]  # Set from first piece’s topic
        
        pending_knowledge = state.get("pending_knowledge", {})
        for term_id, knowledge in pending_knowledge.items():
            upsert_term_node(term_id, state["convo_id"], knowledge)
            time.sleep(5)
        
        store_convo_node(node, user_id)
        time.sleep(5)
        state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
        state.pop("pending_knowledge", None)
    elif response == "no":
        state["prompt_str"] = f"OK, Ami bỏ qua. Còn gì thêm cho {node['primary_topic']} không?"
        state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
    return node

# Upsert Term Node (Enterprise Brain)
def upsert_term_node(term_id, convo_id, new_knowledge):
    term_name = term_id.split("term_")[1].split(f"_{convo_id}")[0]
    vector_id = term_id
    existing_node = index.fetch([vector_id], namespace="term_memory").get("vectors", {}).get(vector_id, None)
    
    if existing_node:
        metadata = existing_node["metadata"]
        knowledge = json.loads(metadata["knowledge"]) + new_knowledge
        embedding_text = f"{term_name} {' '.join(k['text'] for k in knowledge)}"
        vibe_score = metadata["vibe_score"]
    else:
        knowledge = new_knowledge
        embedding_text = f"{term_name} {' '.join(k['text'] for k in knowledge)}"
        vibe_score = 1.0  # Blueprint: Start at 1.0
    
    metadata = {
        "term_id": term_id,
        "term_name": term_name,
        "knowledge": json.dumps(knowledge, ensure_ascii=False),
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

# Store Convo Node (Enterprise Brain)
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

# Recall Knowledge (Selling Path)
def recall_knowledge(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent = detect_intent(state)
    
    message_lower = message.lower()
    for category, preset in PRESET_KNOWLEDGE.items():
        if category in message or any(kw in message_lower for kw in CATEGORY_KEYWORDS.get(category, [])):
            return {"response": f"Ami đây! {preset['text']}—thử không bro?", "mode": "Co-Pilot", "source": "Preset"}

    query_embedding = EMBEDDINGS.embed_query(message)
    convo_results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace="convo_nodes")
    logger.info(f"Recall convo_results: {json.dumps(convo_results, default=str)}")  # Debug Pinecone hits
    
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
        vibe_score = (0.3 * relevance) + (0.5 * recency) + (0.2 * usage)
        nodes.append({"meta": meta, "vibe_score": vibe_score})
    
    # Loosen filter—any nodes with pieces, not just term_refs
    filtered_nodes = [n for n in nodes if n["meta"]["pieces"]]
    if not filtered_nodes and nodes:
        filtered_nodes = nodes[:2]  # Fallback to top vibe_score
    
    if not filtered_nodes:
        return {"response": f"Ami đây! Chưa đủ info, bro thêm tí nha!", "mode": "Co-Pilot", "source": "Enterprise"}
    
    filtered_nodes.sort(key=lambda x: (datetime.fromisoformat(x["meta"]["last_accessed"]), x["vibe_score"]), reverse=True)
    top_nodes = filtered_nodes[:2]
    term_ids = set(t for n in top_nodes for p in n["meta"]["pieces"] for t in p["term_refs"])
    
    # Update Term Vibe Scores
    term_nodes = index.fetch(list(term_ids), namespace="term_memory").get("vectors", {})
    for term_id in term_ids:
        if term_id in term_nodes:
            meta = term_nodes[term_id]["metadata"]
            meta["vibe_score"] += 0.1  # Blueprint: +0.1 per recall
            meta["access_count"] += 1
            meta["last_updated"] = now.isoformat()
            embedding_text = f"{meta['term_name']} {' '.join(k['text'] for k in json.loads(meta['knowledge']))}"
            embedding = EMBEDDINGS.embed_query(embedding_text)
            index.upsert([(term_id, embedding, meta)], namespace="term_memory")
            time.sleep(2)  # Delay for Pinecone indexing
    
    # Pitch Response
    prompt = f"""You’re Ami, pitching for AI Brain Mark 3. Given:
    - Input: '{message}'
    - Intent: '{intent}'
    - Convo Nodes: {json.dumps([n['meta'] for n in top_nodes], ensure_ascii=False)}
    - Term Nodes: {json.dumps({tid: tn['metadata'] for tid, tn in term_nodes.items()}, ensure_ascii=False)}
    Return raw JSON: {{"response": "<response>", "mode": "<mode>", "source": "Enterprise"}}
    Rules:
    - "response": Vietnamese, casual, sales-y—use convo + term data.
    - "mode": "Autopilot" if "request", else "Co-Pilot".
    - "source": "Enterprise".
    - Short, actionable, charming.
    - Predict objections if intent fits.
    - Output MUST be valid JSON, no markdown."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    try:
        result = json.loads(response)
        logger.info(f"Recalled: {json.dumps(result, ensure_ascii=False)}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Recall parse error: {e}. Raw: '{response}'. Falling back")
        return {"response": f"Ami đây! Chưa rõ lắm, bro nói thêm đi!", "mode": "Co-Pilot", "source": "Enterprise"}

# Test Suite
def test_ami():
    logger.info("Testing Ami—AI Brain Mark 3 locked and loaded!")
    state = {"messages": [], "prompt_str": "", "convo_id": str(uuid.uuid4()), "active_terms": {}, "pending_knowledge": {}}
    
    # Test Training Path
    state["messages"].append(HumanMessage("Apple iPhone helps with tech vibes"))
    extract_knowledge(state)
    confirm_knowledge(state, "user123", lambda x: "yes")  # Simulate confirmation
    
    state["messages"].append(HumanMessage("Nó có chip A17 và camera xịn"))
    extract_knowledge(state)
    confirm_knowledge(state, "user123", lambda x: "yes")
    
    # Test Selling Path
    result = recall_knowledge("Apple iPhone có gì hay?", "user123")
    logger.info(f"Recall result: {json.dumps(result, ensure_ascii=False)}")
    
    # Test Preset Fallback
    result = recall_knowledge("Thời tiết hôm nay thế nào?", "user123")
    logger.info(f"Preset recall: {json.dumps(result, ensure_ascii=False)}")

if __name__ == "__main__":
    test_ami()
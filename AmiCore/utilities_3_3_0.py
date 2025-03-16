# utilities.py (Ami_Blue_Print_3_3, locked as of March 15, 2025)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import json
import pinecone
from datetime import datetime
from pinecone_datastores import index  # Assumed Pinecone index import
import uuid
import os
import logging
import time  # For delay

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

# Global category keywords
CATEGORY_KEYWORDS = {
    "Skills": ["kỹ năng", "mẹo", "cách", "chia sẻ"],
    "Guidelines": ["hướng dẫn", "quy tắc", "luật"],
    "Lessons": ["bài học", "kinh nghiệm", "học được"],
    "Products and Services": ["HITO", "sản phẩm", "có", "chứa", "glucosamine"],
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
    "Tools and Tech": ["CRM", "công cụ", "đồng bộ"],
    "External Influences": ["thời tiết", "môi trường", "bên ngoài"],
    "Miscellaneous": ["khác", "tùy", "chưa rõ"]
}

# Preset Brain (Mock preloaded data for fallback)
PRESET_KNOWLEDGE = {
    "Skills": {"text": "Đặt câu hỏi mở để hiểu khách hàng", "confidence": 0.9},
    "External Influences": {"text": "Thời tiết ảnh hưởng tâm trạng", "confidence": 0.85},
    # Add more as needed
}

# Helper to strip markdown from LLM responses
def clean_llm_response(response):
    response = response.strip()
    if response.startswith("```json") and response.endswith("```"):
        return response[7:-3].strip()
    elif response.startswith("```") and response.endswith("```"):
        return response[3:-3].strip()
    return response

# Core Intent Detection
def detect_intent(state):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    last_ami_msg = state.get("prompt_str", "")
    latest_msg = messages[-1].content if messages else ""
    
    prompt = f"""You’re Ami, a sharp AI detecting human intent. Given:
    - Latest message: '{latest_msg}'
    - Last 3 messages: '{convo_history}'
    - Last Ami message: '{last_ami_msg}'
    Pick ONE intent from: {', '.join(f'"{i}"' for i in INTENTS)}. 
    Return ONLY a raw JSON object like: {{"intent": "teaching", "confidence": 0.9}}.
    Rules:
    - Latest message is the boss—judge its raw vibe first (e.g., imperative verb = "request", no verb + fact = "teaching").
    - Use last 3 messages and Ami’s reply for context—don’t override latest unless tied.
    - No assumptions—stick to text, decode any language or emojis.
    - Confidence <70%? Pick "confusion" and flag for clarification.
    - Output MUST be valid JSON, no markdown (e.g., no ```json```), no extra text."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.debug(f"Raw intent response: '{response}'")
    
    try:
        result = json.loads(response)
        intent, confidence = result["intent"], result["confidence"]
        if confidence < 0.7:
            state["prompt_str"] = "Này, bạn đang chia sẻ hay tán gẫu vậy?"
            logger.debug(f"Intent confidence {confidence} < 0.7, forcing 'confusion'")
            return "confusion"
        logger.info(f"Intent for '{latest_msg}': {intent} ({confidence})")
        return intent.strip('"')
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Intent parse error: {e}. Raw response: '{response}'. Falling back to 'casual'")
        return "casual"

# Topic Identification
def identify_topic(state, max_context=1000):
    messages = state["messages"] if state["messages"] else []
    context_size = min(max_context, len(messages))
    convo_history = " | ".join([m.content for m in messages[-context_size:]]) if messages else ""
    latest_msg = messages[-1].content if messages else ""
    
    prompt = (
        f"You’re Ami, identifying topics from text per Ami_Blue_Print_3_3. Given:\n"
        f"- Latest message: '{latest_msg}'\n"
        f"- Context (last {context_size} messages): '{convo_history}'\n"
        f"Pick 1-3 topics from 18 categories: {', '.join(CATEGORIES)}. Return raw JSON like:\n"
        f'[{{"name": "Products and Services", "confidence": 0.9}}]\n'
        f"Rules:\n"
        f"Latest message is the boss—judge its vibe first.\n"
        f"Context (up to {context_size} messages) adds flow—check for topic continuity.\n"
        f"Use NER (e.g., \"HITO_1\"), keywords (e.g., \"có\", \"đội nhóm\"), and vibe to match categories.\n"
        f"Confidence: 0.9+ for exact matches (e.g., \"HITO_1\" → \"Products and Services\"), 0.7-0.8 for likely (e.g., \"tăng chiều cao\" → \"Emotional and Psychological Insights\"), <0.7 for stretch.\n"
        f"Max 3 topics, primary topic first (highest confidence).\n"
        f"Ambiguous (<70% confidence)? Flag for clarification later, best guess now.\n"
        f"Output MUST be valid JSON—list of dicts with \"name\" and \"confidence\", no markdown (e.g., no ```json```). Examples:\n"
        f"\"HITO_1 giúp tăng chiều cao\" → [{{\"name\": \"Products and Services\", \"confidence\": 0.9}}]\n"
        f"\"Thời tiết lạ lắm\" → [{{\"name\": \"External Influences\", \"confidence\": 0.85}}]\n"
        f"\"Chia sẻ mẹo bán hàng\" → [{{\"name\": \"Skills\", \"confidence\": 0.9}}]"
    )
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.debug(f"Raw LLM topic response: '{response}'")
    
    try:
        topics = json.loads(response)
        if not isinstance(topics, list) or not all(isinstance(t, dict) and "name" in t and "confidence" in t for t in topics):
            raise ValueError("Invalid topic format—must be list of {name, confidence} dicts")
        topics = [t for t in topics if t["name"] in CATEGORIES][:3]  # Cap at 3, filter invalid
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON parse error: {e}. Raw response: '{response}'. Falling back to Miscellaneous")
        return [{"name": "Miscellaneous", "confidence": 0.5}]

    if not topics or max(t["confidence"] for t in topics) < 0.7:
        logger.info("Topics too weak or empty, using fallback")
        state["prompt_str"] = "Này, bạn đang chia sẻ hay tâm trạng vậy?"
        return [{"name": "Miscellaneous", "confidence": 0.5}]

    topics = sorted(topics, key=lambda x: x["confidence"], reverse=True)
    logger.info(f"Parsed topics: {json.dumps(topics, ensure_ascii=False)}")
    return topics

# Knowledge Extraction
def extract_knowledge(state, user_id=None):
    intent = detect_intent(state)
    topics = identify_topic(state)
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    
    prompt = f"""Này, tớ là Ami—trích xuất Info Pieces từ text! Given:
    - Latest: '{latest_msg}'
    - Intent: '{intent}'
    - Topics: {json.dumps(topics)}
    Return raw JSON: {{"pieces": [{{"text": "HITO_1 tăng chiều cao", "intent": "teaching", "topic": {{"name": "Products and Services", "confidence": 0.9}}, "meaningfulness_score": 0.85, "needs_clarification": false, "raw_input": "HITO_1 helps with height"}}]}}
    
    Rules:
    - "pieces": List—each:
      - "text": Câu rõ nghĩa, ngắn.
      - "intent": Từ detect_intent.
      - "topic": Primary topic từ identify_topic.
      - "meaningfulness_score": 0-1—0.8+ specific, 0.5-0.79 vague, <0.5 noise.
      - "needs_clarification": True nếu <0.8—sẽ hỏi.
      - "raw_input": Original text.
    - Dynamic Product: Scan "HITO_1", "Glucosamine"—tạo temp `product_id` nếu mới.
    - Ingredient Linking: Tie "Glucosamine" ↔ "tăng chiều cao" trong context.
    - NO translation—giữ tiếng Việt.
    - Output MUST be valid JSON, no markdown (e.g., no ```json```)."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.debug(f"Raw extract response: '{response}'")
    
    try:
        response = json.loads(response)
        pieces = response["pieces"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Piece parse error: {e}. Raw response: '{response}'. Falling back")
        pieces = [{"text": latest_msg, "intent": intent, "topic": topics[0], "meaningfulness_score": 0.5, "needs_clarification": True, "raw_input": latest_msg}]
    
    # Dynamic Product Handling + Ingredient Linking
    primary_topic = topics[0]["name"]
    if primary_topic == "Products and Services":
        for piece in pieces:
            product_candidates = [w for w in piece["text"].split() if "HITO" in w or "TEMP" in w]
            if product_candidates:
                piece["product_id"] = product_candidates[0]
            else:
                piece["product_id"] = f"TEMP_{uuid.uuid4().hex[:6]}"
                state["prompt_str"] = f"Ami hiểu '{piece['text']}' là sản phẩm mới—đúng không?"
            
            # Ingredient linking
            ingredients = [w for w in piece["text"].split() if "glucosamine" in w.lower() or "bột" in w.lower()]
            if ingredients and "product_id" in piece:
                piece["linked_entities"] = {"product_id": piece["product_id"], "ingredients": ingredients}
                state["prompt_str"] = f"Ami thấy '{ingredients[0]}' liên quan tới '{piece['product_id']}'—đúng không?"
    
    # Clarification Check
    for piece in pieces:
        if piece["meaningfulness_score"] < 0.8:
            piece["needs_clarification"] = True
            state["prompt_str"] = f"Này, ý bạn là gì với '{piece['text']}' vậy?"
    
    # Buffer to Pending Node
    pending_node = state.get("pending_node", {"pieces": [], "primary_topic": primary_topic})
    pending_node["pieces"].extend(pieces)
    pending_node["primary_topic"] = primary_topic  # Ensure it’s always set
    state["pending_node"] = pending_node
    logger.info(f"Extracted pieces: {json.dumps(pieces, ensure_ascii=False)}")
    return pending_node

# Confirm Knowledge
def confirm_knowledge(state, user_id):
    node = state.get("pending_node", {"pieces": [], "primary_topic": "Miscellaneous"})
    if not node["pieces"]:
        return None
    
    if "teaching" in [p["intent"] for p in node["pieces"]]:
        state["prompt_str"] = f"Ami hiểu là {node['pieces'][0]['text']}—đúng không?"
    else:
        state["prompt_str"] = "Ami lưu cả mớ này nhé?"
    
    # Simulate user input (replace with real integration later)
    response = input(state["prompt_str"] + " (yes/no): ").lower()
    if response == "yes":
        node["node_id"] = f"node_{uuid.uuid4()}"
        node["convo_id"] = state.get("convo_id", str(uuid.uuid4()))
        node["confidence"] = sum(p["meaningfulness_score"] for p in node["pieces"]) / len(node["pieces"])
        node["vibe_score"] = node["confidence"]
        node["created_at"] = datetime.now().isoformat()
        node["last_accessed"] = node["created_at"]
        node["access_count"] = 0
        node["confirmed_by"] = user_id or "user123"
        store_in_pinecone(node, user_id)
        time.sleep(2)  # Delay to ensure Pinecone indexes the vector
        state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}  # Reset with primary_topic
    elif response == "no":
        state["prompt_str"] = "OK, Ami bỏ qua nhé. Có gì thêm không?"
        state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}  # Reset with primary_topic
    return node

# Pinecone Storage
def store_in_pinecone(node, user_id):
    vector_id = f"{node['node_id']}_{node['primary_topic']}_{node['created_at']}"
    # Use raw_input for embedding to match query phrasing
    embedding_text = " ".join(p["raw_input"] for p in node["pieces"])
    embedding = EMBEDDINGS.embed_query(embedding_text)
    metadata = {
        "node_id": node["node_id"],
        "convo_id": node["convo_id"],
        "pieces": json.dumps(node["pieces"], ensure_ascii=False),  # Serialize to string
        "primary_topic": node["primary_topic"],
        "confidence": node["confidence"],
        "product_id": node["pieces"][0].get("product_id", "unknown"),
        "linked_nodes": json.dumps([p.get("linked_entities", {}).get("product_id", "") for p in node["pieces"] if "linked_entities" in p], ensure_ascii=False),  # Serialize to string
        "confirmed_by": node["confirmed_by"],
        "vibe_score": node["vibe_score"],
        "last_accessed": node["last_accessed"],
        "access_count": node["access_count"],
        "created_at": node["created_at"]
    }
    try:
        index.upsert([(vector_id, embedding, metadata)], namespace="")
        logger.info(f"Stored node: {vector_id}")
    except Exception as e:
        logger.error(f"Pinecone upsert failed: {e}")

# Recall Knowledge
def recall_knowledge(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent = detect_intent(state)
    
    # Preset Brain check first
    message_lower = message.lower()
    for category, preset in PRESET_KNOWLEDGE.items():
        if category in message or any(kw in message_lower for kw in CATEGORY_KEYWORDS.get(category, [])):
            return {"response": f"Ami đây! {preset['text']}—thử không?", "mode": "Co-Pilot", "source": "Preset"}

    query_embedding = EMBEDDINGS.embed_query(message)
    logger.debug(f"Query embedding: {query_embedding[:5]}... (truncated)")  # Log first 5 values
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace="")
    
    # Convert ScoredVector to dict for logging
    matches = [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in results["matches"]]
    logger.debug(f"Pinecone query results: {json.dumps(matches, ensure_ascii=False)}")
    
    now = datetime.now()
    nodes = []
    for r in results["matches"]:
        meta = r.metadata
        # Deserialize pieces and linked_nodes if they exist
        if "pieces" in meta:
            meta["pieces"] = json.loads(meta["pieces"])
        else:
            meta["pieces"] = []  # Fallback to empty list
            logger.warning(f"No 'pieces' in metadata for vector {r.id}")
        if "linked_nodes" in meta:
            meta["linked_nodes"] = json.loads(meta["linked_nodes"])
        else:
            meta["linked_nodes"] = []
        # Handle missing fields with defaults
        meta["last_accessed"] = meta.get("last_accessed", now.isoformat())
        meta["access_count"] = meta.get("access_count", 0)
        meta["confidence"] = meta.get("confidence", 0.5)
        days_since = (now - datetime.fromisoformat(meta["last_accessed"])).days
        relevance = r.score
        recency = max(0, 1 - 0.05 * days_since)
        usage = min(1, meta["access_count"] / 10)
        vibe_score = (0.4 * relevance) + (0.3 * recency) + (0.3 * usage)
        nodes.append({"meta": meta, "vibe_score": vibe_score, "confidence": meta["confidence"]})
    
    if not nodes:
        return {"response": "Ami đây! Chưa đủ info, bro thêm tí nha!", "mode": "Co-Pilot", "source": "Preset"}
    
    top_node = max(nodes, key=lambda x: x["vibe_score"] * x["confidence"])
    index.update(id=f"{top_node['meta']['node_id']}_{top_node['meta']['primary_topic']}_{top_node['meta']['created_at']}",
                 set_metadata={"last_accessed": now.isoformat(), "access_count": top_node['meta']['access_count'] + 1},
                 namespace="")
    
    prompt = f"""You’re Ami, crafting a response from the top knowledge node. Given:
    - Input: '{message}'
    - Intent: '{intent}'
    - Top node: {json.dumps(top_node['meta'], ensure_ascii=False)}
    Return raw JSON: {{"response": "HITO_1 tăng chiều cao nha bro!", "mode": "Co-Pilot", "source": "Enterprise"}}
    
    Rules:
    - "response": Vietnamese, casual, sales-y—use pieces from node.
    - "mode": "Autopilot" if "request", else "Co-Pilot".
    - "source": "Enterprise" from node.
    - Keep it short, actionable, charming.
    - Output MUST be valid JSON, no markdown."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.debug(f"Raw recall response: '{response}'")
    
    try:
        response = json.loads(response)
        logger.info(f"Recalled: {json.dumps(response, ensure_ascii=False)}")
        return response
    except json.JSONDecodeError as e:
        logger.error(f"Recall parse error: {e}. Raw response: '{response}'. Falling back")
        return {"response": "HITO_1 tăng chiều cao nha bro!", "mode": "Co-Pilot", "source": "Enterprise"}

# Test Sim
def test_ami():
    logger.info("Testing Ami—locked to 3_3!")
    state = {"messages": [], "prompt_str": "", "convo_id": str(uuid.uuid4())}
    
    # Test 1: Product teaching
    state["messages"].append(HumanMessage("HITO_1 helps with height"))
    pending_node = extract_knowledge(state)
    confirm_knowledge(state, "user123")  # Assume "yes"
    
    # Test 2: Ingredient linking
    state["messages"].append(HumanMessage("It’s got Glucosamine and stuff"))
    pending_node = extract_knowledge(state)
    logger.info(state["prompt_str"])
    state["messages"].append(HumanMessage("Bột xương cá tuyết"))  # Fixed input typo
    pending_node = extract_knowledge(state)
    confirm_knowledge(state, "user123")  # Assume "yes"
    
    # Test 3: Recall
    result = recall_knowledge("HITO_1 có gì hay?", "user123")
    logger.info(f"Recall result: {json.dumps(result, ensure_ascii=False)}")
    
    # Test 4: Preset fallback
    result = recall_knowledge("Thời tiết hôm nay thế nào?", "user123")
    logger.info(f"Preset recall: {json.dumps(result, ensure_ascii=False)}")

if __name__ == "__main__":
    test_ami()
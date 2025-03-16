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

def extract_knowledge(state, user_id=None):
    intent = detect_intent(state)
    topics = identify_topic(state)
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})  # {term_name: {"term_id": str, "vibe_score": float}}
    convo_history = " ".join(m.content for m in messages)
    convo_id = state.get("convo_id", str(uuid.uuid4()))
    
    prompt = f"""You’re Ami, extracting for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}'
    - Convo History (last 5): '{convo_history}'
    - Active Terms: {json.dumps(active_terms, ensure_ascii=False)}
    Return raw JSON: {{"terms": {{"<term_name>": {{"knowledge": ["<chunk1>", "<chunk2>"], "aliases": ["<variant1>"]}}}}, "piece": {{"intent": "{intent}", "topic": {json.dumps(topics[0])}, "raw_input": "{latest_msg}"}}}}
    Rules:
    - "terms": Extract ALL full noun phrases or product names (e.g., "HITO Cốm", "iPhone") via NER or context. Prioritize product names (e.g., "HITO Cốm") over generics unless explicitly standalone—check convo history for recent product mentions.
    - "knowledge": Chunk input into meaningful phrases tied to each term—split naturally. "Nó"/"it" refers to highest vibe_score term.
    - "aliases": List variants (e.g., "HITO Com" for "HITO Cốm") if detected via context or fuzzy match (>80% similarity).
    - "piece": Single object with intent, primary topic, raw_input.
    - NO translation—keep input language EXACTLY as provided.
    - Example: "HITO Cốm tăng chiều cao, HITO Com ngon" → {{"terms": {{"HITO Cốm": {{"knowledge": ["tăng chiều cao", "ngon"], "aliases": ["HITO Com"]}}}}, "piece": ...}}
    - Output MUST be valid JSON, no extra brackets or trailing chars."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    logger.info(f"Raw LLM response from extract_knowledge: '{response}'")
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
        # Fuzzy match against existing terms
        canonical_name = term_name
        for existing_name in active_terms:
            if fuzz.ratio(term_name.lower(), existing_name.lower()) > 80:
                canonical_name = existing_name
                break
        term_id = active_terms.get(canonical_name, {}).get("term_id", f"term_{sanitize_vector_id(canonical_name)}_{convo_id}")
        active_terms[canonical_name] = {"term_id": term_id, "vibe_score": active_terms.get(canonical_name, {}).get("vibe_score", 1.0)}
        # Add to term_refs if not already present
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
        node["primary_topic"] = node["pieces"][0]["topic"]["name"]
        
        pending_knowledge = state.get("pending_knowledge", {})
        for term_id, knowledge in pending_knowledge.items():
            # Pass aliases along to upsert
            upsert_term_node(term_id, state["convo_id"], knowledge)
            time.sleep(2)  # Reduced delay—batch later if needed
        
        store_convo_node(node, user_id)
        time.sleep(2)  # Reduced delay
        state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
        state.pop("pending_knowledge", None)
    elif response == "no":
        state["prompt_str"] = f"OK, Ami bỏ qua. Còn gì thêm cho {node['primary_topic']} không?"
        state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
    return node

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

def recall_knowledge(message, user_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent = detect_intent(state)
    
    # Check Preset Brain first
    message_lower = message.lower()
    preset_results = index.query(vector=EMBEDDINGS.embed_query(message), top_k=2, include_metadata=True, namespace="Preset")
    for r in preset_results["matches"]:
        if r.score > 0.8:
            return {"response": f"Ami đây! {r.metadata['text']}—thử không bro?", "mode": "Co-Pilot", "source": "Preset"}

    # Enterprise Brain recall
    query_embedding = EMBEDDINGS.embed_query(message)
    convo_results = index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace="convo_nodes")  # Upped to 10
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
        vibe_score = (0.5 * relevance) + (0.3 * recency) + (0.2 * usage)  # Upped relevance weight
        nodes.append({"meta": meta, "vibe_score": vibe_score})
    
    filtered_nodes = [n for n in nodes if n["meta"]["pieces"]] or nodes[:2]
    if not filtered_nodes:
        return {"response": f"Ami đây! Chưa đủ info, bro thêm tí nha!", "mode": "Co-Pilot", "source": "Enterprise"}
    
    filtered_nodes.sort(key=lambda x: (x["vibe_score"], datetime.fromisoformat(x["meta"]["last_accessed"])), reverse=True)
    top_nodes = filtered_nodes[:3]  # Upped to 3 for broader context
    term_ids = set(t for n in top_nodes for p in n["meta"]["pieces"] for t in p["term_refs"])
    
    # Update Term Vibe Scores
    term_nodes = index.fetch(list(term_ids), namespace="term_memory").get("vectors", {})
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
    
    # Pitch Response
    prompt = f"""You’re Ami, pitching for AI Brain Mark 3.4. Given:
    - Input: '{message}'
    - Intent: '{intent}'
    - Convo Nodes: {json.dumps([n['meta'] for n in top_nodes], ensure_ascii=False)}
    - Term Nodes: {json.dumps({tid: tn['metadata'] for tid, tn in term_nodes.items()}, ensure_ascii=False)}
    Return raw JSON: {{"response": "<response>", "mode": "<mode>", "source": "Enterprise"}}
    Rules:
    - "response": Vietnamese, casual, sales-y—use convo + term data (all relevant nodes), check aliases for variants. For 'request', prioritize sales process steps (e.g., address, combos, payment) over general info. For 'question', predict objections (e.g., age, cost).
    - "mode": "Autopilot" if "request", else "Co-Pilot".
    - "source": "Enterprise".
    - Short, actionable, charming—use highest vibe_score terms first.
    - Example: "Tôi muốn mua HITO Cốm" → {{"response": "Bro cho địa chỉ giao hàng đi, Ami gửi combo #1, #2, #3, chuyển tiền @VCB Germany nhé!", "mode": "Autopilot", "source": "Enterprise"}}
    - Output MUST be valid JSON, no markdown."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    try:
        result = json.loads(response)
        logger.info(f"Recalled: {json.dumps(result, ensure_ascii=False)}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Recall parse error: {e}. Raw: '{response}'. Falling back")
        return {"response": f"Ami đây! Chưa rõ lắm, bro nói thêm đi!", "mode": "Co-Pilot", "source": "Enterprise"}

def test_ami():
    logger.info("Testing Ami—AI Brain Mark 3.4 locked and loaded!")
    state = {"messages": [], "prompt_str": "", "convo_id": str(uuid.uuid4()), "active_terms": {}, "pending_knowledge": {}}
    confirm_callback = lambda x: "yes"  # Auto-confirm for stress test
    
    # Training Path - Stack Knowledge with 9 Inputs
    inputs = [
        "HITO Cốm có khả năng tăng cường hấp thụ Calcium giúp người lớn tăng trưởng đột phá chiều cao, phù hợp cho người từ 10 tới 30 tuổi.",
        "Thành phần của HITO cốm là: Bột xương cá tuyết (25% canxi) và Aquamin F( 32% canxi , canxi từ tảo biển đỏ)",
        "Công dụng của Aquamin F( 32% canxi , canxi từ tảo biển đỏ) là Bổ sung canxin hữu cơ dễ hấp thu mà còn không lắng cặn, không bị nóng trong hay táo bón như canxin vô cơ .Củng cố hệ xương, bổ sung canxi giúp xương chắc khỏe, dẻo dai. bảo vệ và tham gia vào quá trình hình thành dịch nhầy ở khớp, giúp khớp chuyển động linh hoạt, thoải mái hơn.giúp ngăn ngừa việc hình thành khối u ở gan, polyp trực tràng. Đồng thời bảo vệ sức khỏe đường tiêu hóa",
        "Công dụng của Bột xương cá tuyết (25% canxi) là Bổ sung canxi ,Hỗ trợ sự phát triển hệ xương và cơ bắp, giúp trẻ tăng trưởng chiều cao,đóng vai trò thiết yếu cho việc phát triển xương, làm chắc xương, răng, chống loãng xương, còi xương, nhuyễn xương,… mà còn có vai trò quan trọng trong việc duy trì hoạt động cơ bắp, thông máu, phát tín hiệu cho các tế bào thần kinh và giúp tiết chế một số kích thích tố",
        "Với người trên 25 tuổi có thể phát triển tiếp 3cm chiều cao. Với người trên 30 tuổi thì giúp ổn định hấp thụ xương",
        "Khi tư vấn cho khách hàng về HITO cốm, cần khai thác tên tuổi giới tính chiều cao hiện tại và chiều cao mong muốn.",
        "Với khách hàng trên 30 tuổi, có thể gợi ý mua cho con hoặc người thân nhỏ tuổi",
        "Khi khách hàng chưa tin tưởng thì cần gửi ảnh <link ảnh>",
        "Khi khách hàng muốn mua thì hỏi địa chỉ giao hàng sau đó gửi các combo: #combo 1, #combo2, #combo 3. Sau đó yêu cầu khách hàng chuyển tiền theo thông tin chuyển tiền @VCB Germany."
    ]
    
    logger.info("Training Path: Stacking 9 inputs...")
    for i, input_text in enumerate(inputs, 1):
        logger.info(f"Input {i}: {input_text}")
        state["messages"].append(HumanMessage(input_text))
        extract_knowledge(state)
        confirm_knowledge(state, "user123", confirm_callback)
        logger.info(f"Active Terms: {json.dumps(state['active_terms'], ensure_ascii=False)}")
        logger.info(f"Pending Node Cleared: {json.dumps(state['pending_node'], ensure_ascii=False)}")
    
    # Selling Path - Stress Test Recall with 5 Queries
    queries = [
        "HITO Cốm có gì hay?",
        "Thành phần HITO Com là gì?",
        "Aquamin F có công dụng gì?",
        "Tôi 35 tuổi, HITO Cốm có tác dụng không?",
        "Tôi muốn mua HITO Cốm, làm sao đây?"
    ]
    
    logger.info("Selling Path: Testing 5 recall queries...")
    for i, query in enumerate(queries, 1):
        logger.info(f"Query {i}: {query}")
        result = recall_knowledge(query, "user123")
        logger.info(f"Recall result: {json.dumps(result, ensure_ascii=False)}")
    
    logger.info("Stress Test Complete—Ami Mark 3.4 flexed hard, bro!")

if __name__ == "__main__":
    test_ami()
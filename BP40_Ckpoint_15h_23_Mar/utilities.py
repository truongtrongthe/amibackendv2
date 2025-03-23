# utilities.py (Final Blue Print 4.0 - Enterprise Brain, deployed March 20, 2025)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import json
import pinecone
from datetime import datetime
from pinecone_datastores import index  # Assuming this is your Pinecone index setup
import uuid
import re
import ast
import logging

# Setup logging

#logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)  # Flask server noise
logging.getLogger("http.client").setLevel(logging.WARNING)  # HTTP requests
logging.getLogger("urllib3").setLevel(logging.WARNING)  # HTTP-related
logger = logging.getLogger(__name__)

# Config
LLM = ChatOpenAI(model="gpt-4o", streaming=True)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

CATEGORIES = ["Products", "Companies", "Skills", "People", "Customer Segments", "Ingredients", "Markets", 
              "Technologies", "Projects", "Teams", "Events", "Strategies", "Processes", "Tools", 
              "Regulations", "Metrics", "Partners", "Competitors"]

PRESET_KNOWLEDGE = {
    "Products": {"text": "Generic Supplement", "confidence": 0.9},
    "Companies": {"text": "Generic Corp", "confidence": 0.85},
}

def sanitize_vector_id(text):
    return text.replace(" ", "_").lower()

def clean_llm_response(response):
    response = response.strip()
    # Remove language prefix/suffix, markdown, and extra text
    for lang in ['python', 'json', '']:
        response = re.sub(rf'^{lang}\s*[\n\r]*', '', response, flags=re.MULTILINE)
        response = re.sub(rf'[\n\r]*\s*{lang}$', '', response, flags=re.MULTILINE)
    if response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()
    # Strip any trailing text after JSON
    json_match = re.match(r'\{.*\}', response, re.DOTALL)
    return json_match.group(0) if json_match else response

def initialize_preset_brain():
    for category, data in PRESET_KNOWLEDGE.items():
        vector_id = f"node_{sanitize_vector_id(category)}_preset_{uuid.uuid4()}"
        embedding = EMBEDDINGS.embed_query(data["text"])
        metadata = {
            "name": data["text"], "category": category, "vibe_score": 1.0,
            "attributes": [], "relationships": [], "created_at": datetime.now().isoformat()
        }
        try:
            index.upsert([(vector_id, embedding, metadata)], namespace="preset_knowledge_tree")
        except Exception as e:
            logger.error(f"Preset upsert failed: {e}")

initialize_preset_brain()


def detect_topic(state):
    messages = state["messages"][-3:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    context = " | ".join([m.content for m in messages]) + " | " + " ".join(active_terms.keys())

    prompt = f"""Detect ONE topic for Enterprise Brain 4.0. Given:
    - Latest message: '{latest_msg}'
    - Context (3 messages + active terms): '{context}'
    Pick ONE from: {', '.join(CATEGORIES)}.
    Return JSON: {{"category": "Products", "confidence": 0.9}}.
    Rules:
    - Prioritize latest message, weigh active terms for relevance.
    - Confidence < 0.7? Return 'unclear'.
    - Output MUST be valid JSON."""
    
    response = ""
    for chunk in LLM.invoke(prompt):
        if isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == 'content':
            response += chunk[1]
    response = clean_llm_response(response)
    try:
        result = json.loads(response)
        category = result["category"]
        confidence = result["confidence"]
        if category not in CATEGORIES or confidence < 0.7:
            return "unclear"
        return category
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Topic parse error: {e}. Raw: '{response}'")
        return "unclear"

def extract_knowledge(state, user_id=None, intent=None):
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    focus_history = state.get("focus_history", [])
    convo_id = state.get("convo_id", "test_convo")
    user_id = user_id or state.get("user_id", "default_user")

    logger.info("Entering extract_knowledge")

    try:
        # Step 1: Intent detection
        intent = intent or detect_intent(state)
        logger.info(f"Step 1 - Intent detected: '{intent}' for message: '{latest_msg}'")

        # Step 2: Term extraction
        term = None
        confidence = 1.0 if intent == "teaching" else 0.0
        if latest_msg.strip():
            convo_history = " | ".join(m.content for m in messages[:-1]) if len(messages) > 1 else "None"
            focus_str = ", ".join([f"{t['term']} (score: {t['score']:.1f})" for t in focus_history]) if focus_history else "None"
            term_prompt = f"""Given:
                - Latest message: '{latest_msg}'
                - Prior messages: '{convo_history}'
                - Intent: '{intent}'
                - Focus history (term, relevance score): '{focus_str}'
                What’s the main term or phrase this is about? Return just the term, nothing else. Examples:
                - 'yo, got some dope info!' → 'None'
                - 'hito granules—height growth booster!' → 'HITO Granules'
                - 'HITO boosts height maybe?' → 'HITO'
                - 'Good—made by novahealth!' → 'HITO Granules' (if HITO Granules is high in focus)
                Rules:
                - Pick the subject or focus—usually a product, company, or key concept at the start of the statement.
                - Strongly favor the highest-scored term in focus history if the message is vague or lacks an explicit subject (e.g., 'của nó', 'Ngoài ra'), unless clearly contradicted.
                - Return 'None' only if no term fits after context analysis."""

            logger.info("Before term extraction")
            term_response = LLM.invoke(term_prompt).content.strip()
            logger.info("After term extraction")
            term = clean_llm_response(term_response)
            if term.lower() == "none" or not term:
                words = latest_msg.split()
                term = next((w for w in words if w[0].isupper() and w not in ["I", "A"]), None)

            if intent == "teaching" and term:
                conf_prompt = f"""Given:
                - Latest message: '{latest_msg}'
                - Prior messages: '{convo_history}'
                - Identified term: '{term}'
                - Intent: '{intent}'
                - Focus history (term, relevance score): '{focus_str}'
                Rate confidence (0.0-1.0) that '{term}' is the main topic being taught or described.
                Rules:
                - Explicit mention (e.g., 'HITO is a height booster') → 0.9-1.0.
                - Pronouns or vague references (e.g., 'It’s made in Japan') → 0.5-0.7, higher if prior messages mention '{term}'.
                - Uncertainty markers (e.g., 'maybe', 'có lẽ') → cap at 0.6 unless prior context is definitive.
                - Contradictory context → < 0.5.
                Return ONLY JSON: {{"confidence": 0.95}}"""
                logger.info("Before confidence assessment")
                conf_response = clean_llm_response(LLM.invoke(conf_prompt).content)
                logger.info("After confidence assessment")
                confidence = json.loads(conf_response).get("confidence", 0.5)

                found = False
                for focus in focus_history:
                    if focus["term"].lower() == term.lower():
                        focus["score"] = min(focus["score"] + 0.3, 5.0)
                        found = True
                        break
                if not found:
                    focus_history.append({"term": term, "score": 1.0})
                for focus in focus_history:
                    if focus["term"] != term:
                        focus["score"] = max(focus["score"] - 0.3, 0.0)
                focus_history = [f for f in focus_history if f["score"] > 0]

        logger.info(f"Identified term: '{term}' (confidence: {confidence:.1f})")

        # Step 3: Attributes and relationships extraction
        attributes = []
        relationships = []
        if term and latest_msg.strip():
            focus_score = max([f["score"] for f in focus_history if f["term"] == term], default=0)
            convo_history = " | ".join([m.content for m in messages[-3:-1]])  # Last 2 messages for context
            attr_prompt = f"""Given:
                - Latest message: '{latest_msg}'
                - Prior messages: '{convo_history}'
                - Main term: '{term}' (focus score: {focus_score:.1f})
                - Intent: '{intent}'
                - Focus history (term, relevance score): '{focus_str}'
                List descriptive properties about '{term}' as a JSON-compatible Python list of dicts.
                Strongly assume the message describes '{term}'—treat pronouns (e.g., 'it'), vague references (e.g., 'made in Japan'), or negations (e.g., 'No, HITO’s from Vietnam') as adding/updating details to '{term}' unless clearly contradicted by prior messages.
                Examples:
                - 'HITO is a height booster' → [{{"key": "Use", "value": "height booster"}}]
                - 'Giá của nó là combo1 $500 và combo2 $800' → [{{"key": "Price", "value": "combo1 $500"}}, {{"key": "Price", "value": "combo2 $800"}}]
                - 'No, HITO’s from Vietnam' after 'It’s made in Japan' → [{{"key": "Origin", "value": "Vietnam"}}]
                Rules:
                - Include features (e.g., 'Use'), origins (e.g., 'Origin'), benefits, prices (e.g., 'Price').
                - Capture monetary values (e.g., '$500') as 'Price'—list each separately if multiple.
                - Use original message language, append '(maybe)' for uncertainty.
                - Check prior messages for context (e.g., 'it' refers to 'HITO').
                - Return `[]` only if nothing fits '{term}' after analysis.
                Output ONLY the list, no 'python' or 'json' prefix."""

            logger.info("Before attributes extraction")
            attr_response = clean_llm_response(LLM.invoke(attr_prompt).content)
            logger.info(f"Raw attributes response: '{attr_response}'")
            try:
                attributes = json.loads(attr_response)
            except (ValueError, SyntaxError):
                logger.warning(f"JSON parse failed, trying ast.literal_eval: {attr_response}")
                attributes = ast.literal_eval(attr_response)
            if not isinstance(attributes, list) or not all(isinstance(a, dict) and "key" in a and "value" in a for a in attributes):
                logger.warning(f"Invalid attributes format: {attr_response}")
                attributes = []

            rel_prompt = f"""
                        Given:
                            - Latest message: '{latest_msg}'
                            - Main term: '{term}' (focus score: {focus_score:.1f})
                            - Intent: '{intent}'
                            - Focus history (term, relevance score): '{json.dumps(focus_history)}'
                            List relationships as a JSON-compatible Python list of dicts with node IDs.
                            Strongly assume the message describes '{term}' if it’s in focus history or explicitly mentioned—treat pronouns (e.g., 'của nó') or vague references (e.g., 'Ngoài ra', 'Phụ Liệu') as adding details to '{term}' unless clearly contradicted by context.
                            Example: [{{"subject": "Canxi cá tuyết", "relation": "Ingredient Of", "object": "{term}", "subject_id": "node_canxi_ca_tuyet_ingredients_user_tfl_test_{uuid.uuid4()}"}}]
                            Rules:
                                - Identify external entities (e.g., companies, customer segments, countries) connected via verbs/prepositions (e.g., 'dành cho', 'được...tin dùng', 'made by', 'for', 'made in').
                                - Generate unique node IDs: `node_<object>_<category>_user_tfl_test_<uuid>`.
                                - For components (e.g., ingredients), use 'Ingredient Of' with subject as the component and object as '{term}'.
                                - For locations (e.g., countries like 'Japan'), use 'Made In' if 'made in' is present, with subject as '{term}' and object as the location.
                                - Use category from context (e.g., 'Companies', 'Customer Segments', 'Countries', 'Ingredients').
                                - Examples: 'dành cho Việt kiều' → Customer Segment, 'được CLB tin dùng' → Endorsement, 'made in Japan' → Countries.
                                - Return `[]` if no clear relationships.
                            Output ONLY the list, no 'python' or 'json' prefix.
                        """
            logger.info("Before relationships extraction")
            rel_response = clean_llm_response(LLM.invoke(rel_prompt).content)
            logger.info(f"Raw relationships response: '{rel_response}'")
            try:
                relationships = json.loads(rel_response)
            except (ValueError, SyntaxError):
                logger.warning(f"JSON parse failed, trying ast.literal_eval: {rel_response}")
                relationships = ast.literal_eval(rel_response)
            if not isinstance(relationships, list) or not all(isinstance(r, dict) and "subject" in r and "relation" in r and "object" in r for r in relationships):
                logger.warning(f"Invalid relationships format: {rel_response}")
                relationships = []

            # Add subject_id if missing in relationships
            for rel in relationships:
                if "subject_id" not in rel or "<uuid>" in rel["subject_id"]:
                    category = "Countries" if rel["relation"] == "Made In" else "Unknown"
                    rel["subject_id"] = f"node_{rel['subject'].lower().replace(' ', '_')}_{category.lower()}_user_{user_id}_{uuid.uuid4()}"
        logger.info(f"Extracted - Attributes: {attributes}, Relationships: {relationships}")

        # Step 4: Update state
        state["focus_history"] = focus_history
        if term and (attributes or relationships):  # Changed to include relationships
            category = detect_topic(state)
            if category == "unclear" or category not in CATEGORIES:
                logger.info(f"Category unclear or invalid: '{category}', defaulting to 'Products'")
                category = "Products"
            
            term_id = f"node_{sanitize_vector_id(term)}_{sanitize_vector_id(category)}_user_{user_id}_{uuid.uuid4()}"
            parent_id = f"node_{sanitize_vector_id(category)}_user_{user_id}_{uuid.uuid4()}"
            existing = active_terms.get(term, {})
            existing_attrs = existing.get("attributes", [])
            vibe_score = existing.get("vibe_score", 1.0)
            
            # Merge only new attributes from this turn, don’t pull Pinecone history
            merged_attrs = list({(a["key"], a["value"]): a for a in existing_attrs + attributes}.values())
            if len(merged_attrs) > len(existing_attrs) or relationships:
                vibe_score += 0.3
            
            active_terms[term] = {
                "term_id": existing.get("term_id", term_id),
                "vibe_score": vibe_score,
                "attributes": merged_attrs  # Only current convo’s attrs
            }
            state["pending_knowledge"] = {
                "term_id": active_terms[term]["term_id"],
                "name": term,
                "category": category,
                "attributes": merged_attrs,
                "relationships": relationships,
                "vibe_score": vibe_score,
                "parent_id": parent_id,
                "needs_confirmation": confidence < 0.8
            }
        else:
            state["pending_knowledge"] = {}

        state["active_terms"] = active_terms
        logger.info(f"Active terms set to state: {state['active_terms']}")
        # Step 5: Return
        logger.info("Exiting extract_knowledge successfully")
        return {"term": term, "attributes": attributes, "relationships": relationships, "confidence": confidence}

    except Exception as e:
        logger.error(f"Extract_knowledge failed: {e}", exc_info=True)
        state["active_terms"] = active_terms
        state["focus_history"] = focus_history
        state["pending_knowledge"] = {}
        logger.info(f"Active terms after failure: {state['active_terms']}")
        return {"term": None, "attributes": [], "relationships": [], "confidence": 0.0}


def recall_knowledge(message, state, user_id=None):
    #intent = detect_intent(state)
    namespace = f"enterprise_knowledge_{user_id}"
    
    active_terms = state.get("active_terms", {})
    query_text = f"{message} {' '.join(active_terms.keys())}"  # e.g., "Tell me about HITO? HITO"
    query_embedding = EMBEDDINGS.embed_query(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )
    #logger.info(f"Query results: {results}")
    logger.info(f"Query found {len(results['matches'])} matches")
    
    nodes = []
    now = datetime.now()
    
    # If query fails, try direct fetch by term_id
    if not results["matches"] and active_terms:
        logger.info("Query returned no matches, attempting direct fetch by term_id")
        term_ids = [v["term_id"] for v in active_terms.values()]
        fetched = index.fetch(term_ids, namespace=namespace).vectors
        logger.info(f"Direct fetch results: {fetched}")
        for term_id, data in fetched.items():
            meta = data.metadata
            days_since = (now - datetime.fromisoformat(meta["created_at"])).days
            vibe_score = meta["vibe_score"] - (0.05 * (days_since // 30))
            vibe_score = min(2.2, max(0.1, vibe_score + 0.1))
            if meta["name"] in active_terms:
                vibe_score = min(2.2, max(vibe_score, active_terms[meta["name"]]["vibe_score"]))
            meta["vibe_score"] = vibe_score
            nodes.append({"id": term_id, "meta": meta, "score": 1.0})  # High score for direct match
    else:
        for r in results["matches"]:
            meta = r.metadata
            logger.info(f"Node metadata: {meta}")
            days_since = (now - datetime.fromisoformat(meta["created_at"])).days
            vibe_score = meta["vibe_score"] - (0.05 * (days_since // 30))
            vibe_score = min(2.2, max(0.1, vibe_score + 0.1))
            if meta["name"] in active_terms:
                vibe_score = min(2.2, max(vibe_score, active_terms[meta["name"]]["vibe_score"]))
            meta["vibe_score"] = vibe_score
            nodes.append({"id": r.id, "meta": meta, "score": r.score})
    
    nodes = [n for n in nodes if "parent_id" in n["meta"] and n["meta"]["name"] != n["meta"]["category"]]
    if not nodes:
        logger.warning(f"No child nodes found in {namespace} for query: '{query_text}'")
        return {"knowledge": [], "terms": {}}
    
    nodes = sorted(nodes, key=lambda x: (x["score"], x["meta"]["created_at"]), reverse=True)
    
    merged = {}
    for n in nodes[:3]:
        name = n["meta"]["name"]
        if name not in merged:
            merged[name] = {
                "name": name,
                "vibe_score": n["meta"]["vibe_score"],
                "attributes": json.loads(n["meta"].get("attributes", "[]")),
                "relationships": json.loads(n["meta"].get("relationships", "[]"))
            }
        else:
            new_attrs = json.loads(n["meta"].get("attributes", "[]"))
            attr_dict = {a["key"]: a["value"] for a in merged[name]["attributes"]}
            for a in new_attrs:
                if a["key"] not in attr_dict or len(a["value"]) > len(attr_dict[a["key"]]):
                    attr_dict[a["key"]] = a["value"]
            merged[name]["attributes"] = [{"key": k, "value": v} for k, v in attr_dict.items()]
            merged[name]["relationships"] = list({(r["subject"], r["relation"], r["object"]): r for r in merged[name]["relationships"] + json.loads(n["meta"].get("relationships", "[]"))}.values())
            merged[name]["vibe_score"] = max(merged[name]["vibe_score"], n["meta"]["vibe_score"])
    
    state["active_terms"] = {n["meta"]["name"]: {"term_id": n["id"], "vibe_score": n["meta"]["vibe_score"]} for n in nodes[:3]}
    knowledge = list(merged.values())
    terms = {k["name"]: {
        "attributes": json.dumps(k["attributes"], ensure_ascii=False),
        "relationships": json.dumps(k["relationships"], ensure_ascii=False),
        "vibe_score": k["vibe_score"],
        "category": n["meta"]["category"],
        "created_at": n["meta"]["created_at"],
        "name": k["name"],
        "parent_id": n["meta"].get("parent_id", None)
    } for k, n in zip(knowledge, nodes)}
    return {"knowledge": knowledge, "terms": terms}
# Test the 5-turn convo
import time
if __name__ == "__main__":
    state = {
    "messages": [],
    "convo_id": "conv_67890",
    "active_terms": {},
    "pending_knowledge": {},
    "user_id": "brian",
    "intent": "Idle",
    "intent_history": []
    }
    user_id = "brian_789"

    turns = [
        "HITO là sản phẩm bổ sung canxi hỗ trợ phát triển chiều cao (từ 2 tuổi trở lên, đăc biệt dành cho người trưởng thành),Đối tượng KH: Việt kiều 20-30 tuổi (cốm viên) Và mẹ có con từ 12-18 tuổi ở VN (sữa, thạch). Sản phẩm cao cấp, công thức toàn diện. Được đội ngũ chuyên viên đồng hành, cung cấp thông tin chuyên khoa, cá nhân hóa. Bộ tứ canxi hữu cơ kết hợp giúp hệ xương phát tri. ển toàn diện: Canxi cá tuyết, canxi tảo đỏ, canxi Gluconate, bột nhung hươu, ở trên bảng thành phần sp A+. Sản phẩm được CLB hàng đầu VN tín nhiệm và đưa vào chế độ dinh dưỡng cho các lứa cầu thủ chuyên nghiệp. Sản phẩm canxi duy nhất được CLB Bóng đá Hoàng Anh Gia Lai tin dùng. Website: https://hitovietnam.com/. Canxi cá tuyết: cá tuyết sống ở mực nước sâu hàng nghìn mét dưới mực nước biển nên có hệ xương vững chắc, mật độ xương cao. Theo chuyên gia Hito thì xương cá tuyết có cầu tạo gần giống hệ xương người, dồi dào canxi hữu cơ (gấp 9-10 lần canxi so với các nguồn khác), tương thích sinh học cao, tăng hấp thụ tối đa canxi vào xương",
        "Thành Phần chính của nó là: Aquamin F( 32% canxi , canxi từ tảo biển đỏ): Bổ sung canxin hữu cơ dễ hấp thu mà còn không lắng cặn, không bị nóng trong hay táo bón như canxin vô cơ .Củng cố hệ xương, bổ sung canxi giúp xương chắc khỏe, dẻo dai. bảo vệ và tham gia vào quá trình hình thành dịch nhầy ở khớp, giúp khớp chuyển động linh hoạt, thoải mái hơn.giúp ngăn ngừa việc hình thành khối u ở gan, polyp trực tràng. Đồng thời bảo vệ sức khỏe đường tiêu hóa",
        "Ngoài ra còn có Collagen Type II: Giúp tăng tế bào não , tăng vận động cho các khớp nối, giảm đau với bệnh viêm khớp mạn tính, phòng ngừa  và làm giảm viêm khớp dạng thấp, bảo vệ tim mạch, chống ăn mòn hoặc  chống đông máu mạnh mẽ ngăn ngừa các cục máu đông  giảm tỉ lệ đột quỵ.",
        "Phụ Liệu : Lactose , polyvinylpyrrolidone K30, Bột talc, Kali sorbat, Hương sữa vừa đủ 1 gói. Lactose là đường tự nhiên có trong thành phần của sữa mẹ, sữa bò, sữa động vật nên an toàn tuyệt đối cho sức khỏe.polyvinylpyrrolidone K30 là chất kết dính cho dạng hạt tồn tại dưới dạng màu trăng màu vàng nhạt có khả năng hấp thụ tốt"
    ]
    for i, turn in enumerate(turns):
        state["messages"].append(HumanMessage(turn))
        intent = detect_intent(state)
        knowledge = extract_knowledge(state, user_id, intent)
        print(f"Turn {i+1} Confidence: {knowledge['confidence']:.1f}, Needs Confirmation: {knowledge['confidence'] < 0.8}")
        print(f"Attributes: {knowledge['attributes']}")
        print(f"Relationships: {knowledge['relationships']}")

    #print("Wait a moment so pinecone upsert done...")
    # Turn 5
    #time.sleep(3)
    #state["messages"].append(HumanMessage("What’s HITO?"))
    #recall = recall_knowledge("Tell me about HITO!", state, user_id)
    #print(f"Turn 5 Recall: {recall}")
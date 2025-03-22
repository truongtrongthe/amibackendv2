# utilities.py (Ami_Blue_Print_3_4 Mark 3.5 - AI Brain, locked March 16, 2025, Optimized March 18, 2025)
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
import py_vncorenlp
import os

# Setup logging (minimal)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

save_dir = "/Users/thetruong/happy/vncorenlp"
os.makedirs(save_dir, exist_ok=True)

# Load VnCoreNLP
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"], save_dir=save_dir)

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

def detect_intent(state, user_id=None):
    """
    Detects intent using 100% LLM (GPT-4o) for Vietnamese inputs.
    Args:
        state (dict): Contains 'messages' (list of HumanMessage) and 'prompt_str' (Ami's last response).
        user_id (str, optional): For future user-specific tweaks (unused here).
    Returns:
        str: Detected intent from INTENTS list.
    """
    messages = state["messages"][-3:] if state["messages"] else []  # Last 3 messages for context
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    last_ami_msg = state.get("prompt_str", "")  # Ami's last response
    latest_msg = messages[-1].content if messages else ""
    
    if not latest_msg.strip():  # Empty input fallback
        logger.info("Empty input, defaulting to greeting")
        return "greeting"

    # LLM prompt for intent detection
    prompt = f"""You’re Ami, detecting intent for a Vietnamese-speaking user. Given:
    - Latest message (70% weight): '{latest_msg}'
    - Last 3 messages (20% weight): '{convo_history}'
    - Last Ami message (10% weight): '{last_ami_msg}'
    Pick ONE intent from: {', '.join(f'"{i}"' for i in INTENTS)}.
    Return JSON: {{"intent": "teaching", "confidence": 0.9}}.
    Rules:
    - Prioritize latest message as the primary intent driver.
    - Confidence < 0.7? Pick 'confusion' and suggest clarification.
    - Output MUST be valid JSON.
    - Process Vietnamese natively, no translation needed."""
    
    response = ""
    try:
        for chunk in LLM.invoke(prompt):  # Streaming LLM response
            if isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == 'content':
                response += chunk[1]
            else:
                logger.debug(f"Ignoring non-content chunk: {chunk}")
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return "casual"

    if not response:  # No response fallback
        logger.warning("No content from LLM, defaulting to confusion")
        return "confusion"
    
    response = clean_llm_response(response)  # Strip markdown, fix JSON
    try:
        result = json.loads(response)
        intent = result["intent"]
        confidence = result["confidence"]
        
        if intent not in INTENTS:  # Validate intent
            logger.error(f"Invalid intent '{intent}' detected, defaulting to casual")
            return "casual"
        
        if confidence < 0.7:  # Low confidence triggers clarification
            state["prompt_str"] = f"Này, bạn đang muốn {INTENTS[3]} hay {INTENTS[2]} vậy?"
            logger.info(f"Low confidence ({confidence}) for '{intent}', returning confusion")
            return "confusion"
        
        logger.info(f"Detected intent: '{intent}' (confidence: {confidence})")
        return intent
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"LLM response parse error: {e}. Raw: '{response}'")
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

    # Step 1: Segment and POS tag with VnCoreNLP
    annotated_text = rdrsegmenter.annotate_text(latest_msg)
    if not isinstance(annotated_text, dict):
        logger.error(f"Unexpected annotated_text format: {annotated_text}")
        return {"terms": {}, "piece": {"intent": intent, "topic": topics[0], "raw_input": latest_msg, "piece_id": f"piece_{uuid.uuid4()}",
                                      "meaningfulness_score": 0.5, "needs_clarification": True, "term_refs": []}}
    logger.info(f"Annotated Text: {annotated_text}")

    # Step 2: Format tagged sentences for LLM
    tagged_sentences = []
    for sentence_idx, sentence_words in annotated_text.items():
        tagged_words = [f"{word['wordForm']}/{word['posTag']}" for word in sentence_words]
        tagged_sentences.append(" ".join(tagged_words))
    tagged_input = "\n".join(f"Sentence {i}: {s}" for i, s in enumerate(tagged_sentences))
    logger.info(f"Tagged Input for LLM:\n{tagged_input}")

    # Step 3: Push to LLM to reason out concepts
    prompt = f"""You’re Ami, extracting meaningful concepts for AI Brain Mark 3.4. Given:
    - Latest: '{latest_msg}' (Vietnamese - KEEP ORIGINAL LANGUAGE, NO TRANSLATION)
    - Active Terms from Context: {json.dumps({k: v['term_id'] for k, v in active_terms.items()})}
    - Tagged Sentences:
    {tagged_input}
    Return raw JSON: {{"terms": {{"<concept_name>": {{"knowledge": ["<fact1>", "<fact2>"], "aliases": ["<variant1>"]}}}}}}
    Rules:
    - Identify ALL "concepts" as nouns or noun phrases (danh từ hoặc cụm danh từ) based on POS tags (N, Nc, Np) and context.
    - Include EVERY standalone N/Nc/Np as a concept (e.g., "canxi/N", "cặn/N", "quá_trình/N") unless it’s redundant within a larger phrase already extracted.
    - Use tags: N (noun), Nc (classifier noun), Np (proper noun), E (preposition), A (adjective), V (verb), R (adverb), CH (punctuation), P (pronoun), etc.
    - Group consecutive N/Nc/Np into phrases when meaningful (e.g., "hệ/N xương/N" → "hệ xương", "trời/N mưa/V" → "trời mưa").
    - Include E if it connects nouns (e.g., "canxi/N từ/E tảo/N biển/N" → "canxi từ tảo biển").
    - Include A if it forms a meaningful compound with a noun (e.g., "canxi/N hữu_cơ/A" → "canxi hữu_cơ").
    - Include V if it’s part of a compound noun (e.g., "sức_khoẻ/N đường/N tiêu_hoá/V" → "sức_khoẻ đường tiêu_hoá").
    - For P (pronouns) like "nó," resolve to an active term from context if present (e.g., "cái bàn") or skip if no context.
    - Override mistagged A/V as N if contextually a noun (e.g., "khối_u/A" → "khối_u", "trực_tràng/A" → "trực_tràng", "mưa/V" → "mưa").
    - End phrases at R, CH, or unrelated tags unless part of a compound.
    - "knowledge" must be concise facts about each concept, derived from the input, in Vietnamese.
    - Ensure no N/Nc/Np is skipped unless it’s fully subsumed by a larger phrase (e.g., "canxi" in "canxi hữu_cơ" should still be separate if it has distinct context).
    - Output MUST be valid JSON with proper UTF-8 encoding.
    Examples:
    - Tagged: "Khách_hàng/N khó_chịu/A vì/E trời/N mưa/V"
    Output: {{"terms": {{"khách_hàng": {{"knowledge": ["khách_hàng khó chịu"], "aliases": []}}, "trời mưa": {{"knowledge": ["trời mưa làm khách_hàng khó chịu"], "aliases": ["mưa"]}}, "trời": {{"knowledge": ["trời mưa"], "aliases": []}}, "mưa": {{"knowledge": ["mưa làm khách_hàng khó chịu"], "aliases": []}}}}}}
    - Tagged: "canxi/N từ/E tảo/N biển/N đỏ/A"
    Output: {{"terms": {{"canxi": {{"knowledge": ["canxi có trong Aquamin F"], "aliases": []}}, "canxi từ tảo biển": {{"knowledge": ["canxi từ tảo biển đỏ là nguồn canxi"], "aliases": ["tảo biển đỏ"]}}, "tảo": {{"knowledge": ["tảo là nguồn canxi"], "aliases": []}}, "biển": {{"knowledge": ["biển liên quan đến tảo"], "aliases": []}}}}}}
    - Tagged: "Nó/P đẹp/A lắm/T", Active Terms: {{"cái bàn": "term_cai_ban_123"}}
    Output: {{"terms": {{"cái bàn": {{"knowledge": ["cái bàn đẹp lắm"], "aliases": ["nó"]}}}}}}"""

    response = ""
    try:
        for chunk in LLM.invoke(prompt):
            if isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == 'content':
                response += chunk[1]
            else:
                logger.debug(f"Ignoring non-content chunk: {chunk}")
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        response = json.dumps({"terms": {}})

    response = clean_llm_response(response)
    logger.info(f"Raw LLM response: {response}")
    try:
        result = json.loads(response)
        terms_dict = result["terms"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"LLM parse error: {e}. Raw: '{response}'")
        terms_dict = {}

    # Step 4: Build piece and update state (no post-processing)
    piece = {
        "intent": intent,
        "topic": topics[0],
        "raw_input": latest_msg,
        "piece_id": f"piece_{uuid.uuid4()}",
        "meaningfulness_score": 0.9 if terms_dict else 0.5,
        "needs_clarification": not terms_dict,
        "term_refs": []
    }

    for term_name, data in terms_dict.items():
        canonical_name = term_name
        for existing_name in active_terms:
            if fuzz.ratio(term_name.lower(), existing_name.lower()) > 80:
                canonical_name = existing_name
                break
        term_id = active_terms.get(canonical_name, {}).get("term_id", f"term_{sanitize_vector_id(canonical_name)}_{convo_id}")
        active_terms[canonical_name] = {
            "term_id": term_id,
            "vibe_score": active_terms.get(canonical_name, {}).get("vibe_score", 1.0)
        }
        if term_id not in piece["term_refs"]:
            piece["term_refs"].append(term_id)
        
        state.setdefault("pending_knowledge", {}).setdefault(term_id, []).extend(
            [{"text": chunk, "confidence": 0.9, "source_piece_id": piece["piece_id"], 
              "created_at": datetime.now().isoformat(), "aliases": data["aliases"]} 
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

    if piece["needs_clarification"]:
        state["prompt_str"] = f"Ami không tìm thấy khái niệm nào trong '{latest_msg}'. Bạn có thể nói rõ hơn không?"

    return {"terms": terms_dict, "piece": piece}

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
            logger.info(f"Upserted {len(vectors)} terms to Pinecone: {', '.join(v['id'] for v in vectors)}")
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
    logger.info("Running updated recall_knowledge v2")
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent_result = detect_intent(state)
    intent = intent_result[0] if isinstance(intent_result, tuple) else intent_result
    
    query_embedding = EMBEDDINGS.embed_query(message)
    preset_results = index.query(vector=query_embedding, top_k=2, include_metadata=True, namespace="Preset")
    for r in preset_results["matches"]:
        if r.score > 0.8:
            logger.debug(f"Preset match found: {r.metadata['text']}")
            return {"knowledge": [{"text": r.metadata['text'], "source": "Preset", "score": r.score}], 
                    "mode": "Co-Pilot", "intent": intent, "terms": {}}

    convo_results = index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace="convo_nodes")
    matches_log = [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in convo_results["matches"]]
    logger.debug(f"Raw convo_results: {json.dumps(matches_log, ensure_ascii=False)}")
    now = datetime.now()
    nodes = []
    for r in convo_results["matches"]:
        meta = r.metadata
        try:
            pieces = json.loads(meta.get("pieces", "[]"))
            if not isinstance(pieces, list) or not pieces or all(not p for p in pieces):
                logger.debug(f"Skipping node {r.id} with invalid/empty pieces: {pieces}")
                continue
            meta["pieces"] = pieces
            meta["last_accessed"] = meta.get("last_accessed", now.isoformat())
            meta["access_count"] = meta.get("access_count", 0)
            days_since = (now - datetime.fromisoformat(meta["last_accessed"])).days
            vibe_score = (0.5 * r.score) + (0.3 * max(0, 1 - 0.05 * days_since)) + (0.2 * min(1, meta["access_count"] / 10))
            nodes.append({"meta": meta, "vibe_score": vibe_score})
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse pieces for node {r.id}: {e}")
            continue
    
    filtered_nodes = [n for n in nodes if n["meta"]["pieces"] and any(p for p in n["meta"]["pieces"])]
    if not nodes or not filtered_nodes:
        logger.info(f"No valid knowledge found for '{message}'")
        return {"knowledge": [], "mode": "Autopilot", "intent": intent, "terms": {}}
    
    filtered_nodes.sort(key=lambda x: (x["vibe_score"], datetime.fromisoformat(x["meta"]["last_accessed"])), reverse=True)
    top_nodes = filtered_nodes[:3]
    term_ids = set(t for n in top_nodes for p in n["meta"]["pieces"] for t in p.get("term_refs", []))
    
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
    
    knowledge = [{"meta": n["meta"], "vibe_score": n["vibe_score"]} for n in top_nodes]
    logger.debug(f"Returning knowledge: {json.dumps(knowledge, ensure_ascii=False)}")
    return {
        "knowledge": knowledge,
        "terms": {tid: tn["metadata"] for tid, tn in term_nodes.items()},
        "mode": "Autopilot" if intent == "request" else "Co-Pilot",
        "intent": intent
    }
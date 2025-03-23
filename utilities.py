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
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
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


def sanitize_vector_id(text):
    return text.replace(" ", "_").lower()


def clean_llm_response(response):
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].replace("'", '"')
    return cleaned

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



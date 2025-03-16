# utilities.py (scaled, full prompts, no hardcodes)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import json
import pinecone
from datetime import datetime
from pinecone_datastores import pinecone_index
import uuid
import os

# Config - Dynamic, no hardcodes
LLM = ChatOpenAI(model="gpt-4o", streaming=True)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")
index = pinecone_index  # Pre-configured Pinecone index

# 18 Categories from Ami_Blue_Print_3_0 (could be config-loaded later)
CATEGORIES = [
    "Skills", "Guidelines", "Lessons", "Products and Services", "Customer Personas and Behavior",
    "Objections and Responses", "Sales Scenarios and Context", "Feedback and Outcomes",
    "Ethical and Compliance Guidelines", "Industry and Market Trends", "Emotional and Psychological Insights",
    "Personalization and Customer History", "Metrics and Performance Tracking", "Team and Collaboration Dynamics",
    "Creative Workarounds", "Tools and Tech", "External Influences", "Miscellaneous"
]

INTENTS = ["greeting", "question", "casual", "teaching", "request", "exit", "humor", "challenge", "confusion"]

# Core Intent Detection (Full Prompt Restored)
def detect_intent(state):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    last_ami_msg = state.get("prompt_str", "")
    latest_msg = messages[-1].content if messages else ""
    
    prompt = f"""You’re Ami, a razor-sharp AI nailing human intent from text alone. Given:
    - Latest message: '{latest_msg}'
    - Last 3 user messages (or less): '{convo_history}'
    - Last Ami message: '{last_ami_msg}'
    Pick ONE intent from: {', '.join(f'"{i}"' for i in INTENTS)}. Return it in quotes (e.g., '"teaching"').

    Read the room:
    - "greeting": Opens the chat—warm, welcoming, fresh-start energy (e.g., "Hi!", "Yo, what’s up?").
    - "question": Seeks info—explicit ask, question mark, or clear curiosity (e.g., "How’s it work?", "What’s this?").
    - "casual": Hangs loose—short, neutral, no strong push or pull (e.g., "Cool", "Yeah, vibing").
    - "teaching": Drops knowledge—states facts, guides, explains, no imperative verb (e.g., "Trials flip skeptics", "Habits affect health").
    - "request": Demands action—direct, imperative verb, do-something-now vibe (e.g., "Call them", "Run the ad").
    - "exit": Shuts it down—farewell, end-of-line tone (e.g., "See ya", "Out").
    - "humor": Plays around—joking, silly, light-hearted flex (e.g., "Lol, nice one").
    - "challenge": Throws a gauntlet—tests or dares, competitive edge (e.g., "Bet you can’t", "Prove it").
    - "confusion": Signals lost vibes—unclear, garbled, "huh?" energy (e.g., "What?", "Huh??").

    Rules of the game:
    - Latest message is boss—judge its raw vibe first (e.g., imperative verb = "request", no verb + fact = "teaching").
    - Use prior 1-2 messages and Ami’s last reply for flow—boosts context but doesn’t override latest standalone intent unless tied (e.g., question-answer).
    - No assumptions—stick to text, no hidden vibes.
    - Any language, typos, emojis—decode the intent, not the words.
    - Stuck? Default "casual", log it, but fight to pick right.

    Think fast, think human."""
    
    
    intent = LLM.invoke(prompt).content.strip()
    print(f"Intent for convo '{convo_history}' with Ami context '{last_ami_msg}': {intent}")
    return intent

# Chunking Helper

def chunk_input(text, product_id, max_phrases=5):
    """Splits big inputs into chunks of max_phrases phrases."""
    # Strip existing product_id prefix if present
    if text.startswith(f"{product_id}:"):
        text = text[len(f"{product_id}:"):].strip()
    phrases = [p.strip() for p in text.split(',') if p.strip()]
    chunks = [phrases[i:i + max_phrases] for i in range(0, len(phrases), max_phrases)]
    return [f"{product_id}: {', '.join(chunk)}" for chunk in chunks]


def extract_knowledge(state, product_id=None):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    latest_msg = messages[-1].content if messages else ""
    intent = detect_intent(state)
    effective_product_id = product_id or "unknown"
    
    if len(latest_msg.split(',')) > 5:
        chunks = chunk_input(latest_msg, effective_product_id)
    else:
        chunks = [f"{effective_product_id}: {latest_msg}" if not latest_msg.startswith(f"{effective_product_id}:") else latest_msg]
    
    all_entities = []
    for chunk in chunks:
        prompt = f"""You’re Ami, extracting sales knowledge from text for product '{effective_product_id}' per Ami_Blue_Print_3_0. Given:
        - Latest message: '{chunk}'
        - Intent: '{intent}'
        - Last 3 messages: '{convo_history}'
        Return JSON: {{"entities": [{{"text": "{effective_product_id} - khai thác thông tin", "categories": [{{"name": "Skills", "confidence": 0.9}}], "linked_entities": ["tên, tuổi"]}}]}}
        - "entities": List—each with:
          - "text": Key phrase—'{effective_product_id} - ' prefix; split ingredients (e.g., "Bột xương cá tuyết") from benefits (e.g., "tăng trưởng chiều cao")
          - "categories": List—{{"name": category, "confidence": 0-1}} from 18: {', '.join(CATEGORIES)}
          - "linked_entities": Related phrases—empty [] if none
        Rules:
        - Valid JSON—{{"entities": []}} if no sales content.
        - Extract 2-5 phrases—always split ingredients (Products and Services) from benefits (Emotional and Psychological Insights) when tied (e.g., "Glucosamine giảm đau khớp" → "Glucosamine" and "giảm đau khớp").
        - NO translation—keep Vietnamese.
        - Skip fluff—e.g., "với", "giúp".
        - Intent guides:
          - "teaching": Ingredients (Products and Services), benefits (Emotional Psych), actions (Skills).
          - "question": Needs—terms (Emotional Psych), context (Products and Services).
          - "request": Actions (Skills).
          - "casual": Broad—sales terms.
        - Link entities: Tie ingredients to benefits and vice versa (e.g., "Bột xương cá tuyết" ↔ "tăng trưởng chiều cao") when tied in text.
        - Confidence: 0.9+ exact, 0.7-0.8 likely, <0.7 stretch.
        - Flexible—extract naturally, ensure product-specific clarity.
        """
        
        try:
            response = LLM.invoke(prompt).content.strip()
            print(f"Raw LLM response: '{response}'")
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            knowledge = json.loads(cleaned_response)
            if not isinstance(knowledge, dict) or "entities" not in knowledge:
                raise ValueError("Invalid JSON structure—missing 'entities'")
            all_entities.extend(knowledge["entities"])
            print(f"Extracted from '{chunk}': {json.dumps(knowledge, ensure_ascii=False)}")
        except Exception as e:
            default = {"entities": []}
            print(f"Extraction failed—error: {e}. Defaulting: {default}")
            all_entities.extend(default["entities"])
    
    return {"entities": all_entities}

def store_in_pinecone(intent, entities,product_id):
    vectors = []
    timestamp = datetime.now().isoformat()
    convo_id = str(uuid.uuid4())
    
    for entity in entities["entities"]:
        text = entity["text"]
        categories = [cat["name"] for cat in entity["categories"]]
        confidence = max(cat["confidence"] for cat in entity["categories"])
        vector_id = f"{intent}_{uuid.uuid4()}_{'_'.join(categories)}_{timestamp}"
        
        embedding = EMBEDDINGS.embed_query(text)
        metadata = {
            "text": text,
            "intent": intent,
            "categories": categories,
            "confidence": confidence,
            "product_id": product_id,
            "source": "Enterprise",
            "linked_entities": entity["linked_entities"],
            "convo_id": convo_id,
            "ingredient_flag": any(keyword in text.lower() for keyword in ["bột", "canxi", "collagen", "vitamin", "aquamin", "glucosamine", "msm"])
        }
        vectors.append((vector_id, embedding, metadata))
    
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Stored {len(vectors)} vectors for '{product_id}': {[v[0] for v in vectors]}")

def recall_knowledge(message, product_id=None):
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent = detect_intent(state)
    input_knowledge = extract_knowledge(state, product_id or "unknown")
    
    full_embedding = EMBEDDINGS.embed_query(message)
    filter_query = {"product_id": product_id} if product_id else None
    results = index.query(vector=full_embedding, top_k=20, include_metadata=True, filter=filter_query)
    print(f"Raw Pinecone hits: {json.dumps([r.metadata for r in results['matches']], ensure_ascii=False)}")
    
    matches = []
    input_cats = set(cat["name"] for item in input_knowledge["entities"] for cat in item["categories"])
    input_entities = set(item["text"] for item in input_knowledge["entities"]) | set(
        ent for item in input_knowledge["entities"] for ent in item["linked_entities"]
    )
    
    for result in results["matches"]:
        match_cats = set(result.metadata["categories"])
        match_entities = set(result.metadata["linked_entities"])
        cat_overlap = len(match_cats.intersection(input_cats))
        ent_overlap = len(match_entities.intersection(input_entities))
        source_boost = 0.1 if result.metadata.get("source") == "Enterprise" else 0
        ingred_boost = 0.1 if result.metadata.get("ingredient_flag", False) and "có gì" in message.lower() else 0
        score = result.score + (cat_overlap * 0.1) + (ent_overlap * 0.05) + source_boost + ingred_boost
        matches.append({
            "text": result.metadata["text"],
            "confidence": result.metadata["confidence"],
            "categories": result.metadata["categories"],
            "linked_entities": result.metadata["linked_entities"],
            "intent": result.metadata["intent"],
            "score": score,
            "source": result.metadata.get("source", "Enterprise")
        })
    
    matches = sorted(matches, key=lambda x: x["score"] * x["confidence"], reverse=True)[:5]
    if not matches:
        return {"response": "Ami đây! Chưa đủ info, bro thêm tí nha!", "mode": "Co-Pilot", "source": "Preset"}
    
    matches_str = json.dumps(matches, ensure_ascii=False, indent=2)
    prompt = f"""You’re Ami, reasoning the best response per Ami_Blue_Print_3_0 Selling Path. Given:
    - Input: '{message}'
    - Intent: '{intent}'
    - Relevant knowledge: {matches_str}
    Return JSON: {{"response": "Ami gợi ý...", "mode": "Co-Pilot", "source": "Enterprise"}}
    - "response": String—charming (e.g., "nha bro"); name product from matches (e.g., 'HITO_1'), suggest if 'unknown'; short, actionable.
    - "mode": "Autopilot" ("request"), "Co-Pilot" (others).
    - "source": "Enterprise" or "Preset" from top match.
    Rules:
    - Use matches—text, categories, linked_entities, intent.
    - Stick to matches—no extras.
    - Tone—casual, sales-y.
    - Link entities if fit—e.g., "với Glucosamine".
    - Intent—"question" → suggest, "request" → direct.
    - No fluff—punchy, predictive.
    """
    

    response = LLM.invoke(prompt).content.strip()
    cleaned_response = response.replace("```json", "").replace("```", "").strip()
    result = json.loads(cleaned_response)
    print(f"LLM reasoned: {json.dumps(result, ensure_ascii=False)}")
    return result

def update_product_memory(product_id, new_input):
    index.delete(filter={"product_id": product_id})
    print(f"Deleted old vectors for '{product_id}'")
    
    state = {"messages": [HumanMessage(new_input)], "prompt_str": ""}
    intent = detect_intent(state)
    knowledge = extract_knowledge(state, product_id)
    store_in_pinecone(intent, knowledge, new_input, product_id)

def test_ami():
    print("Testing Ami—scaled up!")
    
    hito1_input = ""
    hito2_input = ""
    state1 = {"messages": [HumanMessage(hito1_input)], "prompt_str": ""}
    state2 = {"messages": [HumanMessage(hito2_input)], "prompt_str": ""}
    intent1 = detect_intent(state1)
    knowledge1 = extract_knowledge(state1, "HITO_1")
    store_in_pinecone(intent1, knowledge1, hito1_input, "HITO_1")
    #store_in_pinecone(intent2, knowledge2, hito2_input, "HITO_2")
    
    #result1 = recall_knowledge("Con gái tôi 13 tuổi muốn cao lên")
    #result2 = recall_knowledge("Sản phẩm tốt cho khớp không?", "HITO_2")
    #print(f"Test 1 result: {json.dumps(result1, ensure_ascii=False)}")
    #print(f"HITO_2 result: {json.dumps(result2, ensure_ascii=False)}")
    
    #update_input = "HITO_1 giờ có thêm Vitamin K2 tăng mật độ xương"
    #update_product_memory("HITO_1", update_input)
    #result3 = recall_knowledge("HITO_1 có gì mới?", "HITO_1")
    #print(f"HITO_1 updated result: {json.dumps(result3, ensure_ascii=False)}")
    teach_prompt = "## Summaries of Knowledge I've Been Taught\n\n"
    teach_prompt += "Here’s what I’ve learned from you, neatly summarized and categorized:\n\n"
    for entity in knowledge1["entities"]:
        text = entity["text"]
        category = entity["categories"][0]["name"]
        # Simple summary: restating the text under its category
        teach_prompt += f"- **{text}** (in *{category}*): This relates to {text}.\n"
    teach_prompt += "\nPretty cool, right? I’m ready to show off this knowledge to any human!"
    print(teach_prompt)
if __name__ == "__main__":
    test_ami()
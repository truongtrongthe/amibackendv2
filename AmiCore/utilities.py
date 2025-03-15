# utilities.py (evolving)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import json
import pinecone
from datetime import datetime
from pinecone_datastores import pinecone_index
import uuid

# Config - Locked and loaded
LLM = ChatOpenAI(model="gpt-4o", streaming=True)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")  # Assumes API key set elsewhere

index = pinecone_index  # Your pre-configured Pinecone index

# 18 Categories from Ami_Blue_Print_3_0
CATEGORIES = [
    "Skills", "Guidelines", "Lessons", "Products and Services", "Customer Personas and Behavior",
    "Objections and Responses", "Sales Scenarios and Context", "Feedback and Outcomes",
    "Ethical and Compliance Guidelines", "Industry and Market Trends", "Emotional and Psychological Insights",
    "Personalization and Customer History", "Metrics and Performance Tracking", "Team and Collaboration Dynamics",
    "Creative Workarounds", "Tools and Tech", "External Influences", "Miscellaneous"
]

# Simple translation dictionary (expand as needed)
TRANSLATIONS = {
    "need": "cần",
    "me": "tôi",
    "what": "cái gì",
    "hey": "này"
}

# Core Intent Detection (Unchanged)
def detect_intent(state):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    last_ami_msg = state.get("prompt_str", "")
    latest_msg = messages[-1].content if messages else ""
    
    prompt = f"""You’re Ami, a razor-sharp AI nailing human intent from text alone. Given:
    - Latest message: '{latest_msg}'
    - Last 3 user messages (or less): '{convo_history}'
    - Last Ami message: '{last_ami_msg}'
    Pick ONE intent from: "greeting", "question", "casual", "teaching", "request", "exit", "humor", "challenge", "confusion". Return it in quotes (e.g., '"teaching"').

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


def extract_knowledge(state):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages]) if messages else ""
    latest_msg = messages[-1].content if messages else ""
    intent = detect_intent(state)
    prompt = f"""You’re Ami, extracting sales knowledge from text for 18 categories per Ami_Blue_Print_3_0. Given:
    - Latest message: '{latest_msg}'
    - Intent: '{intent}'
    - Last 3 messages: '{convo_history}'
    Return JSON: {{"entities": [{{"text": "khai thác thông tin cá nhân", "categories": [{{"name": "Skills", "confidence": 0.9}}], "linked_entities": ["tên, tuổi"]}}]}}
    - "entities": List of dicts—each with:
      - "text": Key phrase—sales-relevant unit (e.g., action, object, context, insight)
      - "categories": List of {{"name": category, "confidence": 0-1}} from 18: {', '.join(CATEGORIES)}
      - "linked_entities": List of related phrases—empty [] if none
    Rules:
    - MUST return valid JSON—{{"entities": []}} if nothing extracted.
    - Extract 2-5 sales-relevant phrases—any key terms or phrases based on intent and text.
    - NO translation—keep Vietnamese (e.g., "thông tin cá nhân").
    - Skip fluff—e.g., "với", "vì", "cần".
    - Intent guides extraction:
      - "teaching": Extract sales lessons—e.g., actions (Skills), objects (Personalization), context (Personas), insights (Lessons, Emotional Psych).
      - "question": Focus on needs—e.g., "làm gì" (Skills), "ai" (Personas).
      - "request": Focus on actions—e.g., "gọi" (Skills).
      - "casual": Broad—key sales terms without strong push.
    - Link entities: Group related phrases if clearly tied in text—e.g., "khai thác" → "tên, tuổi"—otherwise leave unlinked.
    - Confidence: 0.9+ for dead-on intent match, 0.7-0.8 likely, <0.7 stretch.
    - Use context from history to resolve refs (e.g., "nó" → prior term).
    - Flexible—don’t assume structure (e.g., action + list); extract what’s sales-relevant naturally.
    """
    try:
        response = LLM.invoke(prompt).content.strip()
        print(f"Raw LLM response: '{response}'")  # Debug raw output
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        knowledge = json.loads(cleaned_response)
        if not isinstance(knowledge, dict) or "entities" not in knowledge:
            raise ValueError("Invalid JSON structure—missing 'entities'")
        print(f"Extracted from '{latest_msg}': {json.dumps(knowledge, ensure_ascii=False)}")
        return knowledge
    except Exception as e:
        default = {"entities": []}
        print(f"Extraction failed—error: {e}. Raw response: '{response}'. Defaulting: {default}")
        return default

def store_in_pinecone(intent, entities, convo_history):
    """Stores linked entities in Pinecone per Ami_Blue_Print_3_0."""
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
            "source": "Enterprise",
            "linked_entities": entity["linked_entities"],
            "convo_id": convo_id
        }
        vectors.append((vector_id, embedding, metadata))
    
    if not vectors:
        print("No vectors to store—skipping Pinecone upsert.")
        return
    index.upsert(vectors=vectors)
    print(f"Stored {len(vectors)} vectors: {[v[0] for v in vectors]}")

def recall_knowledge(message):
    """Recalls all relevant knowledge, lets LLM reason for Ami_Blue_Print_3_0."""
    state = {"messages": [HumanMessage(message)], "prompt_str": ""}
    intent = detect_intent(state)
    input_intent = intent.strip()
    print(f"Debug: Recall Intent = {input_intent}")
    input_knowledge = extract_knowledge(state)
    
    # Broad Pinecone query
    full_embedding = EMBEDDINGS.embed_query(message)
    results = index.query(vector=full_embedding, top_k=10, include_metadata=True)
    print(f"Raw Pinecone hits: {json.dumps([r.metadata for r in results['matches']], ensure_ascii=False, indent=2)}")  # Debug hits
    
    # Score relevance
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
        score = result.score + (cat_overlap * 0.1) + (ent_overlap * 0.05) + source_boost
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
    print(f"Scored matches: {json.dumps(matches, ensure_ascii=False, indent=2)}")  # Debug matches
    if not matches:
        return {"response": "Ami đây! Chưa đủ info, bro thêm tí nha!", "mode": "Co-Pilot", "source": "Preset"}
    
    # LLM reasoning
    matches_str = json.dumps(matches, ensure_ascii=False, indent=2)
    prompt = f"""You’re Ami, reasoning the best response per Ami_Blue_Print_3_0 Selling Path. Given:
    - Input: '{message}'
    - Intent: '{input_intent}'
    - Relevant knowledge: {matches_str}
    Return JSON: {{"response": "Ami gợi ý...", "mode": "Co-Pilot", "source": "Enterprise"}}
    - "response": String—best answer, charming tone (e.g., "nha bro", "thử sao nổi").
    - "mode": "Autopilot" (direct, for "request") or "Co-Pilot" (suggestive, for others).
    - "source": "Enterprise" or "Preset" from top match.
    Rules:
    - Pick the most relevant knowledge—use text, categories, linked_entities, and intent match.
    - Stick strictly to provided matches—only use text and linked_entities from matches, no extra info.
    - Flex tone—casual, sales-y, match input vibe (e.g., "nha bro").
    - Use linked_entities if they fit—e.g., "như tên, tuổi".
    - Adapt to intent—e.g., "question" → suggest, "request" → direct.
    - No fluff—actionable, predictive, blueprint-style.
    """
    
    response = LLM.invoke(prompt).content.strip()
    cleaned_response = response.replace("```json", "").replace("```", "").strip()
    result = json.loads(cleaned_response)
    print(f"LLM reasoned: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result

def test_ami():
    print("Yo! Testing Ami—let’s roll one round.")
    
    # Training session
    training_input = "Với khách hàng mới, cần khéo léo khai thác thông tin cá nhân như: tên, tuổi, giới tính, chiều cao, cân nặng và thói quen sinh hoạt vì các yếu tố này ảnh hưởng tới sức khoẻ sinh lý"
    #state = {"messages": [HumanMessage(training_input)], "prompt_str": ""}
    #intent = detect_intent(state)
    #knowledge = extract_knowledge(state)
    #store_in_pinecone(intent, knowledge, training_input)
    
    # Recall test
    test_input = "yếu tố ảnh hưởng tới sức khoẻ sinh lý?"
    result = recall_knowledge(test_input)
    print(f"Final result for '{test_input}': {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    test_ami()
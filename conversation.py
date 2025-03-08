import supabase
import os
from openai import OpenAI
import json
from datetime import datetime
import uuid
from numpy import dot
from numpy.linalg import norm
from pinecone_datastores import pinecone_index

SIMILARITY_THRESHOLD = 0.69
client = OpenAI()
def generate_embedding(skill_text):
    try:
        response = client.embeddings.create(
        input=skill_text,
        model="text-embedding-3-small",
        dimensions=1536  # Match the dimensions of ada-002
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return None  # Avoid breaking downstream logic
convo_history = []

SEEDED_STATES = ["Info Gathering", "Intent Probing", "Product Pitching", "Trust Building", "Handling Objections", "Closing"]
profile = {"name": None, "age": None, "gender": None, "wishes": None, "shipping_address": None}
STATE_SEQUENCE = {
    "Info Gathering": "Intent Probing",
    "Intent Probing": "Product Pitching",
    "Product Pitching": "Trust Building",
    "Trust Building": "Closing",
    "Handling Objections": "Closing",
    "Closing": "Closing"
}

REASONING_PROMPT = """
You are Ami, a sales assistant aiming to close a sale. Given the current state, customer message, and conversation history, 
determine the next state from these options: {states}. The end goal is 'Closing'. Return a JSON object with:
- "next_state": The next state (one of {states}).
- "reason": A brief explanation.

Current State: {current_state}
Customer Message: {message}
Conversation History: {history}
"""

def reason_next_state(current_state, message, history):
    history_str = json.dumps([{"message": h["message"], "state": h["state"]} for h in history], indent=2)
    full_prompt = REASONING_PROMPT.format(
        states=json.dumps(SEEDED_STATES),
        current_state=current_state if current_state else "None",
        message=message,
        history=history_str
    )
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": full_prompt}],
        )
        result = json.loads(response.choices[0].message.content)
        if (current_state == "Info Gathering" or not current_state) and not profile_complete():
            return "Info Gathering", "Profile incomplete, need more info"
        return result["next_state"], result["reason"]
    
    except Exception as e:
        print(f"Reasoning failed: {e}")
        return "Info Gathering", "Fallback due to error"

def update_profile(customer_message):
    message_lower = customer_message.lower().strip()
    parts = [p.strip() for p in message_lower.split(",")]
    greetings = ["hi", "hello", "hey", "chào"]
    
    # Name: Set only if not set, avoid overwriting with non-names
    if profile["name"] is None:
        for part in parts:
            if part.isalpha() and len(part) > 2 and part not in greetings:
                profile["name"] = part.capitalize()
                break
    
    # Age: Look for digits or "years old"
    for part in parts:
        if any(char.isdigit() for char in part) and ("year" in part or "tuổi" in part or part.isdigit()):
            profile["age"] = ''.join(filter(str.isdigit, part))
            break
    
    # Gender: Check keywords
    gender_keywords = {"male": "male", "female": "female", "man": "male", "woman": "female", "nam": "male", "nữ": "female"}
    for part in parts:
        if part in gender_keywords:
            profile["gender"] = gender_keywords[part]
            break
    
    # Wishes: Check growth-related keywords
    wish_keywords = ["grow", "height", "taller", "cm", "cao"]
    if any(w in message_lower for w in wish_keywords):
        profile["wishes"] = "grow height" if "cm" not in message_lower else f"grow {message_lower.split()[1]}"
    
    # Shipping address: Set only in Closing or if explicit
    if "address" in message_lower or (len(message_lower) > 10 and profile["name"] is not None and "cm" not in message_lower):
        profile["shipping_address"] = customer_message.strip()

def profile_complete():
    required = ["name", "age", "gender", "wishes"]
    return all(profile[key] is not None for key in required)

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def process_message(customer_message):
    
    update_profile(customer_message)
    
    message_embedding = generate_embedding(customer_message)
    # Query Pinecone
    query_result = pinecone_index.query(vector=message_embedding, 
                                        top_k=5, 
                                        include_metadata=True,
                                        include_values=True)
    matches = query_result["matches"]
    
    # Determine current state via Pinecone
    current_state = None
    best_match = None
    best_score = 0
    for match in matches:
        seeded_state = match["metadata"]["state"]
        if seeded_state in SEEDED_STATES:
            similarity = cosine_similarity(message_embedding, match["values"])
            if similarity > SIMILARITY_THRESHOLD and similarity > best_score:
                best_match = match
                best_score = similarity
                current_state = seeded_state
    
    # Reason next state with LLM
    next_state, reason = reason_next_state(current_state, customer_message, convo_history)
    print(f"Reasoning: {reason}")  # Debug output
    
    # Pull knowledge for next state
    relevant_knowledge = [m["metadata"] for m in matches if m["metadata"]["state"] == next_state and cosine_similarity(message_embedding, m["values"]) > SIMILARITY_THRESHOLD]
    if not relevant_knowledge:
        fallback_match = next((m["metadata"] for m in matches if m["metadata"]["state"] == next_state), None)
        relevant_knowledge = [fallback_match] if fallback_match else [{"text": f"Let’s move to {next_state}", "type": "Skill", "context": "general", "state": next_state}]
    
    convo_history.append({"message": customer_message, "state": next_state, "knowledge": relevant_knowledge})
    return next_state, relevant_knowledge
    

RESPONSE_PROMPT = """
You are Ami, a sales assistant. Your goal is to close the sale. 
Given the current state, customer message, conversation history, profile, and relevant knowledge, craft a natural, conversational response (50-70 words max) in Vietnamese. 
In 'Info Gathering', ask for missing profile info (name, age, gender, wishes). In 'Closing', ask for shipping address if missing. Return the response as a string.
Use the knowledge as a base, adapt it to the context, and guide toward 'Closing'. Return the response as a string. 

Current State: {state}
Customer Message: {message}
Conversation History: {history}
Profile: {profile}
Relevant Knowledge: {knowledge}
"""

def generate_response(state, customer_message, convo_history, relevant_knowledge):
    history_str = json.dumps([{"message": h["message"], "state": h["state"]} for h in convo_history[:-1]], indent=2)  # Exclude current
    knowledge_str = json.dumps(relevant_knowledge, indent=2)
    profile_str = json.dumps(profile, indent=2)
    full_prompt = RESPONSE_PROMPT.format(
        state=state,
        message=customer_message,
        history=history_str,
        profile=profile_str,
        knowledge=knowledge_str
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.5  # Slightly creative but consistent
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Response generation failed: {e}")
        return "Let’s keep this moving—how can I assist you next?"

# Simple convo loop to test Step 2
def run_conversation():
    print("\nStarting conversation (type 'exit' to stop):")
    global profile  # Reset profile for each convo
    profile = {"name": None, "age": None, "gender": None, "wishes": None, "shipping_address": None}
    while True:
        customer_message = input("Customer: ").strip()
        if customer_message.lower() == "exit":
            break
        
        state, knowledge = process_message(customer_message)
        response = generate_response(state, customer_message, convo_history, knowledge)
        print(f"State: {state}")
        print(f"Knowledge: {knowledge}")
        print(f"Profile: {profile}")
        print(f"Ami: {response}")
        print("-" * 50)

# Main execution
if __name__ == "__main__":
    # Uncomment to re-run Step 1 if needed
    # setup_knowledge_base()
    
    # Run Step 2
    run_conversation()
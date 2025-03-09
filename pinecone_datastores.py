import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "ami-dev"

llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)


# Kiểm tra và tạo index nếu chưa có
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)


# Use your pinecone_index
PINECONE_INDEX = pinecone_index

# Recall from Pinecone (v15.2 - Query-Driven Final)
def recall_from_pinecone(query, user_id, user_lang):
    original_query = query
    triggers_en = ["tell me about", "how to", "what is"]
    triggers_vi = ["cho tôi biết về", "làm thế nào để", "là gì"]
    trigger_map = {
        "cho tôi biết về": "tell me about",
        "làm thế nào để": "how to",
        "là gì": "what is"
    }
    translation_map = {
        "sales": "sales",
        "trò chuyện với khách hàng": "chat with customers",
        "re-engage customer ghosting": "re-engage customer ghosting",
        "re-engage ghosting customer": "re-engage customer ghosting",
        "khách hàng mất liên lạc": "customer ghosting"
    }
    input_lower = query.lower()
    print(f"Input: '{input_lower}'")
    english_trigger = ""
    for trigger in triggers_en + triggers_vi:
        print(f"Checking trigger: '{trigger}'")
        if trigger in input_lower:
            print(f"Trigger matched: '{trigger}'")
            english_trigger = trigger_map.get(trigger, trigger)
            break
    else:
        print("No trigger matched")
        english_trigger = ""
    
    # Strip trigger cleanly
    query_clean = input_lower.replace(english_trigger, "").strip() if english_trigger else input_lower
    english_query = f"{trigger_map.get(english_trigger, english_trigger)} {translation_map.get(query_clean, query_clean)}".strip()
    print(f"Pre-padded query: '{query_clean}'")
    print(f"English query: '{english_query}'")
    query_embedding = embeddings.embed_query(english_query)
    print(f"Embedding snippet: {query_embedding[:5]}")
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter={"user_id": user_id}
    )
    print(f"Raw Pinecone results: {results}")
    
    # Top 3 by score—query drives it
    matches = [m for m in results.get("matches", []) if m["score"] > 0.4]
    final_matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:3]
    
    print(f"Match scores: {[m['score'] for m in final_matches]}")
    
    response = "Here’s what I remember:" if user_lang == "en" else "Đây là những gì tôi có từ trí nhớ:"
    matches_text = ""
    for i, match in enumerate(final_matches, 1):
        text = match["metadata"]["text"]
        vibe = match["metadata"]["vibe"]
        if user_lang == "vi":
            text = llm.invoke(f"Translate to Vietnamese: '{text}'").content.strip()
        response += f" {i}. '{text}' ({vibe})"
        matches_text += f"{i}. '{text}' ({vibe})\n"
    
    if final_matches:
        # Tip flexes to query intent
        tip = llm.invoke(
            f"Given these matches: {matches_text}\n"
            f"Write an answer for the user's query: '{original_query}'—keep it direct and actionable. Answer in {user_lang}"
        ).content.strip()
        response += f"\nTip: {tip}"
    else:
        tip = "My brain is empty—teach me something about it!" if user_lang == "en" else "Ami chưa có kiến thức này. Hãy dạy tôi điều gì đó về nó!"
        response = tip
    
    return response

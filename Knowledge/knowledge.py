import json
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="o3-mini")  # No streaming
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

from pinecone_datastores import pinecone_index
# LLM prompt for parsing


PARSE_PROMPT_OK = """
Extract knowledge from this sentence as a list of JSON objects, one for each distinct action or fact. For each:
- "text": The core action or knowledge, concise (e.g., "talk gently to query their information").
- "type": One of [Skill, Product Info, Combo, Promotion] (default: Skill).
- "context": Specific situation or condition it applies to. Based on sentence clues (e.g., "initial contact", "price objection"), default "none" if unclear.
- "state": One of [Info Gathering, Intent Probing, Product Pitching, Trust Building, Handling Objections, Closing] or null if unspecified. Match intent: "query info" to Info Gathering, "find intent" to Intent Probing, "believe/trust" to Trust Building, "push/close" to Closing.

Split multi-step sentences into separate items based on actions or conditions (e.g., "first X, then Y" → two items). Don't summarize, keep text exact.

Sentence: {sentence}
Return: List of JSON objects, e.g., [
    {{"text": "talk gently to query their information", "type": "Skill", "context": "communication", "state": "Info Gathering"}},
    {{"text": "find their hidden intent", "type": "Skill", "context": "communication", "state": "Intent Probing"}}
]
"""
PARSE_PROMPT_Good = """
Extract knowledge from this sentence as a list of JSON objects, one for each distinct action or fact. For each:
- "text": The core action or knowledge, concise (e.g., "gently ask for their information").
- "type": One of [Skill, Product Info, Combo, Promotion] (default: Skill).
- "context": Specific situation or condition it applies to. Use precise terms:  
  - "initial contact" for first interactions  
  - "need exploration" for understanding intent or deeper needs  
  - "clear buying signal" for final closing stage  
  Default to "none" if unclear.
- "state": One of [Info Gathering, Intent Probing, Product Pitching, Trust Building, Handling Objections, Closing] or null if unspecified. Match intent:  
  - "ask for information" → Info Gathering  
  - "find intent" → Intent Probing  
  - "build trust" → Trust Building  
  - "push/close" → Closing  

Split multi-step sentences into separate items based on actions or conditions (e.g., "first X, then Y" → two items). Don't summarize, keep text exact.

Sentence: {sentence}  
Return: List of JSON objects, e.g., [
    {{"text": "gently ask for their information", "type": "Skill", "context": "initial contact", "state": "Info Gathering"}},
    {{"text": "find their hidden intent", "type": "Skill", "context": "need exploration", "state": "Intent Probing"}}
]
"""

PARSE_PROMPT = """
Extract knowledge from this sentence as a list of JSON objects, one for each distinct action or fact. For each:
- "text": The core action or knowledge, concise (e.g., "talk gently to query their information").
- "type": One of [Skill, Product Info, Combo, Promotion] (default: Skill).
- "context": Specific situation or condition it applies to, based on sentence clues or explicit "with context [specific]" (e.g., "initial contact", "price objection"), required—use "general" if unclear, not "none".
- "state": One of [Info Gathering, Intent Probing, Product Pitching, Trust Building, Handling Objections, Closing] or null if unspecified. Match intent precisely: "query info" → Info Gathering, "find intent" → Intent Probing, "believe/trust" → Trust Building, "push/close" → Closing.

Split multi-step sentences into separate items based on actions or conditions (e.g., "first X, then Y" → two items). Don’t summarize, keep text exact. If "with context [specific]" is present, use [specific] as the context.

Sentence: {sentence}
Return: List of JSON objects, e.g., [
    {{"text": "talk gently to query their information", "type": "Skill", "context": "initial contact", "state": "Info Gathering"}},
    {{"text": "find their hidden intent", "type": "Skill", "context": "needs exploration", "state": "Intent Probing"}}
]
"""

# Function to parse with LLM
def parse_knowledge_sentence(sentence):
    full_prompt = PARSE_PROMPT.format(sentence=sentence)
    try:
        response = llm.invoke(full_prompt)
        parsed_items = json.loads(response.content)
        if not isinstance(parsed_items, list):
            parsed_items = [parsed_items]  # Ensure it’s a list
        return parsed_items
    except Exception as e:
        print(f"LLM parsing failed: {e}")
        return [{
            "text": sentence,  # Fallback: use full sentence
            "type": "Skill",
            "context": "none",
            "state": None
        }]

# Function to collect knowledge from user
def collect_knowledge_from_user():
    knowledge_base = []
    print("Enter training or HITO info as a sentence (type 'done' to finish):")
    print("Example: 'When they say it's pricey, show ROI as a skill in Handling Objections'")
    
    while True:
        sentence = input("Your sentence: ").strip()
        if sentence.lower() == "done":
            break
        
        # Parse with LLM
        parsed_items = parse_knowledge_sentence(sentence)
        
        # Fallback prompt if text is empty
        for item in parsed_items:
            if not item["text"]:
                item["text"] = input(f"Couldn't parse text for one item in '{sentence}', please clarify: ").strip()
            knowledge_base.append(item)
            print("Added:", item)
    
    return knowledge_base

# Function to ingest knowledge into Pinecone

def ingest_knowledge(knowledge_base):
    for item in knowledge_base:
        # Generate embedding with OpenAI        
        kb_embeding = embeddings.embed_query(item["text"])
        # Create Pinecone entry
        pinecone_id = str(uuid.uuid4())  # Unique ID
        metadata = {
            "text": item["text"],  
            "type": item["type"],
            "context": item["context"],
            "state": item["state"] if item["state"] is not None else ""
        }
        
        # Upsert to Pinecone
        pinecone_index.upsert([(pinecone_id, kb_embeding, metadata)])
        print(f"Ingested: {item['text']}")

# Main ingestion process
def setup_knowledge_base():
    knowledge_base = collect_knowledge_from_user()
    if knowledge_base:
        ingest_knowledge(knowledge_base)
        print("Knowledge base ingested successfully!")
    else:
        print("No knowledge added.")

def behappy():
    arrayx =[
        "first:",
        "I love you!"
    ]
    print(arrayx[0])

# Run setup
if __name__ == "__main__":
    setup_knowledge_base()
    #behappy()



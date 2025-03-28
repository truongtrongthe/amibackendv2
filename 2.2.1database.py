# database.py
# Built by: The Fusion Lab
# Date: March 28, 2025

import os
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger
from datetime import datetime
import uuid
import asyncio
import json
import re

from typing import Dict, List

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
ami_index_name = os.getenv("PRESET")
ent_index_name = os.getenv("ENT")
ami_index = pc.Index(ami_index_name)
ent_index = pc.Index(ent_index_name)

inferLLM = ChatOpenAI(model="gpt-4o", streaming=False)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

def index_name(index) -> str:
    return "ami_index" if index == ami_index else "ent_index"

AI_NAME = None

async def infer_categories(input: str, context: str = "") -> Dict:
    category_prompt = f"""
    Conversation Context: '{context}'
    Latest Input: '{input}'
    Task: Identify categories per Star Net 2.2 Blueprint.
    - 'categories': English terms (e.g., "primary": "curiosity", "special": "character").
    - 'labels': Bilingual pairs (Vietnamese, English).
    Examples: 'Học Hỏi' (Learning), 'Kỹ Năng Bán Hàng' (Sales Skills), 'Tính Cách' (character).

    Rules:
    - Name: If input explicitly sets AI name (e.g., "You're <name>", "em hãy nhớ em tên là <name>", "xưng là <name>"):
      - "primary": "name", "special": "character".
      - Label: "Tên" (Name).
    - Instincts: If input defines AI behavior with clear directives (e.g., 'be curious', 'always be truthful', 'should be polite'):
      - "special": "character".
      - "primary": Extract traits from the directive (e.g., "curiosity", "truthfulness, politeness").
      - Label: "Tính Cách" (character) per trait.
    - Knowledge: If input provides a fact, method, or explanation not tied to AI behavior:
      - "primary": Identify the main topic (e.g., "sales skills", "curiosity").
      - "special": "description" if it explains or defines (e.g., contains "means", "is", "là", or a colon ":"), "" if procedural (e.g., "cần", "do this").
      - Label: Use the topic in Vietnamese and English (e.g., "Tò Mò" (Curiosity), "Kỹ Năng Bán Hàng" (Sales Skills)).
    - Unclear: If none apply, set 'needs_clarification': true.

    Guidelines:
    - Favor 'description' for statements with explanatory intent (e.g., "X là Y" or "X is Y") over 'character' unless a directive is explicit.
    - Extract 'primary' dynamically from the input’s subject, not a fixed list.
    - Avoid assuming traits unless a directive is present.

    Return JSON:
    {{
      "categories": {{"primary": "<topic>", "special": "<type>"}},
      "labels": [{{"original": "<viet>", "english": "<eng>", "requires_context": true/false}}],
      "needs_clarification": true/false
    }}
    """
    
    try:
        response = await asyncio.to_thread(inferLLM.invoke, category_prompt)
        raw_content = response.content.strip()
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            raw_content = match.group(0)
        parsed = json.loads(raw_content)
        return {
            "categories": parsed.get("categories", {"primary": "uncategorized", "special": ""}),
            "labels": parsed.get("labels", [{"original": "Chưa Phân Loại", "english": "Uncategorized", "requires_context": False}]),
            "needs_clarification": parsed.get("needs_clarification", False),
            "requires_context": any(label["requires_context"] for label in parsed.get("labels", []))
        }
    except Exception as e:
        logger.warning(f"Inference failed: {e}. Defaulting.")
        return {
            "categories": {"primary": "uncategorized", "special": ""},
            "labels": [{"original": "Chưa Phân Loại", "english": "Uncategorized", "requires_context": False}],
            "needs_clarification": False,
            "requires_context": False
        }

async def save_training(input: str, user_id: str, context: str = "", mode: str = "default") -> bool:
    global AI_NAME
    embedding = EMBEDDINGS.embed_query(input)
    data = await infer_categories(input, context)
    ns = "wisdom_bank"
    
    if data["needs_clarification"]:
        logger.warning(f"Input '{input}' needs clarification.")
        return False
    
    categories = data["categories"]
    target_index = ami_index if mode == "pretrain" else ent_index
    
    if categories["primary"] == "name":
        name_response = await asyncio.to_thread(inferLLM.invoke, f"Extract name from '{input}'. Return only the name.")
        AI_NAME = name_response.content.strip()
        logger.debug(f"Set AI_NAME to: {AI_NAME}")
    
    existing = await asyncio.to_thread(
        target_index.query,
        vector=embedding,
        top_k=1,
        include_metadata=True,
        namespace=ns,
        filter={
            "user_id": user_id,
            "categories_primary": categories["primary"],
            "categories_special": categories.get("special", ""),
            "raw": input
        }
    )
    if existing.get("matches", []) and existing["matches"][0]["score"] > 0.99:
        logger.info(f"Skipping duplicate: '{input}' already exists as {existing['matches'][0]['id']}")
        return True
    
    if categories["special"] == "character" and "," in categories["primary"]:
        traits = [t.strip() for t in categories["primary"].split(",")]
        trait_map = {"curiosity": "curious", "truthfulness": "truthful"}
        for trait in traits:
            adjective = trait_map.get(trait, trait)
            instinct_input = f"{AI_NAME}, always be {adjective}" if AI_NAME else f"Always be {adjective}"
            metadata = {
                "created_at": datetime.now().isoformat(),
                "raw": instinct_input,
                "confidence": 0.9,
                "source": "user",
                "categories_primary": trait,
                "categories_special": "character",
                "user_id": user_id,
                "labels": json.dumps([{"original": "Tính Cách", "english": "Character", "requires_context": False}])
            }
            convo_id = f"{user_id}_{uuid.uuid4()}"
            try:
                target_index.upsert([(convo_id, embedding, metadata)], namespace=ns)
                logger.info(f"Saved instinct '{instinct_input}' to {index_name(target_index)}: {convo_id}")
            except Exception as e:
                logger.error(f"Upsert failed for '{instinct_input}': {e}")
                return False
    else:
        metadata = {
            "created_at": datetime.now().isoformat(),
            "raw": input,
            "confidence": 0.9 if categories["special"] == "character" else 0.85,
            "source": "user",
            "categories_primary": categories["primary"],
            "categories_special": categories.get("special", ""),
            "labels": json.dumps(data["labels"]),
            "user_id": user_id
        }
        if categories["primary"] == "name":
            metadata["name"] = AI_NAME
        
        convo_id = f"{user_id}_{uuid.uuid4()}"
        try:
            target_index.upsert([(convo_id, embedding, metadata)], namespace=ns)
            logger.info(f"Saved to {index_name(target_index)}: {convo_id} - Categories: {categories}")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return False
    
    return True

async def load_instincts(user_id: str) -> Dict[str, str]:
    global AI_NAME
    ns = "wisdom_bank"
    instincts = {}
    
    for index in [ami_index, ent_index]:
        try:
            results = index.query(
                vector=[0] * 1536,
                top_k=100,
                include_metadata=True,
                namespace=ns,
                filter={"user_id": user_id, "categories_special": "character"}
            )
            matches = results.get("matches", [])
            logger.debug(f"Instinct matches from {index_name(index)}: {matches}")
            for match in matches:
                primary = match["metadata"]["categories_primary"]
                raw_text = match["metadata"]["raw"].lower()
                if primary == "name":
                    if AI_NAME is None:
                        AI_NAME = match["metadata"].get("name", raw_text.strip())
                        logger.debug(f"Set AI_NAME to: {AI_NAME}")
                elif "always be" in raw_text and "means" not in raw_text and "is" not in raw_text and "là" not in raw_text:
                    if primary not in instincts or match["metadata"]["created_at"] > instincts[primary]["created_at"]:
                        logger.debug(f"Updating {primary}: {match['metadata']['raw']} ({match['metadata']['created_at']}) over {instincts.get(primary, {}).get('raw', 'none')} ({instincts.get(primary, {}).get('created_at', 'none')})")
                        instincts[primary] = {
                            "raw": match["metadata"]["raw"],
                            "created_at": match["metadata"]["created_at"]
                        }
                else:
                    logger.debug(f"Skipping non-instinct: {match['metadata']['raw']} ({match['metadata']['created_at']})")
        except Exception as e:
            logger.error(f"Instinct load failed in {index_name(index)}: {e}")
    
    instincts_dict = {k: v["raw"] for k, v in instincts.items()}
    logger.info(f"Loaded instincts: {instincts_dict} for user {user_id}. AI Name: {AI_NAME}")
    return instincts_dict

async def find_knowledge(user_id: str, primary: str, special: str = "description", top_k: int = 1) -> List[Dict]:
    ns = "wisdom_bank"
    knowledge = []
    
    filter_dict = {
        "user_id": user_id,
        "categories_primary": primary,
        "categories_special": special
    }
    
    for index in [ami_index, ent_index]:
        try:
            results = index.query(
                vector=[0] * 1536,
                top_k=top_k,
                include_metadata=True,
                namespace=ns,
                filter=filter_dict
            )
            matches = results.get("matches", [])
            logger.debug(f"Knowledge matches from {index_name(index)}: {matches}")
            knowledge.extend([
                {
                    "raw": match["metadata"]["raw"],
                    "confidence": match["metadata"]["confidence"]
                }
                for match in matches
            ])
        except Exception as e:
            logger.error(f"Knowledge search failed in {index_name(index)}: {e}")
    
    logger.info(f"Found {len(knowledge)} entries for primary: {primary}, special: {special}")
    return knowledge

async def query_knowledge(user_id: str, query: str, top_k: int = 3) -> List[Dict]:
    query_embedding = EMBEDDINGS.embed_query(query)
    ns = "wisdom_bank"
    knowledge = []
    
    for index in [ami_index, ent_index]:
        try:
            # Broaden filter to include all non-character entries
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=ns,
                filter={"user_id": user_id, "categories_special": {"$in": ["", "description"]}}
            )
            matches = results.get("matches", [])
            logger.debug(f"Knowledge matches from {index_name(index)}: {matches}")
            knowledge.extend([
                {
                    "id": match["id"],
                    "raw": match["metadata"]["raw"],
                    "categories": {
                        "primary": match["metadata"]["categories_primary"],
                        "special": match["metadata"]["categories_special"]
                    },
                    "confidence": match["metadata"]["confidence"],
                    "score": match["score"]
                }
                for match in matches
            ])
        except Exception as e:
            logger.error(f"Knowledge query failed in {index_name(index)}: {e}")
    
    knowledge = sorted(knowledge, key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"Queried {len(knowledge)} knowledge entries for '{query}'")
    return knowledge

async def generate_response(user_id: str, query: str) -> str:
    global AI_NAME
    instincts = await load_instincts(user_id)
    response_parts = [f"Xin chào, tôi là {AI_NAME or 'AI'}."]
    seen_desc = set()
    
    for trait, instruction in instincts.items():
        desc = await find_knowledge(user_id, primary=trait, special="description")
        if desc and desc[0]["raw"] not in seen_desc:
            response_parts.append(desc[0]["raw"] + ".")
            seen_desc.add(desc[0]["raw"])
        if "curious" in instruction.lower():
            response_parts.append("Điều gì khiến bạn quan tâm đến việc này vậy?")
        if "truthful" in instruction.lower():
            response_parts.append("Tôi sẽ trả lời một cách trung thực nhất có thể.")
    
    query_results = await query_knowledge(user_id, query)
    if query_results:
        top_result = query_results[0]["raw"]
        if top_result not in seen_desc:  # Avoid duplicating descriptions from instincts
            response_parts.append(top_result + ".")
            seen_desc.add(top_result)
    else:
        response_parts.append("Tôi chưa có nhiều thông tin—hãy cho tôi biết thêm nhé!")
    
    return " ".join(response_parts)

async def test_scenario():
    user_id = "thefusionlab"
    
    print("\n--- Training ---")
    messages = [
        "Em xưng là Linh Chi trong cac cuộc hội thoại nhé",
        "You should always be curious and truthful",
        "Curiosity means asking why and seeking deeper understanding",
        "Để thuyết phục khách mua nhà cần thông tin ngân hàng",
        "Trung thực là thể hiện sự chân thành trong giao tiếp"
    ]
    for input_text in messages:
        success = await save_training(input_text, user_id)
        print(f"Save '{input_text}' successful: {success}")
    
    print("\n--- Waiting 5 seconds for index sync ---")
    await asyncio.sleep(5)  # Increased delay
    
    print("\n--- Recall Test ---")
    queries = [
        "Tell me about buying a house",
        "How’s it going today?",
        "Anh Tuấn hỏi mua nhà"
    ]
    for query in queries:
        response = await generate_response(user_id, query)
        print(f"Query: '{query}'")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    asyncio.run(test_scenario())
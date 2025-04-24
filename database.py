# database.py
# Built by: The Fusion Lab
# Date: March 28, 2025

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utilities import EMBEDDINGS, logger
from datetime import datetime
import uuid
import asyncio
import json
import re
from supabase import create_client, Client
from typing import Dict, List
# Import the new hotbrain module for enhanced vector operations
from hotbrain import get_cached_embedding
# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)

from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)

# Initialize Pinecone client with proper error handling


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
ami_index_name = os.getenv("PRESET", "ami-index")
ent_index_name = os.getenv("ENT", "ent-index")
    
    # Log the actual index names
logger.info(f"Pinecone index names: AMI_INDEX={ami_index_name}, ENT_INDEX={ent_index_name}")
    
    # Create placeholder indexes if needed
try:
    ami_index = pc.Index(ami_index_name)
    ent_index = pc.Index(ent_index_name)
except Exception as e:
        logger.error(f"Failed to initialize Pinecone indexes: {e}")


inferLLM = ChatOpenAI(model="gpt-4o", streaming=False)

def index_name(index) -> str:
    if index == ami_index:
        return f"ami_index ({ami_index_name})"
    else:
        return f"ent_index ({ent_index_name})"

AI_NAME = None
async def infer_categories(input: str, context: str = "") -> Dict:
    category_prompt = f"""
    Conversation Context: '{context}'
    Latest Input: '{input}'
    Task: Identify categories per Star Net 2.2 Blueprint.
    - 'categories': English terms (e.g., "primary": "curiosity", "special": "character").
    - 'labels': Bilingual pairs (Vietnamese, English).
    Examples: 'Học Hỏi' (Learning), 'Kỹ Năng Bán Hàng' (Sales Skills), 'Tính Cách' (Character).

    Rules:
    - Name: If input explicitly sets AI name (e.g., "You're <name>", "em hãy nhớ em tên là <name>", "xưng là <name>"):
      - "primary": "name", "special": "character".
      - Label: "Tên" (Name).
    - Instincts: If input defines AI behavior with clear directives (e.g., 'be curious', 'always be truthful', 'should be polite'):
      - "special": "character".
      - "primary": Extract traits from the directive (e.g., "curiosity", "truthfulness").
      - Label: "Tính Cách" (Character).
    - Knowledge: If input provides a fact, method, or explanation not tied to AI behavior:
      - "primary": Identify the main topic (e.g., "sales skills", "curiosity", "truthfulness").
      - "special": "description" if it explains or defines (e.g., contains "means", "is", "là", or ":"), "" if procedural (e.g., "cần", "do this").
      - Label: Use the topic in Vietnamese and English (e.g., "Tò Mò" (Curiosity), "Kỹ Năng Bán Hàng" (Sales Skills), "Trung Thực" (Truthfulness)).
    - Unclear: If none apply, set 'needs_clarification': true.

    Guidelines:
    - Favor 'description' for statements with explanatory intent (e.g., "X là Y", "X is Y") unless a directive explicitly targets AI behavior (e.g., "You should be X", "Always be X").
    - Extract 'primary' dynamically from the input's subject, not a fixed list.
    - Avoid assuming traits unless a directive is present.

    Examples to Follow Strictly:
    - "You're curious and truthful" → {{"primary": "curiosity, truthfulness", "special": "character"}}, [{{"original": "Tính Cách", "english": "Character", "requires_context": false}}]
    - "Thành thật là chân thành với mọi người" → {{"primary": "truthfulness", "special": "description"}}, [{{"original": "Trung Thực", "english": "Truthfulness", "requires_context": false}}]
    - "Người tò mò là người luôn tìm kiếm" → {{"primary": "curiosity", "special": "description"}}, [{{"original": "Tò Mò", "english": "Curiosity", "requires_context": false}}]
    - "Bán hàng là nghệ thuật" → {{"primary": "sales skills", "special": "description"}}, [{{"original": "Kỹ Năng Bán Hàng", "english": "Sales Skills", "requires_context": false}}]
    - "Em hãy nhớ em tên là Linh Chi" → {{"primary": "name", "special": "character"}}, [{{"original": "Tên", "english": "Name", "requires_context": false}}]

    Return JSON:
    {{
      "categories": {{"primary": "<topic>", "special": "<type>"}},
      "labels": [{{"original": "<viet>", "english": "<eng>", "requires_context": true/false}}],
      "needs_clarification": true/false
    }}
    """
    
    try:
        response = await asyncio.to_thread(inferLLM.invoke, category_prompt,temperature=0.1,max_tokens=1000)
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
async def save_training(input: str, user_id: str, context: str = "", bank_name :str = "", mode: str = "default") -> bool:
    global AI_NAME
    embedding = await get_cached_embedding(input)
    data = await infer_categories(input, context)
    ns = bank_name
    
    if data["needs_clarification"]:
        logger.warning(f"Input '{input}' needs clarification.")
        return False
    
    categories = data["categories"]
    target_index = ami_index if mode == "pretrain" else ent_index
    
    if categories["primary"] == "name":
        name_prompt =f"Extract the name from '{input}' and return only the name, e.g., 'Linh Chi' for 'Em hãy nhớ em tên là Linh Chi!'."
        name_response = await asyncio.to_thread(inferLLM.invoke, name_prompt)
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

async def save_training_with_chunk(
    input: str,
    user_id: str,
    context: str = "",
    mode: str = "default",
    doc_id: str = None,
    chunk_id: str = None,
    bank_name: str ="",
    is_raw: bool = False
    
) -> bool:
    global AI_NAME
    embedding = await get_cached_embedding(input)
    logger.info(f"Bank name at Save training chunk={bank_name}")
    logger.info(f"Environment variables: PRESET={os.getenv('PRESET', 'not set')}, ENT={os.getenv('ENT', 'not set')}")
    ns = bank_name 
    target_index = ami_index if mode == "pretrain" else ent_index

    index_name_str = index_name(target_index)
    logger.info(f"Saving to Target index at Save training chunk={index_name_str}, mode={mode}, namespace={ns}")
    logger.info(f"Using user_id={user_id}, doc_id={doc_id}, chunk_id={chunk_id}")

    # Generate a default chunk_id if none provided
    chunk_id = chunk_id or str(uuid.uuid4())

    # If this is a raw chunk, save it directly with minimal processing
    if is_raw:
        metadata = {
            "created_at": datetime.now().isoformat(),
            "raw": input,
            "confidence": 0.95,
            "source": "document",
            "user_id": user_id,
            "doc_id": doc_id or "unknown",
            "chunk_id": chunk_id,
            "categories_primary": "raw",
            "categories_special": "document"
        }
        convo_id = f"{user_id}_{chunk_id}"
        try:
            logger.info(f"Upserting raw chunk with convo_id={convo_id} to {index_name_str}")
            target_index.upsert([(convo_id, embedding, metadata)], namespace=ns)
            logger.info(f"Saved raw chunk to {index_name_str}: {convo_id}")
            return True
        except Exception as e:
            logger.error(f"Raw upsert failed: {e}")
            return False

    # Otherwise, infer categories and extract knowledge
    data = await infer_categories(input, context)
    if data["needs_clarification"]:
        logger.warning(f"Input '{input}' needs clarification, saving as unclassified.")
        categories = {"primary": "unclassified", "special": ""}
        labels = [{"original": "Chưa phân loại", "english": "Unclassified", "requires_context": True}]
    else:
        categories = data["categories"]
        labels = data["labels"]

    # Handle naming
    if categories["primary"] == "name":
        name_prompt = f"Extract the name from '{input}' and return only the name."
        name_response = await asyncio.to_thread(inferLLM.invoke, name_prompt)
        AI_NAME = name_response.content.strip()
        logger.debug(f"Set AI_NAME to: {AI_NAME}")

    # Duplicate check
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

    # Build metadata for knowledge
    metadata = {
        "created_at": datetime.now().isoformat(),
        "raw": input,
        "confidence": 0.85,
        "source": "document",
        "categories_primary": categories["primary"],
        "categories_special": categories.get("special", ""),
        "labels": json.dumps(labels),
        "user_id": user_id,
        "doc_id": doc_id or "unknown",
        "chunk_id": chunk_id
    }
    if categories["primary"] == "name":
        metadata["name"] = AI_NAME

    # Upsert knowledge
    convo_id = f"{user_id}_{uuid.uuid4()}"
    try:
        logger.info(f"Upserting knowledge with convo_id={convo_id} to {index_name_str}, categories={categories}")
        target_index.upsert([(convo_id, embedding, metadata)], namespace=ns)
        logger.info(f"Saved knowledge to {index_name_str}: {convo_id} - Categories: {categories}")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

async def load_instincts(user_id: str) -> Dict[str, str]:
    ns = "wisdom_bank"
    instincts = {}
    name = None
    
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
                    name = match["metadata"].get("name", raw_text.strip())
                    logger.debug(f"Set name to: {name}")
                elif "always be" in raw_text and "means" not in raw_text and "is" not in raw_text and "là" not in raw_text:
                    if primary not in instincts or match["metadata"]["created_at"] > instincts[primary]["created_at"]:
                        logger.debug(f"Updating {primary}: {match['metadata']['raw']} ...")
                        instincts[primary] = {
                            "raw": match["metadata"]["raw"],
                            "created_at": match["metadata"]["created_at"]
                        }
        except Exception as e:
            logger.error(f"Instinct load failed in {index_name(index)}: {e}")
    
    instincts_dict = {k: v["raw"] for k, v in instincts.items()}
    if name:
        instincts_dict["name"] = name
    logger.info(f"Loaded instincts: {instincts_dict} for user {user_id}")
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

async def query_knowledge(query: str, bank_name :str = "", top_k: int = 10) -> List[Dict]:
    """
    Query knowledge from vector databases (ami_index and ent_index)
    
    Args:
        query: The search query
        bank_name: Optional namespace (bank name) to query
        top_k: Maximum number of results to return
        
    Returns:
        List of knowledge entries
    """
    embedding = await get_cached_embedding(query)
    ns = bank_name
    knowledge = []
    
    logger.info(f"Querying namespace={ns} in Index:{ami_index_name} and {ent_index_name}")
    
    for index in [ami_index, ent_index]:
        try:
            # Broaden filter to include all non-character entries
            results = await asyncio.to_thread(
                index.query,
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=ns,
                #filter={"categories_special": {"$in": ["", "description"]}}
                filter={"categories_special": {"$in": ["", "description", "document", "procedural"]}}
            )
            
            matches = results.get("matches", [])
            #logger.info(f"Knowledge matches from {index_name(index)}: {matches}")
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

async def get_all_primary_categories(bank_name :str = "") -> set[str]:
    """
    Scans ent_index and returns a set of all unique primary categories.
    Handles comma-separated categories_primary values.
    """
    ns = bank_name
    all_categories = set()
    top_k = 10000  # Max allowed by Pinecone; adjust if needed

    try:
        # Query with a zero vector to fetch all entries
        results = await asyncio.to_thread(
            ent_index.query,
            vector=[0] * 1536,  # Matches your OpenAI embedding size
            top_k=top_k,
            include_metadata=True,
            namespace=ns,
            filter={}  # No filter to get all entries
        )
        matches = results.get("matches", [])
        
        # Process each match
        for match in matches:
            categories = match["metadata"].get("categories_primary", "")
            # Split comma-separated categories and strip whitespace
            for category in categories.split(","):
                category = category.strip()
                if category:
                    all_categories.add(category)

        logger.info(f"Found {len(all_categories)} unique primary categories")
        return all_categories

    except Exception as e:
        logger.error(f"Failed to retrieve primary categories: {e}")
        return set()

async def get_raw_data_by_category(primary_category: str, top_k: int = 10000, user_id: str = None, bank_name: str = "") -> List[str]:
    """
    Returns all raw data associated with a specific primary category from ent_index.
    Fetches data with optional user_id filter and matches comma-separated categories_primary locally.
    Logs categories_primary for debugging.
    """
    ns = bank_name
    raw_data = []

    try:
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = {"$eq": user_id}

        results = await asyncio.to_thread(
            ent_index.query,
            vector=[0] * 1536,
            top_k=top_k,
            include_metadata=True,
            namespace=ns,
            filter=filter_dict
        )
        matches = results.get("matches", [])
        
        for match in matches:
            categories = match["metadata"].get("categories_primary", "")
            logger.info(f"Checking categories_primary: '{categories}' against '{primary_category}'")
            if primary_category in [cat.strip() for cat in categories.split(",")]:
                raw_text = match["metadata"].get("raw", "")
                if raw_text:
                    raw_data.append(raw_text)

        logger.info(f"Found {len(raw_data)} raw entries for category '{primary_category}'")
        return raw_data

    except Exception as e:
        logger.error(f"Failed to retrieve raw data for '{primary_category}': {e}")
        return []

async def get_all_labels(lang:str ="english",bank_name: str= "") -> set[str]:
    """
    Scans ent_index and returns a set of all unique English labels from the labels field.
    """
    ns = bank_name
    all_labels = set()
    top_k = 10000  # Max allowed by Pinecone

    try:
        results = await asyncio.to_thread(
            ent_index.query,
            vector=[0] * 1536,
            top_k=top_k,
            include_metadata=True,
            namespace=ns,
            filter={}
        )
        matches = results.get("matches", [])
        
        for match in matches:
            labels_json = match["metadata"].get("labels", "[]")
            try:
                labels = json.loads(labels_json)  # Parse JSON string to list
                for label in labels:
                    lang_label = label.get(lang, "").strip()
                    if lang_label:
                        all_labels.add(lang_label)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse labels JSON: {labels_json}, error: {e}")

        logger.info(f"Found {len(all_labels)} unique English labels")
        return all_labels

    except Exception as e:
        logger.error(f"Failed to retrieve labels: {e}")
        return set()

async def get_raw_data_by_label(label: str, lang: str = "english", bank_name :str ="", top_k: int = 10000) -> List[str]:
    """
    Returns all raw data associated with a specific English label from ent_index.
    Fetches all data and filters locally based on the labels field.
    """
    ns = bank_name
    raw_data = []

    try:
        results = await asyncio.to_thread(
            ent_index.query,
            vector=[0] * 1536,
            top_k=top_k,
            include_metadata=True,
            namespace=ns,
            filter={}
        )
        matches = results.get("matches", [])
        
        # Filter locally for matching English labels
        for match in matches:
            labels_json = match["metadata"].get("labels", "[]")
            try:
                labels = json.loads(labels_json)
                if any(l.get(lang, "").strip() == label for l in labels):
                    raw_text = match["metadata"].get("raw", "")
                    if raw_text:
                        raw_data.append(raw_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse labels JSON: {labels_json}, error: {e}")

        logger.info(f"Found {len(raw_data)} raw entries for label '{label}'")
        return raw_data

    except Exception as e:
        logger.error(f"Failed to retrieve raw data for '{label}': {e}")
        return []

def clean_text(raw: str) -> str:
    """
    Cleans raw text by fixing Unicode issues and removing control characters.
    """
    # First, try decoding as UTF-8 if it's bytes-like, then fix escapes
    try:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        # Replace common garbled patterns and remove control chars like \x95
        raw = raw.encode().decode('utf-8', errors='replace')
        raw = ''.join(c for c in raw if ord(c) >= 32 or c == '\n')  # Keep printable chars
        return raw
    except Exception:
        return raw  # Fallback to original if cleaning fails


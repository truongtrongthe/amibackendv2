from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
llm = ChatOpenAI(model="gpt-4o")  # No streaming for structured JSON output
import json
from copilot import extract_node,detect_intent

from digesting import infer_hidden_intent

import re

def infer_hidden_intent_ok1(kg: dict, surface_intent: str, convo_history: str = "") -> list:
    """Dynamically infer multiple hidden intents using LLM reasoning."""
    entities = kg.get("entities", [])
    relationships = kg.get("relationships", [])
    
    entity_summary = []
    for entity in entities:
        if isinstance(entity, dict):
            base = entity.get("name", entity.get("entity", str(entity)))
            if entity.get("type") == "person":
                age = entity.get("age", "")
                base = f"{base} (age {age})" if age else base
            attrs = entity.get("attribute", "")
            opinion = entity.get("opinion", "")
            summary = base
            if attrs or opinion:
                summary += f" ({attrs}{' - ' + opinion if opinion else ''})"
            entity_summary.append(summary)
        else:
            entity_summary.append(str(entity))
    
    rel_summary = []
    for rel in relationships:
        if isinstance(rel, dict):
            from_entity = rel.get('from')
            to_entity = rel.get('to')
            if isinstance(from_entity, dict):
                from_entity = from_entity.get('name', str(from_entity))
            if isinstance(to_entity, dict):
                to_entity = to_entity.get('attribute', str(to_entity))
            rel_str = f"{from_entity} {rel.get('type')} {to_entity}"
            if rel.get('opinion'):
                rel_str += f" ({rel.get('opinion')})"
            rel_summary.append(rel_str)
        else:
            rel_summary.append(str(rel))
    
    prompt = f"""
    You are an expert at understanding hidden user intents in conversations.
    Given the following data, infer the user's likely hidden intents—the deeper needs or goals they might not explicitly state.
    Focus on combining the entities, relationships (including opinions), surface intent, and conversation history.
    Return a list of concise intent descriptions (e.g., ["buy height growth support for children", "explore affordable height growth products"]).
    If no clear hidden intents emerge, return a list with just the surface intent.

    Entities: {', '.join(entity_summary) or 'None'}
    Relationships: {', '.join(rel_summary) or 'None'}
    Surface Intent: {surface_intent}
    Conversation History (if any): {convo_history or 'None'}

    Provide your reasoning briefly, then list the hidden intents as a JSON array (e.g., ["intent1", "intent2"]).
    """
    
    try:
        response = llm.invoke(prompt)
        full_response = response.content.strip()
        print(f"LLM reasoning: {full_response}")
        
        # Simplified regex to catch any JSON array in ```json block or standalone
        json_match = re.search(r'```json\s*(\[.*?\])\s*```|\s*($$ ".*?\" $$)\s*$', full_response, re.DOTALL)
        if json_match:
            intent_json = json_match.group(1) or json_match.group(2)  # Take whichever group matched
            intents = json.loads(intent_json)
            print(f"Extracted JSON: {intent_json}")
        else:
            print("No JSON array found in response, falling back to surface intent.")
            intents = [surface_intent]
        
        if not intents or all(intent == surface_intent for intent in intents):
            return [surface_intent]
        return intents
    except Exception as e:
        print(f"Error inferring intent: {e}")
        return [surface_intent]



def infer_hidden_intent_worked(kg: dict, surface_intent: str, convo_history: str = "") -> list:
    """Dynamically infer multiple hidden intents using LLM reasoning."""
    entities = kg.get("entities", [])
    relationships = kg.get("relationships", [])
    
    entity_summary = []
    for entity in entities:
        if isinstance(entity, dict):
            base = entity.get("name", entity.get("entity", str(entity)))
            if entity.get("type") == "person":
                age = entity.get("age", "")
                base = f"{base} (age {age})" if age else base
            attrs = entity.get("attribute", "")
            opinion = entity.get("opinion", "")
            summary = base
            if attrs or opinion:
                summary += f" ({attrs}{' - ' + opinion if opinion else ''})"
            entity_summary.append(summary)
        else:
            entity_summary.append(str(entity))
    
    rel_summary = []
    for rel in relationships:
        if isinstance(rel, dict):
            from_entity = rel.get('from')
            to_entity = rel.get('to')
            if isinstance(from_entity, dict):
                from_entity = from_entity.get('name', str(from_entity))
            if isinstance(to_entity, dict):
                to_entity = to_entity.get('attribute', str(to_entity))
            rel_str = f"{from_entity} {rel.get('type')} {to_entity}"
            if rel.get('opinion'):
                rel_str += f" ({rel.get('opinion')})"
            rel_summary.append(rel_str)
        else:
            rel_summary.append(str(rel))
    
    prompt = f"""
    You are an expert at understanding hidden user intents in conversations.
    Given the following data, infer the user's likely hidden intents—the deeper needs or goals they might not explicitly state.
    Use the user's age and gender (if implied by names like 'Chị' for female or 'Anh' for male) as key factors to reason about their potential needs or motivations.
    Combine these with the entities, relationships (including opinions), surface intent, and conversation history to form your inferences.
    Return a list of concise intent descriptions (e.g., ["explore height growth products for her kids", "explore affordable height growth products"]).
    If no clear hidden intents emerge, return a list with just the surface intent.

    Entities: {', '.join(entity_summary) or 'None'}
    Relationships: {', '.join(rel_summary) or 'None'}
    Surface Intent: {surface_intent}
    Conversation History (if any): {convo_history or 'None'}

    Provide your reasoning briefly, then list the hidden intents as a JSON array (e.g., ["intent1", "intent2"]).
    """
    
    try:
        response = llm.invoke(prompt)
        full_response = response.content.strip()
        print(f"LLM reasoning: {full_response}")
        
        # Broadened regex to catch any JSON array
        json_match = re.search(r'(\[".*?\"\])', full_response, re.DOTALL)
        if json_match:
            intent_json = json_match.group(1)
            intents = json.loads(intent_json)
            print(f"Extracted JSON: {intent_json}")
        else:
            print("No JSON array found in response, falling back to surface intent.")
            intents = [surface_intent]
        
        if not intents or all(intent == surface_intent for intent in intents):
            return [surface_intent]
        return intents
    except Exception as e:
        print(f"Error inferring intent: {e}")
        return [surface_intent]

test_inputs = [
    "Chị Thu 35 tuổi muốn tăng chiều cao nhưng chê HITO quá đắt",
    "Anh Nam 28 tuổi thích sản phẩm tự nhiên"
]

for test_input in test_inputs:
    print(f"\nTesting input: '{test_input}'")
    messages = [HumanMessage(content=test_input)]
    surface_intent = detect_intent(messages)
    print(f"Surface intent: {surface_intent}")
    
    kg = extract_node(test_input)
    if kg:
        hidden_intents = infer_hidden_intent(kg, surface_intent)
        print(f"Hidden intents: {hidden_intents}")
    else:
        print("Extraction failed, skipping inference.")
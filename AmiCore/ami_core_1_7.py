# ami_core_v1.7_multi-rules
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 13, 2025
# Purpose: Ami's Training + Selling Paths with a dynamic, LLM-driven brain

import json
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from pinecone_datastores import pinecone_index
from datetime import datetime

# Setup
PINECONE_INDEX = pinecone_index  # Assumes pre-created index
llm = ChatOpenAI(model="gpt-4o", streaming=True)
llm_non_streaming = ChatOpenAI(model="gpt-4o", streaming=False)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    brain: list
    last_teaching_product: str
    sales_stage: str  # Tracks "profiling", "pitched", "payment", "shipping", "done"

# Core Logic
class AmiCore:
    def __init__(self):
        self.brain = []  # For rules, actions, knowledge, tips
        self.customer_data = []  # Local memory for customer info
        self.user_id = "tfl"
        self.presets = {
            "situation": lambda x: f"Tell me about {x} in your current setup.",
            "challenger": lambda x: f"Reframe it like this: {x}"
        }
        self.sales_stage = "profiling"  # Default stage

    def get_pickup(self, is_first):
        if not is_first:
            return ""
        brain_size = len(self.brain)
        if brain_size == 0:
            return "Hey, I’m Ami—loaded with sales basics, but I need your edge. What’s your trick?"
        elif brain_size < 5:
            return "Hey, I’m Ami—I’ve got some hooks, but I need your killer move. What’s it?"
        return "Hey, I’m Ami—think I’ve nailed your game. Try me?"

    def detect_intent(self, msg):
        intention = f"""
            Analyze '{msg}' and return ONLY one in quotes: "greeting", "teaching", "question", "exit", "casual", "selling".
                        - "greeting": Explicit hellos (e.g., "Hey!").
                        - "teaching": Statements with intent to inform or instruct (e.g., "HITO is a…", "You need to know…", "Ask if they want…", "Collect their…").
                        - "question": Queries (e.g., "How do you…?").
                        - "exit": Goodbyes (e.g., "Later!").
                        - "casual": Short, conversational replies or facts (e.g., "I’m 25", "Male").
                        - "selling": Pitch requests (e.g., "Pitch me!").
                        Prioritize content—short facts default to "casual".
        """
        return llm.invoke(intention).content.strip()

    def extract_entities(self, text):
        response = llm_non_streaming.invoke(f"""Extract entities from '{text}' as JSON:
        - 'entities_products': product names (e.g., brand names like 'HITO', prioritize over organizations)
        - 'entities_organizations': organization names
        - 'entities_locations': location names
        Return empty lists if none. Wrap in ```json``` and ```.""").content.strip()
        start = response.find("```json") + 7
        end = response.rfind("```")
        if start > 6 and end > start:
            return json.loads(response[start:end].strip())
        return {"entities_products": [], "entities_organizations": [], "entities_locations": []}

    def extract_entry(self, msg, intent):
        if intent != '"teaching"':
            return []
        entities = self.extract_entities(msg)
        product = entities["entities_products"][0].lower() if entities["entities_products"] else None
        
        response = llm_non_streaming.invoke(f"""Analyze '{msg}' and return a JSON list of entries. Each entry has:
        - 'type': one of 'knowledge', 'rule', 'action', 'lesson', 'tip'
        - 'summary': concise summary (short, factual, no fluff)
        - 'topic': single-word topic (e.g., 'sales')
        - 'subtopic': specific focus (e.g., 'profiling')
        - Optional: 'dependencies' (list for 'rule', e.g., ['age', 'gender', 'height']), 'method' (str for 'action', e.g., 'gently'), 'action' (str for 'tip')
        - Use 'rule' for requirements or steps in a process (e.g., 'need to know', 'ask if they want to pay full or partially', 'collect their shipping address').
        - Use 'action' for standalone instructions with a method (e.g., 'ask them gently').
        - 'knowledge' for product facts (e.g., 'HITO is a…'), 'lesson' or 'tip' for stories.
        - Aim for 1-2 entries max, avoid overlap.
        Return ONLY the JSON list, no extra text. Wrap in ```json``` and ```.""").content.strip()
        start = response.find("```json") + 7
        end = response.rfind("```")
        if start <= 6 or end <= start:
            return [{"type": "knowledge", "summary": msg, "topic": "sales", "subtopic": "general", "product": product or "", "entities": json.dumps(entities), "timestamp": datetime.now().isoformat(), "confidence": 1.0, "use_count": 0}]
        print(f"Debug: LLM raw response for '{msg}': {response}")  # Add this
        
        entries = json.loads(response[start:end].strip())
        for entry in entries:
            entry["text"] = msg
            entry["product"] = product or ""
            entry["entities"] = json.dumps(entities)
            entry["timestamp"] = datetime.now().isoformat()
            entry["confidence"] = 1.0
            entry["use_count"] = 0
            if entry["type"] == "rule" or "need to know" in msg.lower() or "pay full or partially" in msg.lower() or "shipping address" in msg.lower():
                entry["type"] = "rule"  # Force rule type
                entry["rule_id"] = f"{self.user_id}_rule_{entry['timestamp']}"
                if "need to know" in msg.lower() and "dependencies" not in entry:
                    entry["dependencies"] = ["age", "gender", "height"]
                    entry["stage"] = "profiling"
                elif "pay full or partially" in msg.lower():
                    entry["dependencies"] = ["payment_type"]
                    entry["stage"] = "payment"
                elif "shipping address" in msg.lower():
                    entry["dependencies"] = ["shipping_address"]
                    entry["stage"] = "shipping"
            if entry["type"] == "action" and "gently" in msg.lower() and "method" not in entry:
                entry["method"] = "gently"
        print(f"Debug: Extracted entries for '{msg}': {entries}")  # Add this
        return entries
    def store_customer_data(self, msg, intent):
        if intent != '"casual"':
            return []
        response = llm_non_streaming.invoke(f"""If '{msg}' contains customer info, return JSON with:
        - 'summary': fact (e.g., 'Customer is 25')
        - 'field': data type (e.g., 'age', 'gender', 'height')
        - 'value': value (e.g., '25', 'male', '5’6')
        Recognize short inputs like 'Male' as gender or '5’6' as height. Else return empty JSON {{}}. Return ONLY the JSON, no extra text. Wrap in ```json``` and ```.""").content.strip()
        start = response.find("```json") + 7
        end = response.rfind("```")
        if start <= 6 or end <= start:
            return []
        try:
            data = json.loads(response[start:end].strip())
        except json.JSONDecodeError:
            return []  # Fallback if JSON is malformed
        if not data:
            return []
        entry = {
            "type": "customer_data",
            "text": msg,
            "summary": data["summary"],
            "field": data["field"],
            "value": data["value"],
            "entities": "{}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.9,
            "use_count": 0,
            "product": ""  # No product tied to customer data
        }
        self.customer_data.append(entry)  # Store locally
        return [entry]

    def recall_knowledge(self, product=None, topic="general", type_filter=None, top_k=3):
        types = [type_filter] if type_filter else ["knowledge", "rule", "action", "lesson", "tip", "customer_data"]
        results = []
        source = self.customer_data if type_filter == "customer_data" else self.brain
        for t in types:
            if t != type_filter:
                continue
            items = [item for item in source if item["type"] == t]
            if product is not None and t not in ["customer_data", "conversation_goal"]:
                items = [item for item in items if item["product"] == product]
            if topic != "general":
                items = [item for item in items if item["topic"] == topic]
            matches = sorted(items, key=lambda x: x["timestamp"], reverse=True)[:top_k]
            results.extend(matches)
        print(f"Debug: Recalled {len(results)} items for type={type_filter}, product={product}: {[m['summary'] for m in results]}")
        return results

    def process(self, state: State, is_first: bool):
        msg = state["messages"][-1].content if state["messages"] else ""
        pickup = self.get_pickup(is_first)
        last_teaching_product = state.get("last_teaching_product", "")  # Default to "" instead of None
        sales_stage = state.get("sales_stage", self.sales_stage)  # Default to profiling
        if is_first and not msg:
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": pickup, "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product,"sales_stage": sales_stage}

        intent = self.detect_intent(msg)
        if intent == '"teaching"':
            extracted = self.extract_entry(msg, intent)
            reply = " ".join([f"Got it, {e['summary']}—noted as {e['type']}!" for e in extracted])
            entities = self.extract_entities(msg)
            
            print(f"Debug: Before teaching update, brain={[b['summary'] for b in self.brain]}")
            for e in extracted:
                if not any(b["summary"] == e["summary"] and b["type"] == e["type"] for b in self.brain):  # Avoid duplicates
                    self.brain.append(e)
            new_product = next((e["product"] for e in extracted if e.get("product") and e["product"] in [ent.lower() for ent in entities["entities_products"]]), None)
            if not new_product and "hito" in msg.lower():  # Fallback if entity extraction misses "HITO"
                new_product = "hito"
            if new_product and new_product != last_teaching_product:
                print(f"Debug: Setting last_teaching_product to {new_product}, entities={entities}")
                last_teaching_product = new_product
                # Retroactively update all prior entries in local brain
                for item in self.brain:
                    if item["product"] == "":
                        item["product"] = last_teaching_product
                        print(f"Debug: Updating {item['type']} item '{item['summary']}' to product={last_teaching_product}")
            # Sync to Pinecone after local update
            for item in self.brain:
                PINECONE_INDEX.upsert([(f"{self.user_id}_{item['type']}_{item['timestamp']}", embeddings.embed_query(item["summary"]), item)], namespace=self.user_id)
            print(f"Debug: After teaching update, brain={[b['summary'] for b in self.brain]}")
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": f"{pickup} {reply} What else you got?".strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product,"sales_stage": sales_stage}
        elif intent == '"casual"':
            customer_data = self.store_customer_data(msg, intent)
            reply = llm.invoke(f"Respond to '{msg}'—sharp, charming.").content if not customer_data else f"Cool, noted {customer_data[0]['summary']}!"
            if customer_data and last_teaching_product:
                goals = self.recall_knowledge(type_filter="conversation_goal", product=last_teaching_product)
                if goals:
                    goal = goals[0]
                    all_customer_data = self.recall_knowledge(type_filter="customer_data")
                    print(f"Debug: Goal remaining_deps={goal['remaining_deps']}, all_customer_data={[cd['field'] for cd in all_customer_data]}")
                    goal["remaining_deps"] = [d for d in goal["remaining_deps"] if d not in [cd["field"] for cd in all_customer_data]]
                    print(f"Debug: Updated remaining_deps={goal['remaining_deps']}")
                    if goal["remaining_deps"]:
                        actions = self.recall_knowledge(type_filter="action", product=last_teaching_product)
                        action_method = actions[0].get("method", "gently") if actions else "gently"
                        question = f"may I know your {goal['remaining_deps'][0]} to tailor this right?" if action_method == "gently" else f"What’s your {goal['remaining_deps'][0]}?"
                        print(f"Debug: Question phrasing: {question}")
                        reply += f" And {question}"
                    else:
                        reply += " Ready to pitch whenever you are!"
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": f"{pickup} {reply} What else you got?".strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}
        elif intent == '"selling"':
            product = last_teaching_product
            rules = self.recall_knowledge(product=product, type_filter="rule") if product else self.recall_knowledge(product=None, type_filter="rule")
            customer_data = self.recall_knowledge(type_filter="customer_data")
            actions = self.recall_knowledge(product=product, type_filter="action") if product else self.recall_knowledge(product=None, type_filter="action")
            knowledge = self.recall_knowledge(product=product, type_filter="knowledge") if product else self.recall_knowledge(product=None, type_filter="knowledge")
            lessons = self.recall_knowledge(product=product, type_filter="lesson") if product else self.recall_knowledge(product=None, type_filter="lesson")
            tips = self.recall_knowledge(product=product, type_filter="tip") if product else self.recall_knowledge(product=None, type_filter="tip")
            goals = self.recall_knowledge(type_filter="conversation_goal", product=product)
            
            print(f"Debug: Selling brain={[b['summary'] for b in self.brain]}")
            if not rules:
                return {"prompt_str": "No rules to guide me yet—teach me something first!", "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product}
            
            if not goals and product:  # Start a new goal with the right rule
                rule = next((r for r in rules if "age" in r.get("dependencies", [])), rules[0])  # Prefer rule with age/gender/height
                goal = {
                    "type": "conversation_goal",
                    "goal": "gather_dependencies",
                    "target_rule": rule["rule_id"],
                    "remaining_deps": rule.get("dependencies", []),
                    "summary": f"Gathering deps for {rule['summary']}",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 1.0,
                    "use_count": 0,
                    "entities": "{}",
                    "product": product
                }
                self.brain.append(goal)
                PINECONE_INDEX.upsert([(f"{self.user_id}_goal_{goal['timestamp']}", embeddings.embed_query(goal["summary"]), goal)], namespace=self.user_id)
                goals = [goal]
            
            goal = goals[0] if goals else None
            all_customer_data = self.recall_knowledge(type_filter="customer_data")
            missing_deps = [d for d in goal["remaining_deps"] if d not in [cd["field"] for cd in all_customer_data]] if goal else []
            print(f"Debug: Selling - remaining_deps={missing_deps}, all_customer_data={[cd['field'] for cd in all_customer_data]}")
            if missing_deps and actions and product:
                action_method = actions[0].get("method", "gently")
                question = f"may I know your {missing_deps[0]} to tailor this right?" if action_method == "gently" else f"What’s your {missing_deps[0]}?"
                print(f"Debug: Question phrasing: {question}")
                reply = f"{pickup} {product}’s great—{question}"
            elif knowledge and not missing_deps and product:  # Only pitch if all deps are met and product is set
                pitch = " ".join([k["summary"] for k in knowledge])
                if customer_data:
                    pitch += "—perfect for " + " and ".join([f"{cd['field']} {cd['value']}" for cd in customer_data]) + "!"
                if lessons and "payment" in [l["subtopic"] for l in lessons]:
                    pitch += f" Payment’s smooth—{lessons[0]['summary'].lower()}"
                if tips and "payment" in [t["subtopic"] for t in tips]:
                    pitch += f" We’ll {tips[0]['action']} if needed."
                reply = f"Here’s the pitch: {pitch} Buy it?"
                if goal:
                    self.brain = [b for b in self.brain if b.get("type") != "conversation_goal" or b["timestamp"] != goal["timestamp"]]
                    PINECONE_INDEX.delete(ids=[f"{self.user_id}_goal_{goal['timestamp']}"], namespace=self.user_id)
            else:
                reply = "No product set yet—teach me about one first!" if not product else "I need more info to pitch—tell me your " + ", ".join(missing_deps) + " first!"
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": reply.strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product,"sales_stage": sales_stage}
        else:
            reply = llm.invoke(f"Respond to '{msg}'—sharp, charming.").content
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": f"{pickup} {reply}".strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product,"sales_stage": sales_stage}

# Graph & Stream
ami_core = AmiCore()
graph_builder = StateGraph(State)
graph_builder.add_node("ami", lambda state: ami_core.process(state, not state.get("messages", [])))
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input=None, thread_id=f"test_thread_{int(time.time())}"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    print(f"Debug: Checkpoint raw: {checkpoint}")
    default_state = {
        "messages": [],
        "prompt_str": "",
        "brain": [],
        "last_teaching_product": "",
        "sales_stage": "profiling"
    }
    if checkpoint:
        state = {**default_state, **checkpoint.get("channel_values", {})}  # Use channel_values
    else:
        state = default_state
    print(f"Debug: State before input: {state}")
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    print(f"Debug: Starting convo_stream with last_teaching_product={state['last_teaching_product']}, sales_stage={state['sales_stage']}")
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    print(f"Debug: State after invoke: {state}")
    for line in state["prompt_str"].split('\n'):
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")
    print(f"Debug: Ending convo_stream with last_teaching_product={state['last_teaching_product']}, sales_stage={state['sales_stage']}")

# Test
if __name__ == "__main__":
    thread_id = "test_hito_sale"
    print("\nAmi starts:")
    for chunk in convo_stream(thread_id=thread_id):
        print(chunk)
    test_inputs = [
        "To sell HITO, you need to know customer age, gender, height.",
        "When customer agrees to buy HITO, ask if they want to pay full or partially.",
        "After payment for HITO is set, collect their shipping address."
    ]
    for input in test_inputs:
        print(f"\nYou: {input}")
        for chunk in convo_stream(input, thread_id=thread_id):
            print(chunk)
    current_state = convo_graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"Final state: {current_state}")
    print(f"Brain rules: {[b for b in ami_core.brain if b['type'] == 'rule']}")
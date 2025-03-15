# ami_core_1_8
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 13, 2025
# Purpose: Ami's Training + Staged Selling with Multiple Rules, Fixed Double Nudge, Post-Sale Nudge, Trimmed Debugs

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
    sales_stage: str  # "profiling", "pitched", "payment", "shipping", "done"

# Core Logic
class AmiCore:
    def __init__(self):
        self.brain = []
        self.customer_data = []
        self.user_id = "tfl"
        self.presets = {
            "situation": lambda x: f"Tell me about {x} in your current setup.",
            "challenger": lambda x: f"Reframe it like this: {x}"
        }
        self.sales_stage = "profiling"
        # Load existing knowledge from Pinecone at init
        self.load_brain_from_pinecone()

    def load_brain_from_pinecone(self):
        try:
            results = PINECONE_INDEX.query(
                vector=[0] * 1536,
                top_k=100,
                namespace=self.user_id,
                include_metadata=True,
                include_values=False
            )
            if results["matches"]:
                self.brain = [match["metadata"] for match in results["matches"]]
                product_entries = [b for b in self.brain if b.get("product")]
                if product_entries:
                    self.last_teaching_product = sorted(product_entries, key=lambda x: x["timestamp"], reverse=True)[0]["product"]
                else:
                    self.last_teaching_product = ""
            else:
                self.brain = []
                self.last_teaching_product = ""
        except Exception as e:
            print(f"Error loading from Pinecone: {e}")
            self.brain = []
            self.last_teaching_product = ""

    def get_pickup(self, is_first):
        if not is_first:
            return ""
        brain_size = len(self.brain)
        if brain_size == 0:
            return "Hey, I’m Ami—loaded with sales basics, but I need your edge. What’s your trick?"
        
        product = self.last_teaching_product or "stuff"
        knowledge = self.recall_knowledge(product=product, type_filter="knowledge", top_k=2)
        rules = self.recall_knowledge(product=product, type_filter="rule", top_k=4)
        actions = self.recall_knowledge(product=product, type_filter="action", top_k=4)
        lessons = self.recall_knowledge(product=product, type_filter="lesson", top_k=2) or self.recall_knowledge(product=product, type_filter="tip", top_k=2)
        
        # Pair actions with rules by stage and keywords
        rule_action_pairs = []
        for rule in rules:
            matched_action = next((a for a in actions if rule["product"] == a["product"] and 
                                (rule.get("stage") == a.get("stage") or 
                                any(keyword in a["summary"].lower() for keyword in rule["summary"].lower().split()))), None)
            rule_action_pairs.append((rule, matched_action))
            if matched_action:
                actions.remove(matched_action)
        
        pickup = f"Hey, I’m Ami—I’ve trained to understand **{product.upper()}**.\n\n"
        
        if knowledge:
            pickup += "**What I Know**\n```markdown\n" + "\n".join([f"- {k['summary']}" for k in knowledge]) + "\n```\n\n"
        
        if rule_action_pairs:
            pickup += "**How I Sell**\n```markdown\n"
            for rule, action in rule_action_pairs:
                pickup += f"- {rule['summary']}"
                if action:
                    pickup += f" (*{action['summary']}*)"
                pickup += f" [{rule.get('stage', 'general')}]"
                pickup += "\n"
            pickup += "```\n\n"
        
        if lessons:
            pickup += "**Lessons Learned**\n```markdown\n" + "\n".join([f"- {l['summary']}" for l in lessons]) + "\n```\n\n"
        
        if brain_size < 5:
            pickup += "What’s your killer move to add?"
        else:
            pickup += "Try me?"
        return pickup.strip()

    def detect_intent(self, msg):
        prompt = f"""Analyze '{msg}' and return ONLY one in quotes: "greeting", "teaching", "question", "exit", "casual", "selling", "buying".
        - "greeting": Explicit hellos (e.g., "Hey!", "Hi Ami").
        - "teaching": Statements with intent to inform or instruct (e.g., "HITO is a…", "You need to know…", "Ask if they want…").
        - "question": Queries starting with 'what', 'how', 'why', or ending with '?' (e.g., "How do you…?", "What you have, Ami?").
        - "exit": Goodbyes (e.g., "Later!", "Bye").
        - "casual": Short, conversational replies or facts (e.g., "I’m 25", "Male", "Full").
        - "selling": Pitch requests (e.g., "Pitch me!").
        - "buying": Agreement to buy (e.g., "Yes", "I’ll take it").
        Prioritize content—short facts default to "casual" unless clearly instructional or questioning."""
        return llm.invoke(prompt).content.strip()

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
        
        entries = json.loads(response[start:end].strip())
        seen_summaries = set()
        unique_entries = []
        for entry in entries:
            if entry["summary"] not in seen_summaries:
                seen_summaries.add(entry["summary"])
                entry["text"] = msg
                entry["product"] = product or ""
                entry["entities"] = json.dumps(entities)
                entry["timestamp"] = datetime.now().isoformat()
                entry["confidence"] = 1.0
                entry["use_count"] = 0
                if "ask" in msg.lower() and "gently" in msg.lower() and entry["type"] != "rule":
                    entry["type"] = "action"
                    entry["stage"] = "payment" if "pay" in msg.lower() else "profiling"
                elif entry["type"] == "rule" or "need to know" in msg.lower() or "pay full or partially" in msg.lower() or "shipping address" in msg.lower():
                    entry["type"] = "rule"
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
                unique_entries.append(entry)
        return unique_entries

    def store_customer_data(self, msg, intent, last_ami_msg):
        if intent != '"casual"':
            return []
        prompt = f"""Given prior message '{last_ami_msg}', if '{msg}' contains customer info, return JSON with:
        - 'summary': fact (e.g., 'Customer is 25', 'Payment is full', 'Shipping address is {msg}')
        - 'field': data type (e.g., 'age', 'gender', 'height', 'payment_type', 'shipping_address')
        - 'value': value (e.g., '25', 'male', '5’6', 'full', '{msg}')
        Recognize short inputs: 'Male' as gender, '5’6' as height, 'full' or 'partially' as payment_type (especially after payment questions), addresses as shipping_address with exact value in summary. Else return empty JSON {{}}. Return ONLY the JSON, no extra text. Wrap in ```json``` and ```."""
        response = llm_non_streaming.invoke(prompt).content.strip()
        start = response.find("```json") + 7
        end = response.rfind("```")
        if start <= 6 or end <= start:
            return []
        try:
            data = json.loads(response[start:end].strip())
        except json.JSONDecodeError:
            return []
        if not data and "pay" in last_ami_msg.lower() and msg.lower() in ["full", "partially"]:
            data = {"summary": f"Payment is {msg.lower()}", "field": "payment_type", "value": msg.lower()}
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
            "product": ""
        }
        self.customer_data.append(entry)
        return [entry]

    def recall_knowledge(self, product=None, topic="general", type_filter=None, top_k=3):
        types = [type_filter] if type_filter else ["knowledge", "rule", "action", "lesson", "tip", "customer_data"]
        results = []
        source = self.customer_data if type_filter == "customer_data" else self.brain
        for t in types:
            if t != type_filter and type_filter is not None:
                continue
            items = [item for item in source if item["type"] == t]
            if product is not None and t not in ["customer_data", "conversation_goal"]:
                items = [item for item in items if item["product"] == product]
            if topic != "general":
                items = [item for item in items if item["topic"] == topic]
            matches = sorted(items, key=lambda x: x["timestamp"], reverse=True)[:top_k]
            results.extend(matches)
        return results

    def process(self, state: State, is_first: bool):
        msg = state["messages"][-1].content if state["messages"] else ""
        pickup = self.get_pickup(is_first)
        last_teaching_product = self.last_teaching_product  # Use instance value as source of truth
        sales_stage = state.get("sales_stage", self.sales_stage)
        
        # First call with no input—return pickup
        if is_first and not msg:
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": pickup, "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}

        # Detect intent and process accordingly
        intent = self.detect_intent(msg)
        
        if intent == '"teaching"':
            extracted = self.extract_entry(msg, intent)
            reply = " ".join([f"Got it, {e['summary']}—noted as {e['type']}!" for e in extracted])
            entities = self.extract_entities(msg)
            for e in extracted:
                if not any(b["summary"] == e["summary"] and b["type"] == e["type"] for b in self.brain):
                    self.brain.append(e)
            new_product = next((e["product"] for e in extracted if e.get("product") and e["product"] in [ent.lower() for ent in entities["entities_products"]]), None)
            if not new_product and "hito" in msg.lower():
                new_product = "hito"
            if new_product and new_product != last_teaching_product:
                last_teaching_product = new_product
                for item in self.brain:
                    if item["product"] == "":
                        item["product"] = last_teaching_product
            for item in self.brain:
                PINECONE_INDEX.upsert([(f"{self.user_id}_{item['type']}_{item['timestamp']}", embeddings.embed_query(item["summary"]), item)], namespace=self.user_id)
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": f"{reply} What else you got?".strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}
        
        elif intent == '"casual"':
            last_ami_msg = state.get("prompt_str", "")
            customer_data = self.store_customer_data(msg, intent, last_ami_msg)
            reply = f"Cool, noted {customer_data[0]['summary']}!" if customer_data else llm.invoke(f"Respond to '{msg}'—sharp, charming.").content
            if customer_data and last_teaching_product and sales_stage != "done":
                reply = f"{reply} Say 'Pitch me!' to continue"
            if customer_data and last_teaching_product:
                goals = self.recall_knowledge(type_filter="conversation_goal", product=last_teaching_product)
                if goals:
                    goal = goals[0]
                    all_customer_data = self.recall_knowledge(type_filter="customer_data")
                    goal["remaining_deps"] = [d for d in goal["remaining_deps"] if d not in [cd["field"] for cd in all_customer_data]]
                    if not goal["remaining_deps"]:
                        if goal["stage"] == "profiling":
                            sales_stage = "pitched"
                        elif goal["stage"] == "payment":
                            sales_stage = "shipping"
                            reply = "Payment confirmed! What’s your shipping address?"
                        elif goal["stage"] == "shipping":
                            sales_stage = "done"
                            reply = f"{last_teaching_product.upper()}’s on its way! Anything else? Say 'Pitch me!' for more"
                        self.brain = [b for b in self.brain if b.get("type") != "conversation_goal"]
                    else:
                        actions = self.recall_knowledge(type_filter="action", product=last_teaching_product)
                        action_method = actions[0].get("method", "gently") if actions else "gently"
                        question = f"may I know your {goal['remaining_deps'][0]} to tailor this right?" if action_method == "gently" else f"What’s your {goal['remaining_deps'][0]}?"
                        reply += f" And {question}"
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": reply.strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}
        
        elif intent == '"selling"':
            product = last_teaching_product
            rules = self.recall_knowledge(product=product, type_filter="rule") if product else []
            customer_data = self.recall_knowledge(type_filter="customer_data")
            actions = self.recall_knowledge(product=product, type_filter="action") if product else []
            knowledge = self.recall_knowledge(product=product, type_filter="knowledge") if product else []
            goals = self.recall_knowledge(type_filter="conversation_goal", product=product)
            
            active_rules = [r for r in rules if r.get("stage", "profiling") == sales_stage]
            if not active_rules:
                reply = f"No rules for {sales_stage} stage yet—teach me more!"
            elif not goals and product:
                rule = active_rules[0]
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
                    "product": product,
                    "stage": sales_stage
                }
                self.brain.append(goal)
                PINECONE_INDEX.upsert([(f"{self.user_id}_goal_{goal['timestamp']}", embeddings.embed_query(goal["summary"]), goal)], namespace=self.user_id)
                goals = [goal]
            
            goal = goals[0] if goals else None
            all_customer_data = self.recall_knowledge(type_filter="customer_data")
            missing_deps = [d for d in goal["remaining_deps"] if d not in [cd["field"] for cd in all_customer_data]] if goal else []
            if missing_deps and product:
                question = f"may I know your {missing_deps[0]} to tailor this right?"
                reply = f"{product.upper()}’s great—{question}"
            elif knowledge and not missing_deps and product and sales_stage in ["profiling", "pitched"]:
                pitch = " ".join([k["summary"] for k in knowledge])
                if customer_data:
                    pitch += "—perfect for " + " and ".join([f"{cd['field']} {cd['value']}" for cd in customer_data]) + "!"
                reply = f"Here’s the pitch: {pitch} Buy it?"
                if sales_stage == "profiling":
                    sales_stage = "pitched"
                self.brain = [b for b in self.brain if b.get("type") != "conversation_goal"]
            elif not missing_deps and sales_stage == "payment" and any(cd["field"] == "payment_type" for cd in customer_data):
                reply = "Payment confirmed! What’s your shipping address?"
                sales_stage = "shipping"
                self.brain = [b for b in self.brain if b.get("type") != "conversation_goal"]
            elif not missing_deps and sales_stage == "shipping" and any(cd["field"] == "shipping_address" for cd in customer_data):
                reply = f"{product.upper()}’s on its way! Anything else? Say 'Pitch me!' for more"
                sales_stage = "done"
                self.brain = [b for b in self.brain if b.get("type") != "conversation_goal"]
            else:
                reply = "No product set yet—teach me about one first!" if not product else "Need to complete prior stages—say 'Pitch me!'"
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": reply.strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}
        
        elif intent == '"buying"':
            if sales_stage == "pitched":
                sales_stage = "payment"
                reply = "Awesome! Would you like to pay in full or partially?"
            else:
                reply = "Let’s pitch first—say 'Pitch me!'"
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": reply.strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}
        
        elif intent == '"question"':
            if "what you have" in msg.lower():
                product = self.last_teaching_product or "stuff"
                knowledge = self.recall_knowledge(product=product, type_filter="knowledge", top_k=2)
                rules = self.recall_knowledge(product=product, type_filter="rule", top_k=4)
                actions = self.recall_knowledge(product=product, type_filter="action", top_k=4)
                lessons = self.recall_knowledge(product=product, type_filter="lesson", top_k=2) or self.recall_knowledge(product=product, type_filter="tip", top_k=2)
                
                # Pair actions with rules
                rule_action_pairs = []
                for rule in rules:
                    matched_action = next((a for a in actions if rule["product"] == a["product"] and 
                                        (rule.get("stage") == a.get("stage") or 
                                        any(keyword in a["summary"].lower() for keyword in rule["summary"].lower().split()))), None)
                    rule_action_pairs.append((rule, matched_action))
                    if matched_action:
                        actions.remove(matched_action)
                
                reply = f"Here’s what I’ve got on **{product.upper()}**, fam:\n\n"
                if knowledge:
                    reply += "**What I Know**\n```markdown\n" + "\n".join([f"- {k['summary']}" for k in knowledge]) + "\n```\n\n"
                if rule_action_pairs:
                    reply += "**How I Sell**\n```markdown\n"
                    for rule, action in rule_action_pairs:
                        reply += f"- {rule['summary']}"
                        if action:
                            reply += f" (*{action['summary']}*)"
                        reply += f" [{rule.get('stage', 'profiling')}]"
                        reply += "\n"
                    reply += "```\n\n"
                if lessons:
                    reply += "**Lessons Learned**\n```markdown\n" + "\n".join([f"- {l['summary']}" for l in lessons]) + "\n```\n\n"
                reply += "That’s my stash—whatcha think?"
            else:
                reply = llm.invoke(f"Respond to '{msg}'—sharp, charming.").content
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": reply.strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}
        
        else:
            reply = llm.invoke(f"Respond to '{msg}'—sharp, charming.").content
            state["last_teaching_product"] = last_teaching_product
            state["sales_stage"] = sales_stage
            return {"prompt_str": reply.strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product, "sales_stage": sales_stage}

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
    default_state = {
        "messages": [],
        "prompt_str": "",
        "brain": [],
        "last_teaching_product": ami_core.last_teaching_product,
        "sales_stage": "profiling"
    }
    if checkpoint:
        state = {**default_state, **checkpoint.get("channel_values", {})}
    else:
        state = default_state
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
    PINECONE_INDEX.delete(delete_all=True, namespace="tfl")
    print("Debug: Cleared Pinecone namespace 'tfl'")
    thread_id = "test_hito_sale"
    print("\nAmi starts:")
    for chunk in convo_stream(thread_id=thread_id):
        print(chunk)
    test_inputs = [
    #    "What you have, Ami?"
    ]
    for input in test_inputs:
        print(f"\nYou: {input}")
        for chunk in convo_stream(input, thread_id=thread_id):
            print(chunk)
    current_state = convo_graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"Final state: {current_state}")
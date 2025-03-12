# ami_core_v1.5_with_ner_brain
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 12, 2025
# Purpose: Ami's Training Path + Selling Path with NER-enriched brain—multilingual, product-tagged, Pinecone-ready

import json
import time
import pinecone
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
llm = ChatOpenAI(model="gpt-4o", streaming=True)
llm_non_streaming = ChatOpenAI(model="gpt-4o", streaming=False)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
PINECONE_INDEX = pinecone_index

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    brain: list
    last_teaching_product: str

# Core Logic
class AmiCore:
    def __init__(self):
        self.brain = []
        self.user_id = "tfl"
        self.presets = {
            "situation": lambda x: f"Tell me about {x} in your current setup.",
            "challenger": lambda x: f"Reframe it like this: {x}"
        }

    def get_pickup(self, is_first):
        if not is_first: return ""
        brain_size = len(PINECONE_INDEX.query(vector=embeddings.embed_query("sales"), top_k=1000, namespace=self.user_id).matches)
        if brain_size == 0:
            return "Hey, I’m Ami—loaded with sales basics, but I need your edge. What’s your trick?"
        elif brain_size < 5:
            return "Hey, I’m Ami—I’ve got some hooks, but I need your killer move. What’s it?"
        return "Hey, I’m Ami—think I’ve nailed your game. Try me?"

    def detect_intent(self, msg):
        return llm.invoke(f"""Analyze '{msg}' and return ONLY one in quotes: "greeting", "teaching", "question", "exit", "casual", "selling".
        - "greeting": Explicit hellos (e.g., "Hey!", "Hi there").
        - "teaching": Factual statements, product info, or tips (e.g., "The CRM syncs in real-time", "HITO is a calcium supplement").
        - "question": Queries (e.g., "How do you close deals?").
        - "exit": Goodbyes (e.g., "Later!").
        - "casual": Chit-chat or vague (e.g., "Nice day").
        - "selling": Explicit pitch requests ONLY (e.g., "Pitch me!", "Sell it!").
        Prioritize content over length—facts default to "teaching". DO NOT FREESTYLE.""").content.strip()

    def extract_entities(self, text):
        try:
            response = llm_non_streaming.invoke(f"""Extract named entities from '{text}' and return as a JSON dict with keys:
            - 'entities_products': list of product names
            - 'entities_organizations': list of organization names
            - 'entities_locations': list of location names
            Return empty lists if none found. Examples:
            - "HITO is great" → ```json{{"entities_products": ["HITO"], "entities_organizations": [], "entities_locations": []}}```
            - "CLB Bóng đá Hoang Anh Gia Lai uses it" → ```json{{"entities_products": [], "entities_organizations": ["Hoang Anh Gia Lai Football Club"], "entities_locations": []}}```
            Wrap output in ```json``` and ``` markers. Return ONLY valid JSON inside markers. DO NOT FREESTYLE.""").content.strip()
            print(f"Debug: Raw NER response: {response}")
            start = response.find("```json") + 7
            end = response.rfind("```")
            if start > 6 and end > start:
                json_str = response[start:end].strip()
                return json.loads(json_str)
            raise ValueError("No valid JSON found in response")
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Debug: NER failed with {e}, returning default entities")
            return {"entities_products": [], "entities_organizations": [], "entities_locations": []}

    def extract_topic(self, summary, entities):
        if entities["entities_products"]:
            return entities["entities_products"][0].lower()  # e.g., "hito"
        if entities["entities_organizations"]:
            return "endorsement"
        return llm.invoke(f"""Extract the main topic from '{summary}' as a single word or short phrase (e.g., "crm", "closing"). 
        Return "general" if unclear or empty. DO NOT FREESTYLE.""").content.strip().lower()

    def extract_subtopic(self, summary, entities):
        summary_lower = summary.lower()
        if entities["entities_products"]:
            if "supplement" in summary_lower or "calcium" in summary_lower:
                return "product"
            if "premium" in summary_lower or "formula" in summary_lower:
                return "quality"
        if entities["entities_organizations"]:
            return "endorsement"
        return "general"

    def extract_knowledge(self, msg):
        chunks = [msg]
        if len(msg.split('.')) > 3:
            chunks = [c.strip() for c in msg.split('.') if c.strip()]
        entities = self.extract_entities(msg)
        product = entities["entities_products"][0].lower() if entities["entities_products"] else None
        entries = []
        for chunk in chunks:
            text = chunk
            summary = llm.invoke(f"""Translate and summarize '{text}' into concise English. 
            Keep it short, factual, no rephrasing beyond translation. Do not add quotes, colons, or extra punctuation. DO NOT FREESTYLE.""").content.strip()
            entries.append({"type": "knowledge", "text": text, "summary": summary, "product": product, "entities": entities})
        return entries

    def recall_knowledge(self, product=None, topic="general", type_filter=None, top_k=3):
        query_text = f"{product} knowledge" if product else f"{topic} knowledge"
        query_vector = embeddings.embed_query(query_text)
        filter_dict = {}
        if product:
            filter_dict["product"] = product
        if topic != "general":
            filter_dict["topic"] = topic
        if type_filter:
            filter_dict["type"] = type_filter
        results = PINECONE_INDEX.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.user_id,
            filter=filter_dict,
            include_metadata=True
        ).matches
        if not results:
            print(f"Debug: Recalled 0 items for product={product}, topic={topic}")
            return []
        for match in results:
            match.metadata["use_count"] += 1
            PINECONE_INDEX.upsert(
                [(match.id, embeddings.embed_query(match.metadata["summary"]), match.metadata)],
                namespace=self.user_id
            )
        unique_results = list({m.metadata["summary"]: m for m in results}.values())
        print(f"Debug: Recalled {len(unique_results)} items for product={product}, topic={topic}: {[m.metadata['summary'] for m in unique_results]}")
        return [m.metadata for m in unique_results]

    def process(self, state: State, is_first: bool):
        msg = state["messages"][-1].content if state["messages"] else ""
        pickup = self.get_pickup(is_first)
        last_teaching_product = state.get("last_teaching_product", None)
        if is_first and not msg:
            print(f"Debug: Initial pickup - {pickup}")
            return {"prompt_str": pickup, "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product}

        intent = self.detect_intent(msg)
        print(f"Debug: Intent={intent}, Msg={msg}")

        if intent == '"teaching"':
            extracted = self.extract_knowledge(msg)
            print(f"Debug: Extracted={json.dumps(extracted)}")
            reply_parts = []
            teaching_product = extracted[0]["product"]
            for entry in extracted:
                entities = entry["entities"]
                topic = self.extract_topic(entry["summary"], entities)
                subtopic = self.extract_subtopic(entry["summary"], entities)
                base_topic = topic.split()[0] if " " in topic else topic
                metadata = {
                    "type": entry["type"],
                    "text": entry["text"],
                    "summary": entry["summary"],
                    "topic": base_topic,
                    "subtopic": subtopic,
                    "product": entry["product"] if entry["product"] is not None else "",
                    "entities_products": entities["entities_products"],
                    "entities_organizations": entities["entities_organizations"],
                    "entities_locations": entities["entities_locations"],
                    "source": "user",
                    "context": "",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 1.0,
                    "use_count": 0,
                    "tags": []
                }
                check = PINECONE_INDEX.query(
                    vector=embeddings.embed_query(entry["summary"]),
                    top_k=1,
                    namespace=self.user_id,
                    filter={"summary": entry["summary"]}
                ).matches
                if not check:
                    self.brain.append(metadata)
                    upsert_id = f"{self.user_id}_{metadata['type']}_{metadata['timestamp']}"
                    upsert_result = PINECONE_INDEX.upsert(
                        [(upsert_id, embeddings.embed_query(entry["summary"]), metadata)],
                        namespace=self.user_id
                    )
                    print(f"Debug: Upserted {upsert_id} - Result: {upsert_result}")
                    time.sleep(1)
                else:
                    print(f"Debug: Skipped duplicate {entry['summary']}")
                reply_parts.append(f"Got it, {entry['summary']}—key {base_topic} stuff!")
            reply = " ".join(reply_parts) + " How’d that play out?"
            print(f"Debug: Teaching reply={reply}")
            return {"prompt_str": f"{pickup} {reply}".strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": teaching_product}
        elif intent == '"greeting"':
            reply = "Hey, good to connect! What’s your superpower?"
        elif intent == '"question"':
            reply = "Sharp question! Hit me with a tip to unpack it."
        elif intent == '"exit"':
            reply = "Catch you later—bring more heat next time!"
        elif intent == '"selling"':
            product = last_teaching_product
            brain_data = self.recall_knowledge(product=product, top_k=3)
            if not brain_data:
                reply = "Brain’s empty—teach me some gold first!"
            else:
                pitch = "Here’s the pitch: " + " ".join([f"{d['summary']}—{d['product'] or 'no-product'} nailed." for d in brain_data])
                reply = pitch + " Buy it?"
            print(f"Debug: Selling reply={reply}")
        else:  # casual
            reply = llm.invoke(f"""Respond to '{msg}'—sharp, charming, short, curious. 
            NO emojis, NO quotes, DO NOT FREESTYLE.""").content

        return {"prompt_str": f"{pickup} {reply}".strip(), "brain": self.brain, "messages": state["messages"], "last_teaching_product": last_teaching_product}

# Graph & Stream
ami_core = AmiCore()
graph_builder = StateGraph(State)
graph_builder.add_node("ami", lambda state: ami_core.process(state, not state.get("messages", [])))
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input=None, thread_id="learning_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}}) or {}
    history = checkpoint.get("channel_values", {}).get("messages", [])
    last_teaching_product = checkpoint.get("channel_values", {}).get("last_teaching_product", None)
    state = {
        "messages": history + [HumanMessage(content=user_input or "")] if user_input or history else [],
        "prompt_str": "",
        "brain": checkpoint.get("channel_values", {}).get("brain", []),
        "last_teaching_product": last_teaching_product
    }
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    for line in state["prompt_str"].split('\n'):
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")

# Test
if __name__ == "__main__":
    print("\nAmi starts:")
    for chunk in convo_stream():
        print(chunk)
    test_inputs = [
        "HITO là sản phẩm bổ sung canxi hỗ trợ phát triển chiều cao (từ 2 tuổi trở lên, đặc biệt dành cho người trưởng thành). Sản phẩm cao cấp, công thức toàn diện. Sản phẩm được CLB Bóng đá Hoàng Anh Gia Lai tin dùng.",
        "Pitch me!"
    ]
    for input in test_inputs:
        print(f"\nYou: {input}")
        for chunk in convo_stream(input, "test_thread"):
            print(chunk)
    print("\nRecalling HITO knowledge:")
    ami = AmiCore()
    hito_knowledge = ami.recall_knowledge(product="hito", type_filter="knowledge")
    for entry in hito_knowledge:
        entities_products = entry.get("entities_products", entry.get("entities", {}).get("entities_products", []))
        print(f"Recalled: {entry['summary']} (subtopic: {entry['subtopic']}, entities_products: {entities_products}, use_count: {entry['use_count']})")
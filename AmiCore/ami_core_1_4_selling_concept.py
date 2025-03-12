# ami_core_v1.4_with_selling_concept_fixed_v6
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 12, 2025
# Purpose: Ami's Training Path + Selling Path concept—proactive, intent/topic-aware, Pinecone-ready

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
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
PINECONE_INDEX = pinecone_index

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    brain: list

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
            return "Hey, I’m Ami—loaded with sales basics, but I need your company’s edge. What’s the one trick I should know to sell your stuff?"
        elif brain_size < 5:
            return "Hey, I’m Ami—I’ve got some of your sales hooks, but I’m missing the killer move. What’s your go-to?"
        return "Hey, I’m Ami—think I’ve nailed your sales game. Try me?"

    def detect_intent(self, msg):
        return llm.invoke(f"""Analyze '{msg}' and return ONLY one in quotes: "greeting", "teaching", "question", "exit", "casual", "selling".
        - "greeting" for hellos (e.g., "Hey!", "Hi there").
        - "teaching" for sales facts/techniques/outcomes (e.g., "The CRM syncs in real-time").
        - "question" for queries (e.g., "How do you close deals?", "What’s your trick?").
        - "exit" for goodbyes (e.g., "Later!", "See ya").
        - "casual" if empty or chit-chat (e.g., "Nice day").
        - "selling" for pitch requests (e.g., "Pitch me!", "Sell it!").
        DO NOT FREESTYLE.""").content.strip()

    def extract_topic(self, msg):
        return llm.invoke(f"""Extract the main topic from '{msg}' as a single word or short phrase (e.g., "crm", "closing"). 
        Return "general" if unclear or empty. DO NOT FREESTYLE.""").content.strip().lower()

    def extract_knowledge(self, msg):
        raw = llm.invoke(f"""Extract EXACT TEXT from '{msg}' into JSON:
        {{"skills": ["techniques"], "knowledge": ["facts"], "lessons": ["outcomes"]}}
        - "skills": Specific sales techniques (e.g., "Ask open-ended questions").
        - "knowledge": Factual statements (e.g., "The CRM syncs in real-time").
        - "lessons": Results or outcomes (e.g., "Deals close faster with follow-ups").
        - NO rephrasing, return empty lists if unsure, use EXACT wording.
        - Example: "Deals close faster with follow-ups" -> {{"skills": [], "knowledge": [], "lessons": ["Deals close faster with follow-ups"]}}
        - DO NOT FREESTYLE, RETURN ONLY JSON.""").content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw) if raw else {"skills": [], "knowledge": [], "lessons": []}

    def recall_knowledge(self, topic="general", type_filter=None, top_k=3):
        # Query with a vector close to stored content
        query_vector = embeddings.embed_query("The CRM syncs in real-time" if topic == "crm" else "sales knowledge")
        filter_dict = {"topic": topic} if topic != "general" else {}
        if type_filter:
            filter_dict["type"] = type_filter
        results = PINECONE_INDEX.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.user_id,
            filter=filter_dict,
            include_metadata=True
        ).matches
        for match in results:
            match.metadata["use_count"] += 1
            PINECONE_INDEX.upsert(
                [(match.id, embeddings.embed_query(match.metadata["content"]), match.metadata)],
                namespace=self.user_id
            )
        unique_results = list({m.metadata["content"]: m for m in results}.values())
        print(f"Debug: Recalled {len(unique_results)} items for topic={topic}: {[m.metadata['content'] for m in unique_results]}")
        return [m.metadata for m in unique_results]
    
    def process(self, state: State, is_first: bool):
        msg = state["messages"][-1].content if state["messages"] else ""
        pickup = self.get_pickup(is_first)
        if is_first and not msg:
            print(f"Debug: Initial pickup - {pickup}")
            return {"prompt_str": pickup, "brain": self.brain, "messages": state["messages"]}

        intent = self.detect_intent(msg)
        topic = self.extract_topic(msg)
        print(f"Debug: Intent={intent}, Topic={topic}, Msg={msg}")

        if intent == '"teaching"':
            extracted = self.extract_knowledge(msg)
            print(f"Debug: Extracted={json.dumps(extracted)}")
            for key, items in extracted.items():
                for item in items:
                    base_topic = topic.split()[0] if " " in topic else topic
                    check = PINECONE_INDEX.query(
                        vector=embeddings.embed_query(item),
                        top_k=1,
                        namespace=self.user_id,
                        filter={"content": item}
                    ).matches
                    if not check:
                        entry = {
                            "type": "knowledge" if key == "knowledge" else key[:-1],
                            "content": item,
                            "topic": base_topic,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 1.0,
                            "use_count": 0
                        }
                        self.brain.append(entry)
                        upsert_id = f"{self.user_id}_{entry['type']}_{entry['timestamp']}"
                        upsert_result = PINECONE_INDEX.upsert(
                            [(upsert_id, embeddings.embed_query(item), entry)],
                            namespace=self.user_id
                        )
                        print(f"Debug: Upserted {upsert_id} - Result: {upsert_result}")
                        time.sleep(1)
                    else:
                        print(f"Debug: Skipped duplicate {item}")
            reply_parts = []
            base_topic = topic.split()[0] if " " in topic else topic  # Use base_topic in reply
            if extracted["knowledge"]:
                reply_parts.extend([f"Got it, {k}—key {base_topic} stuff!" for k in extracted["knowledge"]])
            if extracted["skills"]:
                reply_parts.extend([f"{self.presets['situation'](s)} Slick {base_topic} move!" for s in extracted["skills"]])
            if extracted["lessons"]:
                reply_parts.extend([f"{self.presets['challenger'](l)} Clutch {base_topic} win!" for l in extracted["lessons"]])
            reply = " ".join(reply_parts) or f"Spill more {base_topic} gold—I’m hooked!"
            reply += " How’d that play out?"
            print(f"Debug: Teaching reply={reply}")
        elif intent == '"greeting"':
            reply = "Hey, good to connect! What’s your sales superpower?"
            print(f"Debug: Greeting reply={reply}")
        elif intent == '"question"':
            reply = "Sharp question! Hit me with your best sales tip to unpack it."
            print(f"Debug: Question reply={reply}")
        elif intent == '"exit"':
            reply = "Catch you later—bring more sales heat next time!"
            print(f"Debug: Exit reply={reply}")
        elif intent == '"selling"':
            brain_data = self.recall_knowledge(topic=topic, top_k=3)
            if not brain_data:
                reply = "Brain’s empty—teach me some sales gold first!"
            else:
                pitch = "Here’s the pitch: " + " ".join([f"{d['content']}—{d['topic']} nailed." for d in brain_data])
                reply = pitch + " Buy it?"
            print(f"Debug: Selling reply={reply}")
        else:  # casual
            reply = llm.invoke(f"""Respond to '{msg}'—sharp, charming, short, curious, about {topic}. 
            NO emojis, NO quotes, NO rephrasing, DO NOT FREESTYLE.""").content
            print(f"Debug: Casual reply={reply}")

        return {"prompt_str": f"{pickup} {reply}".strip(), "brain": self.brain, "messages": state["messages"]}

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
    state = {"messages": history + [HumanMessage(content=user_input or "")] if user_input or history else [],
             "prompt_str": "", "brain": checkpoint.get("channel_values", {}).get("brain", [])}
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    for line in state["prompt_str"].split('\n'):
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")

# Test
if __name__ == "__main__":
    #PINECONE_INDEX.delete(delete_all=True, namespace="tfl")
    #print("Debug: Cleared Pinecone namespace 'tfl'")
    print("\nAmi starts:")
    for chunk in convo_stream():
        print(chunk)
    test_inputs = [
        "Hey!",
        "The CRM syncs in real-time",
        "Deals close faster with follow-ups",
        "How do you close deals?",
        "Pitch me!",
        "Later!"
    ]
    for input in test_inputs:
        print(f"\nYou: {input}")
        for chunk in convo_stream(input, "test_thread"):
            print(chunk)
    print("\nRecalling CRM knowledge:")
    ami = AmiCore()
    crm_knowledge = ami.recall_knowledge(topic="crm", type_filter="knowledge")
    for entry in crm_knowledge:
        print(f"Recalled: {entry['content']} (use_count: {entry['use_count']})")
    print("\nRaw Pinecone check:")
    raw_results = PINECONE_INDEX.query(
        vector=embeddings.embed_query("The CRM syncs in real-time"),  # Match stored vector
        top_k=10,
        namespace="tfl",
        include_metadata=True
    ).matches
    for r in raw_results:
        print(f"Stored: {r.metadata['content']} (topic: {r.metadata['topic']})")
    print(f"Debug: Total vectors in 'tfl': {PINECONE_INDEX.describe_index_stats()['namespaces']['tfl']['vector_count'] if 'tfl' in PINECONE_INDEX.describe_index_stats()['namespaces'] else 0}")
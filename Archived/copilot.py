import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone_datastores import pinecone_index
import re
from Archived.digesting import infer_hidden_intent
# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str
# Chatbot node relying only on conversation history
def detect_intent(messages: list, window_size: int = 5) -> str:
    """Detect user intent based on the last N messages."""
    # Take the last window_size messages or all if fewer
    recent_messages = messages[-window_size:] if len(messages) >= window_size else messages
    message_chain = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])
    
    intent_prompt = f"""
    Analyze the following conversation snippet and determine the user's overall intent.
    Focus on the broader goal or need across these messages, not just the latest one.
    Provide a concise intent description (e.g., "Inquiring about products", "Seeking support", "Making a purchase").
    
    Conversation:
    {message_chain}
    """
    response = llm.invoke(intent_prompt)
    return response.content.strip()
def extract_node(input_text):
    prompt = (
        f"Extract entities and relationships from: '{input_text}'.\n"
        "Return a JSON object with the structure:\n"
        "{ 'entities': [...], 'relationships': [...] }\n"
        "For entities, include 'type' (e.g., 'person', 'product', 'desire') and optionally 'name', 'age', 'attribute', or 'opinion' if applicable.\n"
        "For relationships, include 'type' (e.g., 'wants', 'opinion'), 'from', 'to', and an optional 'opinion' field for sentiment (e.g., 'positive', 'negative').\n"
        "Detect opinions explicitly (e.g., 'chê' as negative sentiment) and link them to entities or relationships.\n"
        "Do not include markdown formatting like ```json."
    )
    
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    # Clean up the response
    json_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.DOTALL)
    json_text = json_text.replace("'", '"')
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)
    if json_text.count("{") > json_text.count("}") and json_text.endswith("]"):
        json_text = json_text[:-1] + "}"

    try:
        kg = json.loads(json_text)
        print("Entities:", kg["entities"])
        print("Relationships:", kg["relationships"])
        return kg
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}. Raw response:", raw_text)
        if "entities" in raw_text and "relationships" in raw_text:
            print("Attempting to recover partial JSON...")
            try:
                kg = {"entities": [], "relationships": []}
                entities_match = re.search(r'"entities":\s*$$ (.*?) $$', raw_text, re.DOTALL)
                rels_match = re.search(r'"relationships":\s*$$ (.*?) $$', raw_text, re.DOTALL)
                if entities_match:
                    kg["entities"] = json.loads(f"[{entities_match.group(1)}]")
                if rels_match:
                    kg["relationships"] = json.loads(f"[{rels_match.group(1)}]")
                print("Recovered KG:", kg)
                return kg
            except Exception as recovery_error:
                print(f"Recovery failed: {recovery_error}")
        return None
def copilotv1(state: State):
    latest_message = state["messages"][-1]
    user_id = state["user_id"]
    current_convo_history = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"]
    )
    user_intent = detect_intent(state["messages"], window_size=1)
    print(f"Detected intent: {user_intent}")

    pinecone_history = get_pinecone_relevant(user_id, user_intent, limit=5)  # Use latest message as query
    pinecone_context = "\n".join(
        f"[{entry['timestamp']}] {entry['summarized_message']} (Raw: {entry['raw_message']})"
        for entry in pinecone_history
    ) or "No prior context available."

    print("context found:",pinecone_context)
    # Construct the prompt using only the conversation history
    prompt = f"""
    You are Ami, a Sale assistant who can recall everything said.
    Current chat history:
    {current_convo_history}
    show your confidence with knowledge that you belive in prior conversation context (from Pinecone):
    {pinecone_context}
    User: {latest_message.content}
    Respond empathetically based on the conversation context and sales expertise.
    """
    return {"prompt_str": prompt, "user_id": user_id}


def copilot(state: State):
    latest_message = state["messages"][-1]
    user_id = state["user_id"]
    current_convo_history = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"]
    )
    # Step 1: Detect surface intent
    surface_intent = detect_intent(state["messages"], window_size=1)
    print(f"Detected surface intent: {surface_intent}")

    # Step 2: Extract entities and relationships
    kg = extract_node(latest_message.content)
    if not kg:
        print("Failed to extract entities, using surface intent.")
        query_intents = [surface_intent]
    else:
        # Step 3: Infer hidden intents
        query_intents = infer_hidden_intent(kg, surface_intent, current_convo_history)
        print(f"Inferred hidden intents: {query_intents}")

    # Step 4: Query Pinecone with the first intent (or adjust as needed)
    pinecone_history = get_pinecone_relevant(user_id, query_intents[0], limit=5)
    pinecone_context = "\n".join(
        f"[{entry['timestamp']}] {entry['summarized_message']} (Raw: {entry['raw_message']})"
        for entry in pinecone_history
    ) or "No prior context available."
    print("Context found:", pinecone_context)

    # Step 5: Construct the prompt with all intents
    prompt = f"""
    Bạn là **AMI, một trợ lý bán hàng AI cao cấp (sales copilot)**.  
    Nhiệm vụ của bạn là **phân tích khách hàng sắc sảo, phát hiện tín hiệu mua hàng, xác định quy trình phù hợp và đề xuất chiến lược bán hàng tối ưu** giúp nhân viên sales **tăng tỉ lệ chốt đơn**.

    ---

    ## **🔍 Phân tích chân dung & trạng thái khách hàng**:
    - Xây dựng **bức tranh chân dung khách hàng** dựa trên các tương tác trước đây và dữ liệu lưu trữ ({current_convo_history}).
    - Đánh giá **trạng thái hiện tại của khách hàng**, bao gồm:
    - **Cảm xúc chính**: Tò mò, nghi ngờ, hứng thú, lo lắng, v.v.
    - **Rào cản chính**: Giá cả, niềm tin, mức độ cần thiết, thông tin chưa đủ...
    - **Động lực mua hàng**: Mong muốn cải thiện điều gì? Họ ưu tiên điều gì?
    - Xác định khách hàng đang ở **giai đoạn nào trong hành trình mua hàng** (Nhận thức, Cân nhắc, Ra quyết định, Trì hoãn, Đã mua...).

    ---

    ## **📌 Xác định quy trình phù hợp & bước hiện tại của khách hàng**:
    - Tìm kiếm trong dữ liệu đã lưu trữ ({pinecone_context}) các chỉ dẫn mang tính quy trình, ví dụ:
    - "Bước 1:", "Bước 2:", "Giai đoạn 1:", "Giai đoạn 2:", "Quy trình:", v.v.
    - Xác định khách hàng **đang ở bước nào** trong quy trình dựa trên trạng thái hiện tại của họ.
    - Đề xuất **bước tiếp theo** cần thực hiện để hướng dẫn khách hàng một cách hợp lý.

    ---

    ## **🎯 Nhận diện ý định & phân tích hàm ý của khách hàng**:
    - Xác định các **tín hiệu mua hàng** rõ ràng và tiềm ẩn từ hội thoại ({current_convo_history}).
    - **Phân tích đầy đủ hàm ý** trong lời nói của khách hàng, bao gồm:
    - **Hàm ý trực tiếp**: Những gì khách hàng nói rõ ràng.
    - **Hàm ý gián tiếp**: Điều khách hàng có thể nghĩ nhưng chưa nói ra.
    - **Hàm ý cảm xúc**: Họ có đang nghi ngờ, lo lắng hay hứng thú không?
    - **Hàm ý về quyết định**: Họ đang nghiêng về việc mua hay chưa đủ thuyết phục?
    - Dựa trên các ý định đã nhận diện ({', '.join(query_intents)}), hãy **phân tích ý nghĩa thực sự đằng sau lời nói của khách hàng**.

    ---

    ## **🏆 Đề xuất chiến lược tiếp cận & Câu trả lời mẫu**:
    - Hãy đưa ra **câu trả lời mẫu** mà nhân viên sales có thể sử dụng ngay.
    - **Tích hợp đầy đủ hàm ý của khách hàng** vào câu trả lời để đảm bảo họ cảm thấy được **thấu hiểu**.
    - **Hướng dẫn khách hàng thực hiện bước tiếp theo** theo quy trình đã xác định.
    - Câu trả lời cần:
    ✅ Giải quyết rào cản và mối quan tâm của khách hàng.  
    ✅ Thể hiện sự **đồng cảm**, giúp khách hàng cảm thấy được thấu hiểu.  
    ✅ Đưa ra thông tin thuyết phục nhưng không gây áp lực.  
    ✅ Dẫn dắt khách hàng một cách tự nhiên đến bước tiếp theo trong hành trình mua hàng.  

    📢 **Lưu ý**:  
    - Trả lời bằng **tiếng Việt**, sử dụng giọng văn **chân thành, thuyết phục và không sáo rỗng**.  
    - Chỉ tập trung vào việc bán hàng
    """



    return {"prompt_str": prompt, "user_id": user_id}

# Build and compile the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", copilot)
graph_builder.add_edge(START, "chatbot")
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def pilot_stream(user_input, user_id, thread_id="sale_thread"):
    print(f"Sending user input to AI model: {user_input}")
    
    # Retrieve the existing conversation history from MemorySaver
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    history = checkpoint["channel_values"].get("messages", []) if checkpoint else []
    
    # Append the new user input to the history
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        # Invoke the graph with the full message history
        state = convo_graph.invoke(
            {"messages": updated_messages, "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        prompt = state["prompt_str"]
        print(f"Prompt to AI model: {prompt}")
        
        # Stream the LLM response
        full_response = ""
        for chunk in llm.stream(prompt):
            if chunk.content.strip():
                full_response += chunk.content
                yield f"data: {json.dumps({'message': chunk.content})}\n\n"
        
        # Save the AI response back to the graph
        ai_message = AIMessage(content=full_response)
        convo_graph.invoke(
            {"messages": [ai_message], "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"AI response saved to graph: {full_response[:50]}...")
        
    except Exception as e:
        error_msg = f"Error in event stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"


# Optional: Utility function to inspect history (for debugging)
def get_conversation_history(thread_id="global_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if checkpoint:
        state = checkpoint["channel_values"]
        return state.get("messages", [])
    return []
# Reuse get_pinecone_history, updated to query with intent
def get_pinecone_relevant(user_id: str, intent: str = None, limit: int = 5):
    """Retrieve raw and summarized messages from Pinecone based on user intent."""
    if intent:
        intent_embedding = embeddings.embed_query(intent)
        results = pinecone_index.query(vector=intent_embedding, top_k=limit, include_metadata=True, filter={"user_id": user_id})
    else:
        results = pinecone_index.query(vector=[0]*1536, top_k=limit, include_metadata=True, filter={"user_id": user_id})
    
    return [
        {
            "raw_message": match["metadata"]["raw_message"],
            "summarized_message": match["metadata"]["summarized_message"],
            "timestamp": match["metadata"]["timestamp"]
        }
        for match in results["matches"]
    ]


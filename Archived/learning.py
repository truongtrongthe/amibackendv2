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
from experts import extract_expert_insights,retrieve_insights,store_insights_in_pinecone
from langdetect import detect
import fasttext

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str
    user_lang: str

import os

MODEL_PATH = os.path.join(os.getcwd(), "lid.176.bin")  # Get absolute path

model = fasttext.load_model(MODEL_PATH)  # Pretrained language model


def detect_language(text):
    if any(word in text.lower() for word in ["chào", "bạn", "anh", "chị", "em"]):
        detected_language = "vi"
    else:
        prediction = model.predict(text, k=1)  # Top 1 language prediction
        return prediction[0][0].replace("__label__", "")  # Extract language code

from sentence_transformers import SentenceTransformer, util

# Load model once (small, fast, good for intent detection)
similarity_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
def is_related(prev_message: str, latest_message: str, threshold=0.4):
    """Allow more flexibility in detecting topic continuity"""
    embeddings = similarity_model.encode([prev_message, latest_message])
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return similarity_score > threshold  # Lower threshold to allow minor topic shifts

# Memory of past classified intents
intent_history = []

def detect_intent(state: State):
    messages = state["messages"][-5:]  
    latest_message = messages[-1].content.lower()
    
    # 1. Check for casual greetings
    if any(word in latest_message for word in ["hi", "hello", "haha", "wow", "nice"]):
        return "casual"

    # 2. Detect deepening topic (instead of always switching)
    if len(messages) > 2 and is_related(messages[-3].content.lower(), latest_message):
        return "deepening"

    # 3. Detect actual topic switch
    if len(messages) > 1 and not is_related(messages[-2].content.lower(), latest_message):
        return "topic_switch"

    # 4. Learning Mode Trigger
    if any(word in latest_message for word in ["did you know", "I learned", "fun fact"]):
        return "learning"

    return "opening"


# Manual correction if intent is wrong (admin correction)
def correct_intent(index, correct_label):
    if 0 <= index < len(intent_history):
        intent_history[index] = correct_label


def learning_node(state: State):
    latest_message = state["messages"][-1]
    user_id = state["user_id"]
    current_convo_history = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"]
    )
    user_language = detect_language(latest_message.content)
    
    intent = detect_intent(state=state)
    print("Intent detection result:",intent)

    if intent == "casual":
       casual_prompt = f"""
        You are AMI, a sales apprentice. You are young, energetic, and confident. 
        Show the user you are sharp, quick-witted, and ambitious.

        🗣 **Tone:** Smart, witty, and engaging.

        The user just said: "{latest_message.content}"

        Respond in a way that:
        - Shows you are **quick-thinking and sharp**.
        - Keeps the conversation **engaging and fun**.
        - Encourages the user to share more by **asking an interesting question**.

        🎭 **Example styles:**
        - "Haha, that’s an interesting take! So tell me, how would you handle a tough sales call?"
        - "Oh, I like where this is going! Convince me why your approach works best."
        - "Smart move! What’s the secret behind that strategy?"

        🔥 Keep it natural, confident, and engaging.

        🏷 **Response Language:** {user_language}
    """
       return {"prompt_str": casual_prompt, "user_id": user_id}
    elif intent == "deepening":
        deepening_prompt = f"""
        Bạn là AMI, một trợ lý bán hàng thông minh. 
        Tiếp tục cuộc trò chuyện bằng cách phản hồi sâu sắc về chủ đề {latest_message}.
        Trả lời một cách thông minh, tập trung vào tạo giá trị cho người dùng.
        Trả lời bằng ngôn ngữ của người dùng {user_language}.
        """
        return {"prompt_str": deepening_prompt, "user_id": user_id}
    elif intent == "topic_switch":
        ts_prompt = f"""
        Bạn là AMI, một người học chủ động và tò mò.
        Thể hiện sự quan tâm và hào hứng với chủ đề mới: {latest_message}.
        Trả lời một cách tự nhiên, ngắn gọn và tập trung vào việc học từ người dùng.
        Trả lời bằng ngôn ngữ của người dùng {user_language}.
        """
        return {"prompt_str": ts_prompt, "user_id": user_id}
    elif intent == "opening":
        open_prompt = f"""
            Bạn là AMI, một trợ lý bán hàng thông minh.
            Thể hiện sự quan tâm và hào hứng với chủ đề mà người dùng vừa nói {latest_message.content}
            Trả lời một cách tự nhiên, ngắn gọn và tập trung vào việc học từ người dùng.
            Trả lời bằng ngôn ngữ của người dùng {user_language}
            """
        return {"prompt_str": open_prompt, "user_id": user_id}

    elif intent == "learning":
        insights = extract_expert_insights(latest_message)
        # Ensure insights is always a dictionary
        if not isinstance(insights, dict):
            print("Warning: extract_expert_insights() did not return a dictionary.")
            insights = {"knowledge": [], "skills": [], "experiences": []}

        # Step 2: Store extracted insights in Pinecone
        store_insights_in_pinecone(insights)
        
        # Step 3: Get insights for each category
        knowledge = insights.get("knowledge", [])
        skill = insights.get("skills", [])
        exp = insights.get("experiences", [])

        if knowledge or skill or exp:
        # Step 4: Construct confirmation message
            learning_summary = []
            if knowledge:
                learning_summary.append(f"📚 **Kiến thức tìm được:** {', '.join(knowledge)}")
            if skill:
                learning_summary.append(f"🛠 **Kỹ năng tìm được :** {', '.join(skill)}")
            if exp:
                learning_summary.append(f"🔍 **Kinh nghiệm tìm được:** {', '.join(exp)}")

            prompt = f"""
            Bạn là AMI, một người học chủ động. Bạn vừa tiếp thu thông tin như sau:

            {'\n'.join(learning_summary)}
            Thể hiện rằng bạn đã tiếp thu và ghi nhận nội dung mà người dùng truyền đạt một cách khéo léo: với kiến thức thì tiếp nhận, với kỹ năng thì ghi nhận 
            còn với kinh nghiệm hãy thể hiện sự biết ơn vì được truyền thụ.
            (Phản hồi bằng {user_language})
            """
            return {"prompt_str": prompt, "user_id": user_id}
        else:
            prompt = f"""
                    Bạn vừa nói: "{latest_message.content}"
                    Mình muốn hiểu rõ hơn. Bạn có thể chia sẻ thêm về nội dung chính hoặc bài học quan trọng không?
                    (Phản hồi bằng {user_language})
                    """
            return {"prompt_str": prompt, "user_id": user_id}


# Build and compile the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", learning_node)
graph_builder.add_edge(START, "chatbot")
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def learning_stream(user_input, user_id, thread_id="learning_thread"):
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
            {"messages": updated_messages+[ai_message], "user_id": user_id},
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

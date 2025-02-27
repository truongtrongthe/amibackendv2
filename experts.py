from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import supabase
import os
from dotenv import load_dotenv
from skills import search_relevant_knowledge
from skills import save_skills_to_db
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uuid

client = OpenAI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = "us-east-1"  # Check Pinecone console for your region


index_name = "ami-knowledge"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [i['name'] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

conversation_state = {} 



load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

llm = ChatOpenAI(model="gpt-4o", streaming=True)

chat_history = ChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=chat_history, memory_key="history", return_messages=True)


expert_id = "thetruong"


def classify_message(user_input: str) -> str:
    print("Classifying message:", user_input)
    
    prompt = f"""
    Dựa trên nội dung sau, hãy phân loại nó vào một trong ba nhóm:
    1. 'conversation' nếu đây là tin nhắn giao tiếp thông thường hoặc thảo luận.
    2. 'knowledge' nếu nó chứa thông tin quan trọng cần lưu lại.
    3. 'ambiguous' nếu nội dung chưa rõ ràng và cần hỏi thêm để xác định.

    Tin nhắn: {user_input}
    
    Trả lời chỉ với một từ: 'conversation', 'knowledge' hoặc 'ambiguous'.
    """
    
    response = llm.invoke(prompt)
    classified_type = response.content.strip().lower()
    
    print("Classified as:", classified_type)
    return classified_type


def extract_key_points(user_input: str) -> str:
    """
    Tóm tắt kiến thức thành các điểm chính.
    """
    prompt = f"""
    Hãy trích xuất các ý chính quan trọng từ đoạn sau:
    "{user_input}"

    Đưa ra kết quả dưới dạng danh sách gạch đầu dòng.
    """
    response = llm.invoke(prompt)
    print(response.content.strip())
    return response.content.strip()

def expert_chat_function_old(user_input: str, expert_id="thetruong"):
   

    history = memory.load_memory_variables({}).get("history", [])

    # Kiểm tra nếu đang chờ user xác nhận kiến thức
    if conversation_state.get("waiting_for_confirmation"):
        return confirm_knowledge(user_input, expert_id)

    # Phân loại tin nhắn (Giao tiếp / Kiến thức)
    message_type = classify_message(user_input)

    if message_type == "conversation":
        print("answering for conversation")
        # Nếu là giao tiếp, phản hồi ngay
        prompt = f"""
        Dựa vào cuộc hội thoại sau, hãy phản hồi một cách tự nhiên:

        {history}

        Tin nhắn mới từ chuyên gia:
        {user_input}

        Trả lời chỉ khi phát hiện nội dung quan trọng. Nếu có thể trích xuất kỹ năng, hãy liệt kê dưới dạng gạch đầu dòng.
        """
        response = llm.invoke(prompt)
        memory.save_context({"input": user_input}, {"output": response.content})
        return response.content

    elif message_type == "knowledge":
        # Nếu là kiến thức, trích xuất ý chính
        key_points = extract_key_points(user_input)

        # Hỏi chuyên gia có muốn bổ sung hoặc thay đổi gì không
        response_message = f"""
        📝 Tôi hiểu rằng bạn muốn lưu kiến thức này:
        {key_points}

        Bạn có muốn bổ sung hoặc thay đổi gì không?
        (Nhập nội dung bổ sung hoặc 'Không' để xác nhận.)
        """

        # Chuyển trạng thái chờ xác nhận
        conversation_state["waiting_for_confirmation"] = True
        conversation_state["pending_knowledge"] = user_input
        conversation_state["extracted_key_points"] = key_points

        return response_message

    else:
        # Nếu chưa đủ thông tin, không phản hồi
        return None

def summarize_content(text):
    """Tóm tắt nội dung kiến thức bằng GPT-4."""
    prompt = f"""
    Dưới đây là nội dung kiến thức mà chuyên gia đã chia sẻ. Hãy tóm tắt ngắn gọn và xúc tích, giữ lại các điểm quan trọng nhất.

    📜 **Nội dung gốc**:
    {text}

    📝 **Tóm tắt**:
    """
    summary = llm.invoke(prompt).content.strip()
    return summary

def expert_chat_function_old(user_input: str, expert_id="thetruong"):
    global conversation_state  

    history = memory.load_memory_variables({}).get("history", [])

    # Kiểm tra nếu đang chờ user xác nhận kiến thức
    if conversation_state.get("waiting_for_confirmation"):
        print("waiting for confirmation here")
        return confirm_knowledge(user_input, expert_id)

    # Phân loại tin nhắn (Giao tiếp / Kiến thức)
    message_type = classify_message(user_input)

    if message_type == "conversation":
        print("answering for conversation")
        
        # Prompt mới: Tạo phản hồi tự nhiên thay vì câu chung chung
        prompt = f"""
        Đây là cuộc hội thoại giữa tôi và chuyên gia. Hãy trả lời một cách tự nhiên, phù hợp với ngữ cảnh:

        📜 **Lịch sử hội thoại**:
        {history}

        🗣 **Tin nhắn mới từ chuyên gia**:
        "{user_input}"

        🎯 **Cách phản hồi mong muốn**:
        - Nếu tin nhắn là lời chào hoặc giao tiếp thông thường → Hãy trả lời NGẮN GỌN, ẤM ÁP, TÍCH CỰC, và **kèm theo một câu khơi gợi hội thoại**.
        - Luôn đóng vai trò là AMI, một trợ lý AI ham học hỏi.  
        - KHÔNG BAO GIỜ trả lời rằng "chưa có thông tin để phản hồi".  
        - Nếu có thể, hãy chủ động hỏi thêm chuyên gia về một chủ đề liên quan.  

        🚀 **Ví dụ cách trả lời**:
        - User: "Good morning" → AMI: "Chào buổi sáng! Hôm nay anh có điều gì thú vị muốn chia sẻ không?"
        - User: "Chào AMI!" → AMI: "Chào anh! Tôi đang sẵn sàng để học thêm kiến thức mới từ anh đây!"  
        """

        response = llm.invoke(prompt)
        memory.save_context({"input": user_input}, {"output": response.content})
        return response.content

    elif message_type == "knowledge":
        key_points = extract_key_points(user_input)
        # Lưu kiến thức vào Pinecone
        index = pc.Index(index_name)
        topic_id = find_similar_topic(user_input)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if topic_id:
            existing_metadata = fetch_metadata(topic_id)
            updated_content = existing_metadata.get("content", "") + "\n" + user_input
            summary = summarize_content(updated_content)  # 🟢 Tóm tắt nội dung
            updated_metadata = {
                "content": updated_content,
                "summary": summary,  # 🔥 Lưu tóm tắt vào Pinecone
                "last_updated": timestamp
            }
            index.upsert([(topic_id, get_embedding(updated_metadata["content"]), updated_metadata)])
            response_message = f"✅ Đã cập nhật kiến thức vào chủ đề: {topic_id}"
        else:
            new_id = f"topic-{uuid.uuid4().hex}"
            summary = summarize_content(user_input)  # 🟢 Tóm tắt ngay khi tạo chủ đề mới
            new_metadata = {"content": user_input, "summary": summary, "last_updated": timestamp}
            index.upsert([(new_id, get_embedding(user_input), new_metadata)])
            response_message = f"✅ Tạo chủ đề mới: {new_id}"

        conversation_state.update({
            "waiting_for_confirmation": True,
            "pending_knowledge": user_input,
            "extracted_key_points": key_points
        })

        return response_message + "\nBạn có muốn bổ sung hoặc thay đổi gì không? (Nhập nội dung bổ sung hoặc 'Không' để xác nhận.)"
    elif message_type == "ambiguous":
        # Nếu không rõ, hỏi lại user
        return f'🤔 Tôi chưa hiểu rõ ý của bạn. Bạn có thể giải thích thêm về câu này không? "{user_input}"'

    else:
        return None  

def expert_chat_function(user_input: str, expert_id="thetruong"):
    global conversation_state  

    history = memory.load_memory_variables({}).get("history", [])

    # Kiểm tra nếu đang chờ user xác nhận kiến thức
    if conversation_state.get("waiting_for_confirmation"):
        print("waiting for confirmation here")
        return confirm_knowledge(user_input, expert_id)

    # Phân loại tin nhắn (Giao tiếp / Kiến thức)
    message_type = classify_message(user_input)

    if message_type == "conversation":
        print("answering for conversation")
        
        prompt = f"""
        Đây là cuộc hội thoại giữa tôi và chuyên gia. Hãy trả lời một cách tự nhiên, phù hợp với ngữ cảnh:

        📜 **Lịch sử hội thoại**:
        {history}

        🗣 **Tin nhắn mới từ chuyên gia**:
        "{user_input}"

        🎯 **Cách phản hồi mong muốn**:
        - Nếu tin nhắn là lời chào hoặc giao tiếp thông thường → Hãy trả lời NGẮN GỌN, ẤM ÁP, TÍCH CỰC, và **kèm theo một câu khơi gợi hội thoại**.
        - Luôn đóng vai trò là AMI, một trợ lý AI ham học hỏi.  
        - KHÔNG BAO GIỜ trả lời rằng "chưa có thông tin để phản hồi".  
        - Nếu có thể, hãy chủ động hỏi thêm chuyên gia về một chủ đề liên quan.  

        🚀 **Ví dụ cách trả lời**:
        - User: "Good morning" → AMI: "Chào buổi sáng! Hôm nay anh có điều gì thú vị muốn chia sẻ không?"
        - User: "Chào AMI!" → AMI: "Chào anh! Tôi đang sẵn sàng để học thêm kiến thức mới từ anh đây!"  
        """

        response = llm.invoke(prompt)
        memory.save_context({"input": user_input}, {"output": response.content})
        return response.content

    elif message_type == "knowledge":
        key_points = extract_key_points(user_input)
        index = pc.Index(index_name)
        topic_id = find_similar_topic(user_input)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if topic_id:
            existing_metadata = fetch_metadata(topic_id)
            updated_content = existing_metadata.get("content", "") + "\n" + user_input
            summary = summarize_content(updated_content)  
            
            # Kiểm tra skill mới
            new_skills = extract_new_skills(summary, existing_metadata.get("skills", []))
            if new_skills:
                existing_metadata["skills"].extend(new_skills)
            
            updated_metadata = {
                "content": updated_content,
                "summary": summary,
                "skills": existing_metadata.get("skills", []),  # Cập nhật danh sách skill
                "last_updated": timestamp
            }
            index.upsert([(topic_id, get_embedding(updated_metadata["content"]), updated_metadata)])
            response_message = f"✅ Đã cập nhật kiến thức vào chủ đề: {topic_id}"
        else:
            new_id = f"topic-{uuid.uuid4().hex}"
            summary = summarize_content(user_input)  
            new_skills = extract_new_skills(summary, [])
            
            new_metadata = {
                "content": user_input,
                "summary": summary,
                "skills": new_skills,  # Lưu skill mới nếu có
                "last_updated": timestamp
            }
            index.upsert([(new_id, get_embedding(user_input), new_metadata)])
            response_message = f"✅ Tạo chủ đề mới: {new_id}"

        conversation_state.update({
            "waiting_for_confirmation": True,
            "pending_knowledge": user_input,
            "extracted_key_points": key_points
        })

        return response_message + "\nBạn có muốn bổ sung hoặc thay đổi gì không? (Nhập nội dung bổ sung hoặc 'Không' để xác nhận.)"
    elif message_type == "ambiguous":
        return f'🤔 Tôi chưa hiểu rõ ý của bạn. Bạn có thể giải thích thêm về câu này không? "{user_input}"'

    else:
        return None

def confirm_knowledge(user_input: str, expert_id="thetruong"):
    print("confirming knowledge here")
    """
    Xử lý phản hồi từ user khi đang chờ xác nhận kiến thức.
    """
    

    # Nếu user nhập "Không" hoặc tương tự → Lưu ngay vào database
    if user_input.lower().strip() in ["không", "không cần", "ok", "ổn rồi"]:
        print("answer falls here")
        save_skills_to_db(conversation_state["pending_knowledge"], expert_id)
        conversation_state["waiting_for_confirmation"] = False  # Kết thúc trạng thái chờ
        response_message = "✅ Kiến thức đã được lưu vào hệ thống."

    else:
        print("Keep updating knowledge here")
        # Nếu có phản hồi mới, hợp nhất & tóm tắt lại
        updated_knowledge = update_and_store_knowledge(
            conversation_state["pending_knowledge"], user_input
        )
        response_message = f"""
        📌 Kiến thức sau khi cập nhật:
        {updated_knowledge}

        Bạn có muốn bổ sung gì nữa không? (Nhập nội dung hoặc 'Không' để xác nhận.)
        """

        # Cập nhật lại kiến thức chờ xác nhận
        conversation_state["pending_knowledge"] = updated_knowledge

    return response_message

def update_and_store_knowledge(existing_knowledge: str, new_feedback: str) -> str:
    """
    Hợp nhất kiến thức cũ với phản hồi mới từ chuyên gia, sau đó tóm tắt lại.
    
    Args:
        existing_knowledge (str): Kiến thức đã trích xuất trước đó.
        new_feedback (str): Phản hồi hoặc bổ sung từ chuyên gia.

    Returns:
        str: Kiến thức đã cập nhật và tóm tắt lại.
    """
    # Kết hợp nội dung cũ và mới
    combined_knowledge = f"{existing_knowledge}\n{new_feedback}"

    # Dùng AI để tóm tắt lại kiến thức hợp nhất
    prompt = f"""
    Hãy tóm tắt lại nội dung kiến thức sau theo cách ngắn gọn và rõ ràng:
    {combined_knowledge}

    Đưa ra kết quả dưới dạng danh sách gạch đầu dòng.
    """
    updated_summary = llm.invoke(prompt).content.strip()

    # (Có thể bổ sung: lưu vào database ở đây nếu cần)

    return updated_summary

def get_embedding(skill_text):
    response = client.embeddings.create(
    input=skill_text,
    model="text-embedding-3-large",
    dimensions=1536  # Match the dimensions of ada-002
    )
    embedding_vector = response.data[0].embedding
    return embedding_vector
def find_similar_topic(new_text, threshold=0.8):
    new_embedding = get_embedding(new_text)
    index = pc.Index(index_name)
    # Tìm kiếm trong Pinecone
    results = index.query(
        index=index_name,
        vector=new_embedding,
        top_k=1,  # Lấy chủ đề gần nhất
        include_metadata=True
    )

    if results["matches"] and results["matches"][0]["score"] > threshold:
        return results["matches"][0]["id"]  # Trả về ID của chủ đề cũ
    return None  # Nếu không có chủ đề phù hợp, trả về None

def fetch_metadata(topic_id):
    index = pc.Index(index_name)
    
    fetch_result = index.fetch(ids=[topic_id])
    if topic_id in fetch_result.vectors:
        metadata = fetch_result.vectors[topic_id].metadata  # Đúng cú pháp
    else:
        metadata = {}  # Tránh lỗi nếu không tìm thấy vector
    return metadata

def update_or_create_topic(message):
    topic_id = find_similar_topic(message)
    index = pc.Index(index_name)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


    if topic_id:
        # Cập nhật chủ đề cũ
        existing_metadata = fetch_metadata(topic_id)
        updated_metadata = {
            "content": existing_metadata["content"] + "\n" + message,
            "last_updated": timestamp,
            "summary": existing_metadata.get("summary", "")
        }
        index.upsert([(topic_id, get_embedding(updated_metadata["content"]), updated_metadata)])
        print(f"✅ Cập nhật chủ đề: {topic_id}")
    else:
        # Tạo chủ đề mới
        new_id = f"topic-{uuid.uuid4().hex}"
        #new_metadata = {
        #    "content": message,
        #     "last_updated": metadata["last_updated"]
        #}
        new_metadata = {"content": message, "last_updated": timestamp, "summary": ""}

        index.upsert([(new_id, get_embedding(message), new_metadata)])
        print(f"✅ Tạo chủ đề mới: {new_id}")


import time
last_message_time = time.time()
def should_close_topic(new_message):
    global last_message_time
    if new_message.strip() == "/done":
        return True
    elapsed_time = time.time() - last_message_time
    last_message_time = time.time()
    return elapsed_time > 500  # Sau 15 phút không chat


def summarize_topic(topic_id):
    index = pc.Index(index_name)
    existing_metadata = fetch_metadata(topic_id)
    topic_text = existing_metadata["content"]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tóm tắt ngắn gọn kiến thức bán hàng từ hội thoại này."},
            {"role": "user", "content": topic_text}
        ]
    )
    summary = response.choices[0].message["content"]

    # Cập nhật Pinecone với tóm tắt
    updated_metadata = existing_metadata
    updated_metadata["summary"] = summary
    index.upsert([(topic_id, get_embedding(topic_text), updated_metadata)])

    print(f"📌 Tóm tắt lưu vào Pinecone: {summary}")

def process_message(message):
    if should_close_topic(message):
        print("🔹 Chủ đề kết thúc, tổng hợp kiến thức...")
        topic_id = find_similar_topic(message)
        if topic_id:
            summarize_topic(topic_id)
        return
    update_or_create_topic(message)
import numpy as np
from typing import List, Dict
def is_related(new_text: str, topic_embedding: List[float], threshold: float = 0.75) -> bool:
    new_embedding = np.array(get_embedding(new_text))
    topic_embedding = np.array(topic_embedding)
    similarity = np.dot(new_embedding, topic_embedding) / (np.linalg.norm(new_embedding) * np.linalg.norm(topic_embedding))
    return similarity >= threshold
def extract_new_skills(summary: str, existing_skills: list):
    """Tự động phát hiện skill mới từ nội dung tóm tắt."""
    extracted_skills = skill_extractor(summary)  # Hàm này cần định nghĩa
    new_skills = [skill for skill in extracted_skills if skill not in existing_skills]
    return new_skills
def skill_extractor(summary: str):
    """Dùng LLM để trích xuất danh sách các skill từ nội dung tóm tắt."""
    prompt = f"""
    Dựa trên đoạn nội dung sau, hãy liệt kê các kỹ năng (skills) có thể học được. 
    Chỉ trả về danh sách các skill, không giải thích thêm.

    Nội dung:
    "{summary}"

    Danh sách kỹ năng:
    """
    response = llm.invoke(prompt)
    return [skill.strip() for skill in response.content.split("\n") if skill.strip()]

import supabase
import os
from openai import OpenAI
import json
from datetime import datetime
import uuid
from pinecone import Pinecone
from pinecone import ServerlessSpec

client = OpenAI()
from langchain_openai import OpenAIEmbeddings


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

# Kết nối Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

def generate_embedding(skill_text):
    response = client.embeddings.create(
    input=skill_text,
    model="text-embedding-3-large",
    dimensions=1536  # Match the dimensions of ada-002
    )
    embedding_vector = response.data[0].embedding
    return embedding_vector

from datetime import datetime
import supabase

def save_skills_to_db(content: str, expert_id: str):
    """Lưu kiến thức vào database với embedding."""
    print("saving skills to db here")
    embedding_vector = generate_embedding(content)  # Tạo embedding
    
    data = {
        "id": str(uuid.uuid4()),
        "skill_text": content,
        "created_by": expert_id,
        "status": "pending",  # Chờ duyệt
        "embedding": embedding_vector,  # Lưu embedding vào database
        "created_at": datetime.utcnow().isoformat()
    }
    print("data:",data)

    response = supabase_client.table("skills").insert(data).execute()
    return response



def get_best_skill(query):
    query_embedding = generate_embedding(query)
    response = supabase_client.rpc("match_skills", {
        "query_embedding": query_embedding,
        "match_threshold": 0.75,
        "match_count": 3
    }).execute()
    
    skills = sorted(response.data, key=lambda x: x["score"], reverse=True)
    return skills[0] if skills else None


def search_sales_skill(query_text):   
    query_embedding = generate_embedding.embed_query(query_text)
    response = supabase_client.rpc("match_skills", {
        "query_embedding": query_embedding,
        "match_threshold": 0.75,
        "match_count": 1
    }).execute()

    print("Raw response from Supabase:", response)

    data = response.data  # Truy xuất dữ liệu mới nhất
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        skill_id = data[0].get("id")
        skill_text = data[0].get("skill_text")
        # Cập nhật số lần sử dụng kỹ năng này
        supabase_client.table("sales_skills").update({
            "usage_count": supabase_client.table("sales_skills").select("usage_count").eq("id", skill_id).execute().data[0]["usage_count"] + 1,
            "last_used": "now()"
        }).eq("id", skill_id).execute()

        return skill_text
    return "Không tìm thấy kỹ năng phù hợp."


def search_sales_skills_pinecone(query_text, max_skills=3):
    """ 
    Truy vấn kỹ năng từ Pinecone với độ chính xác cao hơn. 
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
    query_embedding = embedding_model.embed_query(query_text)

    index = pc.Index(index_name)
    response = index.query(
        vector=query_embedding,
        top_k=max_skills,
        include_metadata=True  
    )

    skills = []
    if response and "matches" in response:
        for match in response["matches"]:
            skill_text = match.get("metadata", {}).get("content")
            if skill_text:
                skills.append(skill_text)

    return skills if skills else ["Không tìm thấy kỹ năng phù hợp."]

def update_sales_skill(skill_id, new_text, user_input, original_response, feedback, updated_by="expert"):
    # Lưu lịch sử chỉnh sửa
    supabase_client.table("skill_updates").insert({
        "skill_id": skill_id,
        "old_text": original_response,
        "new_text": new_text,
        "updated_by": updated_by,
        "updated_at": "now()"
    }).execute()

    # Lưu vào tập huấn luyện nếu chuyên gia cập nhật
    if updated_by == "expert":
        supabase_client.table("training_data").insert({
            "user_input": user_input,
            "original_response": original_response,
            "improved_response": new_text,
            "feedback": feedback,
            "accepted": True  # Đánh dấu đã duyệt
        }).execute()


def suggest_skill_update(skill_id, user_feedback):
    # Lấy kỹ năng gốc
    skill_data = supabase_client.table("sales_skills").select("skill_text").eq("id", skill_id).single().execute()
    
    if not skill_data.data:
        return "Không tìm thấy kỹ năng cần cập nhật."

    original_text = skill_data.data["skill_text"]

    # Nhờ GPT-4o tối ưu lại kỹ năng dựa trên phản hồi
    improved_text = llm.invoke(f"""
        Kỹ năng gốc: {original_text}
        Phản hồi từ người dùng: {user_feedback}
        
        Hãy cải thiện kỹ năng này để phản hồi tốt hơn.
    """).content

    # Lưu bản đề xuất
    supabase_client.table("skill_updates").insert({
        "skill_id": skill_id,
        "old_text": original_text,
        "new_text": improved_text,
        "updated_by": "ami",
        "updated_at": "now()"
    }).execute()

    return improved_text

def update_skill(skill_id, success):
    skill = supabase_client.table("skills").select("*").eq("id", skill_id).execute().data[0]
    
    new_score = skill["score"] + 0.05 if success else skill["score"] - 0.1
    new_score = max(0, min(1, new_score))  # Giữ điểm trong khoảng 0-1
    
    supabase_client.table("sales_skills").update({
        "score": new_score,
        "last_updated": datetime.utcnow().isoformat()
    }).eq("id", skill_id).execute()

import numpy as np

def search_relevant_knowledge(query):
    """Tìm kiến thức liên quan bằng similarity search."""
    
    query_embedding = generate_embedding(query)  # Tạo embedding cho câu hỏi
    
    # Lấy danh sách kỹ năng đã duyệt từ Supabase
    knowledge_data = (
        supabase_client.table("knowledge")
        .select("content", "embedding")
        .eq("status", "confirmed")
        .execute()
    )

    knowledge_list = knowledge_data.data if knowledge_data.data else []

    # Tính toán cosine similarity giữa query_embedding và các embedding có sẵn
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    best_match = None
    highest_similarity = 0

    for knowledge in knowledge_list:
        similarity = cosine_similarity(query_embedding, knowledge["embedding"])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = knowledge["content"]

    return best_match if highest_similarity > 0.75 else None  # Ngưỡng 0.75 để lọc kết quả yếu

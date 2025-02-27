import supabase
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Kết nối Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

def add_skill_to_db(skill_text):

    # 1. Nhận embedding từ OpenAI
    response = client.embeddings.create(
    input=skill_text,
    model="text-embedding-3-large",
    dimensions=1536  # Match the dimensions of ada-002
    )
    embedding_vector = response.data[0].embedding
    print(embedding_vector)
    # 2. Lưu vào Supabase
    data, count = supabase_client.table("sales_skills").insert({
        "skill_text": skill_text,
        "embedding": embedding_vector
    }).execute()
    
    print(f"Đã lưu kỹ năng: {skill_text}")


def search_sales_skill(query_text):
    # 1. Tạo embedding cho câu hỏi
    query_embedding_response =client.embeddings.create(
        input=query_text,
        model="text-embedding-3-large",
        dimensions=1536  # Match the dimensions of ada-002
    )
    query_embedding = query_embedding_response.data[0].embedding

    # 2. Tìm kỹ năng gần nhất trong Supabase bằng cosine similarity
    query, _ = supabase_client.rpc("match_skills", {
        "query_embedding": query_embedding,
        "match_threshold": 0.75,  # Ngưỡng tương đồng (0.75 là mức tốt)
        "match_count": 1  # Chỉ lấy kỹ năng phù hợp nhất
    }).execute()
    print(query)
    if "error" in query:
        print(f"Lỗi từ Supabase: {query['error']}")
        return None
    if query and len(query[1]) > 0:  # Check if the list of skills is not empty
        matched_skill = query[1][0]  # Get the first matched skill
        print(f"Kỹ năng phù hợp nhất: {matched_skill['skill_text']}")
        return matched_skill['skill_text']
    else:
        print("Không tìm thấy kỹ năng phù hợp.")
        return None
    
# Ví dụ: Thêm kỹ năng mới vào DB
#add_skill_to_db("Đừng vội vàng khi khách hàng nói chuyện với bạn. Hãy lắng nghe và tìm hiểu nhu cầu của khách hàng.")
#add_skill_to_db("Khi khách có vẻ chưa quan tâm, cứ từ từ để nghĩ cách nối dài câu chuyện.")
#add_skill_to_db("Cần khiến cho khách tin tưởng.")



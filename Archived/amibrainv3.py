from langchain_openai import ChatOpenAI
from Archived.knowledge import  retrieve_relevant_infov2 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Initialize LLM for conversation

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = "us-east-1"  # Check Pinecone console for your region
index_name = "ami-knowledge"
llm = ChatOpenAI(model="gpt-4o", streaming=True)

prompt_old = PromptTemplate(
    input_variables=["history", "user_input", "products", "user_style", "sales_skills"],
    template="""
    Dựa vào các thông tin trước đây của người dùng, hãy đảm bảo câu trả lời phù hợp với phong cách của họ.
    Lịch sử cuộc trò chuyện:
    {history}

    Sản phẩm liên quan:
    {products}

    📌 Kỹ năng bán hàng cần áp dụng:
    {sales_skills}

    📢 **Hướng dẫn cho AMI:**
    1. Áp dụng các kỹ năng vào phản hồi để giúp cuộc trò chuyện tự nhiên hơn.
    2. Nếu có kỹ năng "Lắng nghe chủ động" → Hãy paraphrase lại ý của khách hàng trước khi tư vấn.
    3. Nếu có kỹ năng "Kể chuyện" → Hãy thêm một câu chuyện ngắn để minh họa sản phẩm.
    4. Nếu có kỹ năng "Giải quyết phản đối" → Hãy xử lý lo ngại của khách hàng trước khi tư vấn.

    Người dùng: {user_input}
    🎯 **AMI (giữ nguyên phong cách của người dùng + áp dụng kỹ năng bán hàng):**
    """
)
prompt = PromptTemplate(
    input_variables=["history", "user_input", "products", "user_style", "sales_skills"],
    template="""
    🎯 **Mục tiêu**: Hiểu ý định của người dùng và phản hồi một cách phù hợp.

    1️⃣ **Nếu người dùng đang hỏi về sản phẩm** → Dựa vào thông tin sản phẩm đã tìm thấy ({products}) để tư vấn ngắn gọn, đủ ý, có dẫn dắt hợp lý.  
    2️⃣ **Nếu người dùng đang hỏi về kỹ năng bán hàng** → Áp dụng kỹ năng phù hợp từ ({sales_skills}) vào câu trả lời.  
    3️⃣ **Nếu người dùng đang trò chuyện bình thường** → Duy trì hội thoại một cách tự nhiên, có thể thêm câu hỏi gợi mở.  
    4️⃣ **Luôn phản hồi theo phong cách của người dùng trước đây**: {user_style}  

    📜 **Lịch sử cuộc trò chuyện**:  
    {history}  

    🗣 **Tin nhắn từ người dùng**:  
    "{user_input}"  

    ✍️ **Phản hồi của AMI** (giữ phong cách hội thoại phù hợp):  
    """
)

#  chat_history = ChatMessageHistory()
chat_history = ChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=chat_history,
    memory_key="history",  # REQUIRED in newer versions
    return_messages=True
)

def retrieve_product(user_input):
    """Retrieve relevant context from Pinecone and return a structured summary."""
    if user_input is None:
        return "Không tìm thấy sản phẩm phù hợp."  # Return an appropriate message if input is None

    retrieved_info = retrieve_relevant_infov2(user_input, top_k=10)

    if not retrieved_info:
        return "Không tìm thấy sản phẩm phù hợp."

    structured_summary = []
    for doc in retrieved_info:
        content = doc.get("content", "").strip()
        if content:
            structured_summary.append(content)
    return "\n\n".join(structured_summary) if structured_summary else "Không tìm thấy sản phẩm phù hợp."

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

def handle_user_message(user_message, user_context):
    """
    Xử lý tin nhắn từ người dùng, xác định mục tiêu, tạo Best_map và dẫn dắt hội thoại.
    """
    # Bước 1: Phân tích khách hàng từ context
    customer_info = get_customer_info(user_context)
    print("customer_info:", customer_info)
    
    # Bước 2: Xác định mục tiêu hội thoại
    conversation_goal = determine_conversation_goal(user_message, customer_info)
    print("conversation_goal:", conversation_goal)

    # Bước 3: Tạo Best_map (bản đồ dẫn đến thành công)
    #best_map = create_best_map(conversation_goal)
    best_map = create_best_map(conversation_goal, user_context.get("customer_info", {}))


    # Bước 4: Dẫn dắt hội thoại theo Best_map
    response = guide_conversation(user_message, best_map, customer_info)
    #response = "test"
    return response

def get_customer_info(user_context):
    """
    Phân tích khách hàng từ context để hiểu rõ hơn về người dùng.
    """
    if user_context is None:
        return {"name": "Khách hàng", "age": None, "gender": None, "profession": None}  

    customer_info = {
        "name": user_context.get("name"),
        "age": user_context.get("age"),
        "gender": user_context.get("gender"),
        "occupation": user_context.get("occupation"),
        "interests": user_context.get("interests"),
        "purchase_history": user_context.get("purchase_history"),
    }

    # Nếu thiếu thông tin quan trọng, yêu cầu bổ sung
    missing_info = [key for key, value in customer_info.items() if value is None]

    if missing_info:
        return {"status": "missing_info", "missing_fields": missing_info}

    return {"status": "complete", "customer_info": customer_info}

import regex as re
def update_customer_info(user_message, customer_info):
    """
    Phân tích tin nhắn người dùng để cập nhật thông tin khách hàng.
    """
    updated = False

    age_match = re.search(r'(\d{1,2}) tuổi', user_message)
    if age_match:
        customer_info["age"] = int(age_match.group(1))
        customer_info["missing_fields"].remove("age")
        updated = True

    name_match = re.search(r'(tôi tên là|anh là|em là|tên tôi là)\s*([\w\s]+)', user_message, re.IGNORECASE)
    if name_match:
        customer_info["name"] = name_match.group(2).strip()
        customer_info["missing_fields"].remove("name")
        updated = True

    occupation_match = re.search(r'anh làm (\w+)', user_message, re.IGNORECASE)
    if occupation_match:
        customer_info["occupation"] = occupation_match.group(1)
        customer_info["missing_fields"].remove("occupation")
        updated = True

    if updated:
        customer_info["status"] = "info_updated"

    return customer_info


def determine_conversation_goal(customer_info, user_message):
    """
    Xác định mục tiêu hội thoại dựa trên thông tin khách hàng và tin nhắn.
    """
    if "missing_fields" in customer_info and len(customer_info["missing_fields"]) > 0:
        return "Khởi tạo hội thoại chung"
    
    if "muốn cao thêm" in user_message:
        return "Tư vấn sản phẩm tăng chiều cao"
    
    return "Tiếp tục hội thoại"


def determine_conversation_goal_v1(user_message, customer_info):
    """
    Xác định mục tiêu của cuộc trò chuyện dựa trên tin nhắn của user và thông tin khách hàng.
    """
    if not customer_info or customer_info.get("name") is None:
        return "Khởi tạo hội thoại chung"
    
    if customer_info.get("status") == "missing_info":
        missing_fields = customer_info.get("missing_fields", [])
        if missing_fields:
            return f"Hãy hỏi thêm về {', '.join(missing_fields)} trước khi tiếp tục."
        return "Khởi tạo hội thoại chung"
    
    # Dựa trên tin nhắn của user để xác định intent
    if "tăng chiều cao" in user_message:
        return "Tư vấn sản phẩm tăng chiều cao"
    if "giảm cân" in user_message:
        return "Tư vấn sản phẩm giảm cân"
    if "muốn tìm hiểu" in user_message:
        return "Cung cấp thông tin chi tiết về sản phẩm"
    if "giá bao nhiêu" in user_message:
        return "Cung cấp thông tin giá sản phẩm"
    if "tư vấn giúp" in user_message:
        return "Tư vấn cá nhân hóa dựa trên nhu cầu khách hàng"
    
    return "Dẫn dắt hội thoại để tìm hiểu nhu cầu khách hàng"

def create_best_map(conversation_goal, customer_info):
    print("customer_info in the conversation_goal:", customer_info)
    """
    Tạo bản đồ hội thoại tốt nhất dựa trên mục tiêu hội thoại và thông tin khách hàng.
    """
    if conversation_goal == "Khởi tạo hội thoại chung":
        missing = customer_info.get("missing_fields", [])
        if "name" in missing:
            return ["Bước 1: Hỏi tên khách hàng"]
        if "age" in missing:
            return ["Bước 2: Hỏi tuổi khách hàng"]
        if "occupation" in missing:
            return ["Bước 3: Hỏi nghề nghiệp khách hàng"]
        return ["Bước 4: Hoàn thành hồ sơ khách hàng"]

    if conversation_goal == "Tư vấn sản phẩm tăng chiều cao":
        return [
            "Bước 1: Xác nhận mong muốn tăng chiều cao",
            "Bước 2: Giới thiệu sản phẩm phù hợp",
            "Bước 3: Giải đáp thắc mắc"
        ]
    
    return ["Bước 1: Tiếp tục hội thoại"]


def create_best_map_v1(conversation_goal):
    """
    Dựa vào mục tiêu hội thoại, tìm các kỹ năng phù hợp trong Pinecone để xây dựng Best_map.
    """
    relevant_skills = search_sales_skills(conversation_goal)
    print("DEBUG: relevant_skills =", relevant_skills)
    
    best_map = []
    
    if conversation_goal == "Khởi tạo hội thoại chung":
        best_map.append("Bước 1: Chào hỏi và tìm hiểu nhu cầu khách hàng")
    
    if "khai thác thông tin cá nhân" in relevant_skills:
        best_map.append("Bước 1: Khai thác thông tin khách hàng (Tên, tuổi, nghề nghiệp)")
    if "khơi gợi động lực" in relevant_skills:
        best_map.append("Bước 2: Khơi gợi động lực (Lợi ích sản phẩm, tác động thực tế)")
    if "đưa ví dụ thuyết phục" in relevant_skills:
        best_map.append("Bước 3: Đưa ví dụ thực tế (Câu chuyện thành công của khách hàng khác)")
    if "đề xuất giải pháp" in relevant_skills:
        best_map.append("Bước 4: Đề xuất sản phẩm phù hợp với nhu cầu khách hàng")
    
    # Nếu không có bước nào, cung cấp fallback
    if not best_map:
        best_map.append("Bước 1: Mở đầu cuộc trò chuyện")
    
    print("DEBUG: best_map =", best_map)
    return best_map

def guide_conversation(user_message, best_map, customer_info):
    """
    Hướng dẫn hội thoại dựa trên tin nhắn người dùng, Best_map và thông tin khách hàng.
    """
    print(f"DEBUG: user_message = {user_message}")
    print(f"DEBUG: best_map = {best_map}")
    
    # Đảm bảo best_map có ít nhất 3 phần tử trước khi truy cập
    step_1 = best_map[0] if len(best_map) > 0 else "Bước 1: Chào hỏi và tìm hiểu nhu cầu khách hàng"
    step_2 = best_map[1] if len(best_map) > 1 else None
    step_3 = best_map[2] if len(best_map) > 2 else None

    # Bước 1: Chào hỏi nếu chưa có thông tin khách hàng
    if step_1 and "chào" in user_message.lower():
        return "Xin chào! Bạn có thể cho mình biết thêm về bạn không? (Tên, tuổi, nghề nghiệp...)"

    # Bước 2: Khai thác thông tin cá nhân
    if step_2 and "tên tôi" in user_message.lower():
        return f"Cảm ơn {user_message.split('tên tôi là')[-1].strip()}! Bạn có thể chia sẻ thêm sở thích hoặc nhu cầu của mình không?"

    # Bước 3: Xử lý nghi ngờ hoặc phản đối
    if step_3 and "nghi ngờ" in user_message.lower():
        return "Tôi hiểu bạn có một số thắc mắc. Đây là một số phản hồi từ khách hàng đã từng sử dụng sản phẩm của chúng tôi..."

    # Nếu không khớp bất kỳ bước nào, dẫn dắt lại
    return "Bạn có thể nói rõ hơn về nhu cầu hoặc câu hỏi của bạn không?"


def search_sales_skills(query_text, max_skills=3):
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


def search_sales_skills_basic(query_text, max_skills=3):
    """Truy vấn kỹ năng bán hàng từ Pinecone dựa trên độ tương đồng với input của người dùng."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
    query_embedding = embedding_model.embed_query(query_text)

    # Tìm kiếm trong Pinecone
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
            score = match.get("score", 0.0)  # Lấy điểm tương đồng

            # 🚀 Chỉ loại bỏ nếu score < 0.0 (còn lại vẫn lấy)
            if skill_text and score >= 0.0:  
                skills.append(skill_text)

    print("🔍 Kỹ năng tìm được:", skills)
    return skills if skills else ["Không tìm thấy kỹ năng phù hợp."]


def search_sales_skills_ok1(query_text, max_skills=3):
    """Truy vấn kỹ năng bán hàng từ Pinecone dựa trên độ tương đồng với input của người dùng."""
    
    # 🔹 Nếu là câu chào, tìm kỹ năng mở đầu
    greeting_keywords = ["chào", "hello", "hi", "xin chào"]
    if any(word in query_text.lower() for word in greeting_keywords):
        query_text = "Kỹ năng mở lời"
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
    query_embedding = embedding_model.embed_query(query_text)

    try:
        # 🔥 Gọi Pinecone để tìm kỹ năng gần nhất
        index = pc.Index(index_name)
        response = index.query(
            vector=query_embedding,
            top_k=max_skills,  
            include_metadata=True  
        )

        skills = []
        if response and "matches" in response:
            for match in response["matches"]:
                score = match.get("score", 0)
                skill_text = match.get("metadata", {}).get("content").strip()
                # 🔹 Lọc kết quả dựa trên ngưỡng (threshold)
                if skill_text and score >= 0.6:  # Giảm threshold để không bỏ sót kỹ năng
                    skills.append(skill_text)

        # 🔹 Lọc bỏ kỹ năng trùng lặp
        unique_skills = list(set(skills))

        print("🔍 Kỹ năng tìm được:", unique_skills)
        return unique_skills if unique_skills else ["Không tìm thấy kỹ năng phù hợp."]

    except Exception as e:
        print("⚠️ Lỗi truy vấn Pinecone:", str(e))
        return ["Không tìm thấy kỹ năng phù hợp."]


chain = (
    RunnablePassthrough.assign(
        history=lambda _: memory.load_memory_variables({}).get("history", []),
        products=lambda x: retrieve_product(x["user_input"]),
        sales_skills=lambda x: ", ".join(search_sales_skills(x["user_input"], max_skills=3)),  # Lấy kỹ năng từ Pinecone
        user_style=lambda _: "lịch sự"
    )  
    | prompt
    | llm
)

def ami_selling_basic(query):
    print("query:", query)
    input_data = {"user_input": query}
    # Lấy kỹ năng phù hợp
    relevant_skills = search_sales_skills(query)

    # Nếu có kỹ năng, sửa query để nhắc AMI
    if relevant_skills and relevant_skills[0] != "Không tìm thấy kỹ năng phù hợp.":
        input_data["user_input"] += f"\n\n📌 Hãy áp dụng kỹ năng này vào câu trả lời: {', '.join(relevant_skills)}"

    last_response =""
    response_stream = chain.stream(input_data)
    yield from (chunk.content if hasattr(chunk, 'content') else chunk for chunk in response_stream)  # Handle both cases
    #yield "Hey hello I'm Ami" # ✅ Cách khác
    memory.save_context({"input": query}, {"output": last_response.strip()})
def ami_selling(user_message, user_context=None):
    """
    Hàm chính xử lý hội thoại bán hàng của Ami.
    """
    if user_context is None:
        user_context = {}
    # Gọi handle_user_message để lấy phản hồi chính theo Best_map
    response = handle_user_message(user_message, user_context)

    # Bổ sung yếu tố bán hàng vào phản hồi
    sales_prompt = "Đây là sản phẩm em đề xuất cho anh/chị: ..."
    if "Bước 4" in response:
        response += f"\n{sales_prompt}"

    return response

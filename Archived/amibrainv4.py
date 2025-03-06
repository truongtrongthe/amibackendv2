from langchain_openai import ChatOpenAI
from Archived.knowledge import  retrieve_relevant_infov2 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import json
import logging
from openai import OpenAI
logging.basicConfig(level=logging.INFO)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = "us-east-1"  # Check Pinecone console for your region
index_name = "ami-knowledge"
llm = ChatOpenAI(model="gpt-4o", streaming=True)

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


def handle_user_message(user_message, user_context,company_goal,product_info):
    """
    Xử lý tin nhắn từ người dùng, xác định mục tiêu, tạo Best_map và dẫn dắt hội thoại.
    """
    # Bước 1: Lấy thông tin khách hàng từ user_context
    customer_info = user_context.get("customer_info", {})
    chat_history = user_context.get("chat_history", "")
    chat_history += f"\nUser: {user_message}"
    user_context["chat_history"] = chat_history  # Cập nhật lịch sử hội thoại

    # Bước 2: Xác định customer_stage từ lịch sử hội thoại
    customer_stage = get_customer_stage(chat_history)
    user_context["customer_stage"] = customer_stage
    print("customer_stage:", customer_stage)

    # Bước 3: Xác định mục tiêu hội thoại
    conversation_goal = determine_conversation_goal(customer_info, user_message, customer_stage)
    print("conversation_goal in handle_user_message:", conversation_goal)

    # Bước 3: Cập nhật customer_info với customer_stage
    customer_info["customer_stage"] = customer_stage
    # Bước 4: Tạo Best_map
    best_map = create_best_map(conversation_goal, customer_info,company_goal,product_info)
    print("best_map in handle_user_message:", best_map)

    response = generate_response(best_map, company_goal, customer_info)
    return response


def get_customer_stage(chat_history, company_goal="khách chuyển khoản"):
    """
    Dùng LLM để xác định giai đoạn của khách hàng dựa trên lịch sử hội thoại.
    """
    prompt = f"""
    Bạn là một AI tư vấn bán hàng. Dưới đây là lịch sử hội thoại giữa nhân viên và khách hàng:
    {chat_history}
    
    Công ty có mục tiêu cuối cùng là '{company_goal}'.
    Dựa vào lịch sử hội thoại, hãy xác định khách hàng đang ở giai đoạn nào trong hành trình này:
    - Awareness (Nhận thức)
    - Interest (Quan tâm)
    - Consideration (Cân nhắc)
    - Decision (Quyết định)
    - Action (Chuyển khoản)
    
    Chỉ trả về một trong các giai đoạn trên mà không có bất kỳ giải thích nào.
    """

    response= llm.invoke(prompt).content
    return response.strip()


def get_customer_emotion(chat_history):
   
    prompt = f"""
    Bạn là một chuyên gia tâm lý tinh tế. Hãy phát hiện cảm xúc hiện tại của khách hàng dựa trên lịch sử hội thoại:
    {chat_history}
    Chỉ trả về trạng thái cảm xúc mà không giải thích thêm
    """
    response= llm.invoke(prompt).content
    return response.strip()

def extract_customer_info(chat_history):
    print("chat_history in the extract_customer_info:", chat_history)
    """
    Dùng LLM để phân tích lịch sử hội thoại và trích xuất thông tin khách hàng.
    """
    prompt = f"""
    Dưới đây là lịch sử hội thoại giữa AI và khách hàng:
    {chat_history}

    Dựa vào nội dung này, hãy trích xuất các thông tin sau (nếu có):
    - Tên khách hàng (name)
    - Tuổi (age)
    - Giới tính (gender)
    - Nghề nghiệp (occupation)
    - Sở thích (interests)
    - Lịch sử mua hàng (purchase_history)

    Trả về một JSON với các trường tương ứng.
    Nếu không có thông tin, để trống.
    """

    response = llm.invoke(prompt).content  # Gọi LLM để phân tích

    try:
        # Làm sạch chuỗi JSON nếu có dấu ```json hoặc ``` thừa
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        clean_json = response[json_start:json_end]

        # Parse JSON
        extracted_info = json.loads(clean_json)
        print("Extracted customer info:", extracted_info)

        return extracted_info
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        return {}

def update_customer_info(current_info, new_info):
    """
    Cập nhật thông tin khách hàng với dữ liệu mới.
    """
    for key, value in new_info.items():
        if value:  # Chỉ cập nhật nếu có thông tin mới
            current_info[key] = value
    
    # Kiểm tra nếu vẫn còn missing fields
    missing_fields = [key for key, value in current_info.items() if not value]
    if missing_fields:
        current_info["status"] = "missing_info"
        current_info["missing_fields"] = missing_fields
    else:
        current_info["status"] = "completed"

    return current_info

def chat_pipeline(user_message, chat_history, customer_info, llm):
    """
    Xử lý hội thoại theo pipeline:
    1. Trích xuất thông tin khách hàng
    2. Cập nhật customer_info
    3. Tiếp tục hội thoại dựa trên tình trạng customer_info
    """
    # 1. Trích xuất thông tin từ lịch sử chat
    new_info = extract_customer_info(chat_history, llm)
    print("new_info:", new_info)

    # 2. Cập nhật thông tin khách hàng
    customer_info = update_customer_info(customer_info, new_info)
    print("customer_info:", customer_info)
    # 3. Kiểm tra xem đã đủ thông tin chưa
    if customer_info["status"] == "missing_info":
        missing = customer_info["missing_fields"]
        next_question = ask_for_missing_info(missing)
        return next_question, customer_info
    
    # 4. Nếu đã đủ thông tin, tiếp tục hội thoại
    response = continue_conversation(user_message, customer_info, llm)
    return response, customer_info
def ask_for_missing_info(missing_fields):
    """
    Sinh câu hỏi để tiếp tục hoàn thiện thông tin khách hàng.
    """
    questions = {
        "name": "Bạn có thể cho tôi biết tên của bạn không?",
        "age": "Bạn bao nhiêu tuổi?",
        "gender": "Bạn là nam hay nữ?",
        "occupation": "Bạn đang làm nghề gì?",
        "interests": "Bạn quan tâm đến lĩnh vực nào?",
        "purchase_history": "Bạn đã từng mua sản phẩm nào tương tự chưa?"
    }
    for field in missing_fields:
        if field in questions:
            return questions[field]  # Hỏi lần lượt từng câu
    return "Hãy cho tôi biết thêm về bạn!"  # Nếu không có câu hỏi cụ thể
def continue_conversation(user_message, customer_info, llm):
    """
    Tiếp tục hội thoại dựa trên thông tin khách hàng và mục tiêu hội thoại.
    """
    conversation_goal = determine_conversation_goal(user_message, customer_info)
    
    best_map = create_best_map(conversation_goal, customer_info)
    
    response = llm.generate(f"Dựa vào mục tiêu '{conversation_goal}', hãy phản hồi: {user_message}")
    
    return response


def determine_conversation_goal_hardcoded(customer_info, user_message, customer_stage):
    """
    Xác định mục tiêu hội thoại dựa trên điểm hiện tại (customer_stage) và mục tiêu tiếp theo.
    """
    print("user_message in the determine_conversation_goal:", user_message)
    print("customer_info in the determine_conversation_goal:", customer_info)
    print("customer_stage in the determine_conversation_goal:", customer_stage)

    if "missing_fields" in customer_info and len(customer_info["missing_fields"]) > 0:
        return "Khởi tạo hội thoại chung"

    # Xác định "điểm B" dựa trên hành trình khách hàng
    stage_transitions = {
        "Awareness (Nhận thức)": "Interest (Quan tâm)",
        "Interest (Quan tâm)": "Consideration (Cân nhắc)",
        "Consideration (Cân nhắc)": "Decision (Quyết định)",
        "Decision (Quyết định)": "Action (Chuyển khoản)",
        "Action (Chuyển khoản)": "Hoàn thành đơn hàng"
    }

    next_stage = stage_transitions.get(customer_stage, "Tiếp tục hội thoại")

    print(f"Next stage: {next_stage}")

    return next_stage

def infer_conversation_goal(customer_stage, user_message):
    """
    Sử dụng LLM để suy luận conversation_goal phù hợp với customer_stage và nội dung tin nhắn.
    """
    prompt = f"""
    Dựa trên giai đoạn khách hàng trong hành trình mua hàng: "{customer_stage}", 
    và tin nhắn: "{user_message}", hãy xác định bước hợp lý tiếp theo để dẫn khách hàng đến mục tiêu "Chuyển khoản".
    
    Trả về chỉ một mục tiêu hội thoại cụ thể (không giải thích), ví dụ: "Giới thiệu sản phẩm", "Thuyết phục khách hàng", "Hướng dẫn thanh toán".
    """

    response = llm.invoke(prompt).content  # Gọi LLM để phân tích
    return response.strip()


def determine_conversation_goal(customer_info, user_message, customer_stage):
    """
    Xác định mục tiêu hội thoại dựa trên thông tin khách hàng, nội dung tin nhắn và giai đoạn khách hàng.
    """
    print("user_message in the determine_conversation_goal:", user_message)
    print("customer_info in the determine_conversation_goal:", customer_info)
    print("customer_stage in the determine_conversation_goal:", customer_stage)

    # Nếu thông tin khách còn thiếu, cần tiếp tục hỏi để hoàn chỉnh
    if "missing_fields" in customer_info and len(customer_info["missing_fields"]) > 0:
        return "Khởi tạo hội thoại chung"

    # Xác định mục tiêu tiếp theo bằng cách suy luận từ company_goal
    conversation_goal = infer_conversation_goal(customer_stage, user_message)

    return conversation_goal


def create_best_map(conversation_goal, customer_info, company_goal, product_info):
    """
    Sử dụng LLM để suy luận Best_map phù hợp dựa trên conversation_goal, customer_info và company_goal.
    """
    prompt = f"""
    🛒 Khách hàng đang ở giai đoạn: "{customer_info.get('customer_stage', 'Unknown')}"
    🎯 Mục tiêu hội thoại: "{conversation_goal}"
    🏆 Mục tiêu cuối cùng của công ty: "{company_goal}"
    👤 Thông tin khách hàng: {customer_info}
    📦 Thông tin sản phẩm công ty: {product_info}

    ✅ Hãy tạo một hướng dẫn phản hồi tốt nhất (Best_map) giúp nhân viên bán hàng nói chuyện hợp lý và hướng khách hàng đến {company_goal}.
    ✅ Điều chỉnh phản hồi dựa trên cảm xúc và giai đoạn của khách hàng:
    - Nếu chưa biết tên khách hàng, hãy hỏi tên khách hàng trước.
    - Nếu khách hàng còn phân vân, hãy nhấn mạnh lợi ích của sản phẩm.
    - Nếu khách hàng có hứng thú, hãy gợi mở một lý do mạnh mẽ để hành động ngay.
    - Nếu khách hàng có lo ngại, hãy trấn an và cung cấp thông tin hỗ trợ.

    🎤 Nếu biết tên khách hàng, hãy xưng hô thân thiện.
    📢 Trả về một đoạn văn ngắn, không quá 3 câu, với phong cách giao tiếp thường thức (casual).
    """
    response = llm.invoke(prompt).content  # Gọi OpenAI hoặc mô hình AI khác
    return response.strip()


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
def ami_selling(user_message, user_context=None):
    """
    Hàm chính xử lý hội thoại bán hàng của Ami.
    """
    if user_context is None:
        user_context = {}

    print("user_message in the ami_selling:", user_message)

    # Trích xuất thông tin khách hàng
    extracted_info = extract_customer_info(user_message)
    print("extracted_info in the ami_selling:", extracted_info)

    # Cập nhật user_context với customer_info mới
    if extracted_info:
        user_context["customer_info"] = extracted_info
    else:
        user_context["customer_info"] = {"status": "missing_info"}  # Giữ trạng thái nếu chưa có dữ liệu

    print("Updated user_context:", user_context)

    company_goal = "Khách chuyển khoản"
    product_info = retrieve_product(user_message)

    # Gọi handle_user_message để lấy phản hồi chính theo Best_map
    response = handle_user_message(user_message, user_context,company_goal,product_info)

    return response

def generate_response(best_map, company_goal, customer_info):
    """
    Sinh phản hồi dựa trên Best_map + hướng khách hàng đến company_goal.
    """
    prompt = f"""
    Khách hàng: {customer_info}
    Best_map: "{best_map}"
    Company_goal: "{company_goal}"

    Hãy tạo một phản hồi tự nhiên, thân thiện, dẫn dắt khách hàng theo Best_map và hướng họ đến {company_goal}.
    """

    response = llm.invoke(prompt).content  # Gọi OpenAI hoặc mô hình AI khác để sinh phản hồi
    return response.strip()
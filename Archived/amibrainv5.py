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
from typing import Dict, Any
from openai import OpenAI
logging.basicConfig(level=logging.INFO)

# Declare user_context as a global variable
user_context = {"chat_history": [], "customer_info": {}}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = "us-east-1"  # Check Pinecone console for your region
index_name = "ami-knowledge"
llm = ChatOpenAI(model="gpt-4o", streaming=True)
client = OpenAI()
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

    retrieved_info = retrieve_relevant_infov2(user_input, top_k=3)

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

def detect_customer_intent_dynamic(message: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": """You are an AI that detects customer intent and categorizes it into three main groups:
                1. general_conversation (e.g., greetings, small talk, general inquiries)
                2. sales_related (e.g., asking about price, product details, promotions)
                3. after_sales (e.g., warranty, support, returns, complaints)
                Return a JSON object with "intent", "intent_group", and optional "sub_intent" fields.
            """},
            {"role": "user", "content": f"Analyze intent from this message: {message}"}
        ]
    )
    intent_data = response.choices[0].message.content.strip()
    
    # Attempt to parse the intent_data as JSON
    try:
        return json.loads(intent_data)  # Use json.loads instead of eval
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        return {"intent": "unknown", "intent_group": "general_conversation"}  # Return a default value in case of error
   
def ami_drive(user_message, user_context,company_goal,product_info):
    """
    Xử lý tin nhắn từ người dùng, xác định mục tiêu, tạo Best_map và dẫn dắt hội thoại.
    """
    intent_data = detect_customer_intent_dynamic(user_message)
    intent = intent_data.get("intent", "unknown")
    intent_group = intent_data.get("intent_group", "general_conversation")  # Mặc định là giao tiếp thông thường
    sub_intent = intent_data.get("sub_intent", None)
    print("intent_group in the ami_drive:", intent_group)
    if intent_group == "general_conversation":
        return handle_general_conversation(intent, sub_intent,user_message, user_context)

    elif intent_group == "sales_related":
        return handle_sales(user_message, user_context,company_goal,product_info)

    elif intent_group == "after_sales":
        return "Post-sales support"

    else:
        return "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể nói rõ hơn không?"

def handle_general_conversation(intent, sub_intent, user_message, user_context):
    # Lấy thông tin khách hàng từ user_context
    customer_info = user_context.get("customer_info", {})
    chat_history = user_context.get("chat_history", [])

    # Append the new user message to the chat history
    chat_history.append(f"User: {user_message}")
    user_context["chat_history"] = chat_history  

    # **Danh sách thông tin cần thu thập**
    required_fields = ["name", "age", "occupation", "interests"]
    missing_fields = [field for field in required_fields if not customer_info.get(field)]

    # **Chỉ hỏi tên nếu thật sự chưa có**
    if "name" in missing_fields:
        probing_prompt = f"""
        Lịch sử hội thoại: {', '.join(chat_history)}  # Join the list into a string for display
        Thông tin khách hàng hiện có: {customer_info}
        Bạn là một trợ lý AI. Khách hàng chưa cung cấp tên. Hãy đặt một câu hỏi lịch sự để hỏi tên.
        """
        return llm.invoke(probing_prompt).content

    # **Nếu đã có tên nhưng còn thiếu thông tin khác → Hỏi tiếp thông tin còn thiếu**
    if missing_fields:
        probing_prompt = f"""
        Lịch sử hội thoại: {', '.join(chat_history)}  # Join the list into a string for display
        Thông tin khách hàng hiện có: {customer_info}
        Thông tin còn thiếu: {missing_fields}
        Hãy đặt một câu hỏi tự nhiên để khai thác một trong các thông tin còn thiếu mà không làm khách hàng khó chịu.
        """
        return llm.invoke(probing_prompt).content

    # **Nếu đã có đủ thông tin → Trả lời theo ngữ cảnh**
    response_prompt = f"""
    Tóm tắt hội thoại: {', '.join(chat_history)}  # Join the list into a string for display
    Thông tin khách hàng: {customer_info}
    Câu khách hàng vừa hỏi: {user_message}
    Hãy phản hồi một cách tự nhiên, phù hợp với thông tin khách hàng, giữ cuộc trò chuyện mượt mà.
    """
    extract_prompt = f"""
    Hội thoại: {', '.join(chat_history)}  # Join the list into a string for display
    Thông tin hiện có: {user_context.get("customer_info", {})}
    Hãy cập nhật thông tin khách hàng dựa trên hội thoại mới. 
    Chú ý: Nếu đã có thông tin, không được làm mất thông tin cũ. Chỉ bổ sung phần còn thiếu.
    """

    return llm.invoke(response_prompt).content
    
    
def handle_sales(user_message, user_context,company_goal,product_info):
    """
    Xử lý tin nhắn từ người dùng, xác định mục tiêu, tạo Best_map và dẫn dắt hội thoại.
    """
    # Bước 1: Lấy thông tin khách hàng từ user_context
    customer_info = user_context.get("customer_info", {})
    chat_history = user_context.get("chat_history", [])
    chat_history.append(f"User: {user_message}")
    user_context["chat_history"] = chat_history  # Cập nhật lịch sử hội thoại
    
    # Bước 2: Xác định customer_stage từ lịch sử hội thoại
    customer_stage = get_customer_stage(chat_history)
    user_context["customer_stage"] = customer_stage
    print("customer_stage:", customer_stage)

    # Bước 3: Xác định mục tiêu hội thoại
    cg = get_conversation_goal(customer_info, user_message, customer_stage)
    print("conversation_goal in handle_user_message:", cg)

    # Bước 3: Cập nhật customer_info với customer_stage
    customer_info["customer_stage"] = customer_stage
    # Bước 4: Tạo Best_map
    best_map = create_best_map(cg, customer_info,company_goal,product_info)
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


def customer_emotion(chat_history):
   
    prompt = f"""
    Bạn là một chuyên gia tâm lý tinh tế. Hãy phát hiện cảm xúc hiện tại của khách hàng dựa trên lịch sử hội thoại:
    {chat_history}
    Chỉ trả về trạng thái cảm xúc mà không giải thích thêm
    """
    response= llm.invoke(prompt).content
    return response.strip()

def get_customer_info(chat_history):
    """
    Dùng LLM để phân tích lịch sử hội thoại và trích xuất thông tin khách hàng.
    """
    # Join the chat history into a single string for processing
    chat_history_str = "\n".join(user_context["chat_history"])
    print("Formatted chat_history_str:", repr(chat_history_str))

    
 

    prompt = f"""
Dưới đây là lịch sử hội thoại giữa AI và khách hàng:
{chat_history_str}

Hãy cố gắng suy luận thông tin từ cuộc trò chuyện này.
Nếu khách hàng đề cập đến một tên riêng, giả định đó là "name".
Nếu khách hàng nói về công việc của họ, đó là "occupation".
Nếu chưa có đủ thông tin, hãy dự đoán dựa trên ngữ cảnh hoặc để trống.

Trả về một JSON với các trường:
- name (nếu có thể suy luận)
- age (nếu có thể suy luận)
- gender (nếu có thể suy luận)
- occupation (nếu có thể suy luận)
- interests (nếu có thể suy luận)
- purchase_history (nếu có thể suy luận)
"""


    response = llm.invoke(prompt).content  # Gọi LLM để phân tích
    print("response in the get_customer_info:", response)

    try:
        # Clean the JSON string if it has extra formatting
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
    Cập nhật thông tin khách hàng với dữ liệu mới mà không làm mất thông tin cũ.
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
   
    # 1. Trích xuất thông tin từ lịch sử chat
    new_info = customer_info(chat_history, llm)
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
    cg = get_conversation_goal(user_message, customer_info)
    
    best_map = create_best_map(cg, customer_info)
    
    response = llm.generate(f"Dựa vào mục tiêu '{cg}', hãy phản hồi: {user_message}")
    
    return response

def get_conversation_goal(customer_info, user_message, customer_stage):
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
    prompt = f"""
    Dựa trên giai đoạn khách hàng trong hành trình mua hàng: "{customer_stage}", 
    và tin nhắn: "{user_message}", hãy xác định bước hợp lý tiếp theo để dẫn khách hàng đến mục tiêu "Chuyển khoản".
    
    Trả về chỉ một mục tiêu hội thoại cụ thể (không giải thích), ví dụ: "Giới thiệu sản phẩm", "Thuyết phục khách hàng", "Hướng dẫn thanh toán".
    """

    response = llm.invoke(prompt).content  # Gọi LLM để phân tích


    return response.strip()

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



def ami_selling(user_message):
    """
    Hàm chính xử lý hội thoại bán hàng của Ami.
    """
    global user_context  # Declare user_context as global

    print("user_message in the ami_selling:", user_message)

    # Append the new user message to the chat history
    print("Before appending message:", user_context["chat_history"])
    user_context["chat_history"].append(f"User: {user_message}")
    print("After appending message:", user_context["chat_history"])


    # Trích xuất thông tin khách hàng
    extracted_info = get_customer_info(user_message)
    print("extracted_info in the ami_selling:", extracted_info)

    # Cập nhật user_context với customer_info mới
    user_context["customer_info"] = update_customer_info(user_context.get("customer_info", {}), extracted_info)

    print("Updated user_context:", user_context)

    company_goal = "Khách chuyển khoản"
    product_info = retrieve_product(user_message)

    # Gọi handle_user_message để lấy phản hồi chính theo Best_map
    response = ami_drive(user_message, user_context, company_goal, product_info)

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
from langchain_openai import ChatOpenAI
from knowledge import  retrieve_relevant_infov2 
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
import json


# Initialize conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Declare user_context as a global variable
user_context = {"customer_info": {}}

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

# Initialize chat history
chat_history = ChatMessageHistory()

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
    # Load chat history from memory
    chat_history_list = memory.load_memory_variables({})["history"]
    

    # Append the new user message to the chat history
    chat_history_list.append(f"User: {user_message}")
    print(chat_history_list)
    print([type(msg) for msg in chat_history_list])

    # Join the chat history into a single string
    # chat_history_str = "\n".join(chat_history_list)
    # chat_history_str = "\n".join([message.content for message in chat_history_list if hasattr(message, 'content')])
    #chat_history_str = "\n".join(msg.content for msg in chat_history_list)
    chat_history_str = "\n".join(
    msg.content if hasattr(msg, "content") else msg for msg in chat_history_list
    )


    
    # Save the joined chat history to memory
    memory.save_context({"input": "chat_history"}, {"output": chat_history_str})

    # Lấy thông tin khách hàng từ user_context
    customer_info = user_context.get("customer_info", {})

    # **Danh sách thông tin cần thu thập**
    required_fields = ["name", "age", "occupation", "interests"]
    missing_fields = [field for field in required_fields if not customer_info.get(field)]

    # **Chỉ hỏi tên nếu thật sự chưa có**
    if "name" in missing_fields:
        probing_prompt = f"""
        Lịch sử hội thoại: {chat_history_str}
        Thông tin khách hàng hiện có: {customer_info}
        Bạn là một trợ lý AI. Khách hàng chưa cung cấp tên. Hãy đặt một câu hỏi lịch sự để hỏi tên.
        """
        return llm.invoke(probing_prompt).content

    # **Nếu đã có tên nhưng còn thiếu thông tin khác → Hỏi tiếp thông tin còn thiếu**
    if missing_fields:
        probing_prompt = f"""
        Lịch sử hội thoại: {chat_history_str}
        Thông tin khách hàng hiện có: {customer_info}
        Thông tin còn thiếu: {missing_fields}
        Hãy đặt một câu hỏi tự nhiên để khai thác một trong các thông tin còn thiếu mà không làm khách hàng khó chịu.
        """
        return llm.invoke(probing_prompt).content

    # **Nếu đã có đủ thông tin → Trả lời theo ngữ cảnh**
    response_prompt = f"""
    Tóm tắt hội thoại: {chat_history_str}
    Thông tin khách hàng: {customer_info}
    Câu khách hàng vừa hỏi: {user_message}
    Hãy phản hồi một cách tự nhiên, phù hợp với thông tin khách hàng, giữ cuộc trò chuyện mượt mà.
    """
    extract_prompt = f"""
    Hội thoại: {chat_history_str}
    Thông tin hiện có: {user_context.get("customer_info", {})}
    Hãy cập nhật thông tin khách hàng dựa trên hội thoại mới. 
    Chú ý: Nếu đã có thông tin, không được làm mất thông tin cũ. Chỉ bổ sung phần còn thiếu.
    """

    return llm.invoke(response_prompt).content
import json

def handle_sales(user_message, user_context, company_goal, product_info):
    """
    Xử lý tin nhắn từ người dùng, kết hợp Best Approach vào phản hồi.
    """

    # Step 1: Retrieve customer info
    customer_info = user_context.get("customer_info", {})
    chat_history_list = memory.load_memory_variables({})["history"]
    chat_history_list.append(f"User: {user_message}")

    # Convert chat history to a string and save
    chat_history_str = "\n".join(
        msg.content if hasattr(msg, "content") else msg for msg in chat_history_list
    )
    memory.save_context({"input": "chat_history"}, {"output": chat_history_str})

    # Step 2: Determine customer stage
    customer_stage = get_customer_stage(chat_history_list)
    user_context["customer_stage"] = customer_stage
    print("customer_stage:", customer_stage)

    next_stop = get_customer_next_stop(customer_stage)
    print("next_stop:", next_stop)

    # Step 3: Identify conversation goal
    convo_goal = get_conversation_goal(customer_info, user_message, customer_stage, next_stop)

    # Step 4: Get Best Approach & Instruction
    approach_data = analyse_approach(customer_stage, convo_goal, customer_info, product_info)

    # 🔍 Ensure response is valid
    if not approach_data or not isinstance(approach_data, dict):
        print("⚠️ Warning: analyse_approach returned an invalid response!")
        approach_data = {
            "best_approach": "Hãy tạo sự tin tưởng và khuyến khích khách hàng.",
            "instruction": "Hãy phản hồi lịch sự, tạo sự tin tưởng và cung cấp thêm thông tin hữu ích."
        }

    best_approach = approach_data.get("best_approach", "Hãy tạo sự tin tưởng và khuyến khích khách hàng.")
    instruction = approach_data.get("instruction", "Hãy phản hồi lịch sự, tạo sự tin tưởng và cung cấp thêm thông tin hữu ích.")

    # 🔹 Markdown Analysis
    analysis_markdown = f"""
    **📊 Phân tích chiến lược:**  
    - **📍 Giai đoạn khách hàng:** {customer_stage}  
    - **🎯 Điểm đến tiếp theo:** {next_stop}  
    - **💡 Chiến thuật tiếp cận:** {best_approach}  
    - **💡 Hướng dẫn phản hồi:** {instruction}  

    Có thể sử dụng câu dưới đây:
    ---
    """

    # Step 5: Generate Final Response
    final_response = generate_conversation_response(user_message, customer_info, best_approach, instruction)

    return analysis_markdown + final_response


def generate_conversation_response(user_message, customer_info, best_approach, instruction):
    """
    Tạo phản hồi hội thoại dựa trên Best Approach, Instruction và thông tin khách hàng.
    - Best Approach: hướng tiếp cận phù hợp.
    - Instruction: chỉ dẫn chi tiết về cách phản hồi.
    """

    print("Best Approach in generate_conversation_response:", best_approach)
    print("Instruction in generate_conversation_response:", instruction)

    prompt = f"""
    🗣️ Tin nhắn khách hàng: "{user_message}"
    👤 Thông tin khách hàng: {json.dumps(customer_info, ensure_ascii=False)}
    💡 Best Approach: "{best_approach}"
    🎯 Instruction: "{instruction}"

    🔹 Hãy tạo một phản hồi **tự nhiên, thân thiện, gần gũi**, phản ánh phong cách nói chuyện của khách hàng.
    🔹 **Tích hợp Best Approach một cách tinh tế**, không lặp lại nguyên văn.
    🔹 **Tuân theo hướng dẫn trong Instruction** để đảm bảo phản hồi có chiến thuật phù hợp.
    🔹 Đừng tạo phản hồi quá dài – tối đa 3 câu.
    

    📝 Trả lời:
    """

    response = llm.invoke(prompt)

    if not response or not response.content.strip():
        print("⚠️ LLM response is empty or None")
        return "Đây là một sản phẩm rất tốt, bạn có thể tham khảo thêm nhé!"

    return response.content.strip()


def generate_response(best_map, next_stop, customer_info):
    """
    Sinh phản hồi dựa trên Best_map + hướng khách hàng đến đích đến.
    """
    print("best_map in generate_response:", best_map)
    prompt = f"""
    Khách hàng: {customer_info}
    Best_map: "{best_map}"
    Company_goal: "{next_stop}"

    Hãy tạo một phản hồi tự nhiên, thân thiện, dẫn dắt khách hàng theo Best_map và hướng họ đến {next_stop}. Hãy trả lời dùng ngôn ngữ của dùng.
    """

    response = llm.invoke(prompt).content  # Gọi OpenAI hoặc mô hình AI khác để sinh phản hồi
    return response.strip()

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

def get_customer_next_stop(current_stop):
    if "Awareness" in current_stop:
        return "Interest"
    elif "Interest" in current_stop:
        return "Consideration"
    elif "Consideration" in current_stop:
        return "Decision"
    elif "Decision" in current_stop:
        return "Action"
    else:
        return "Unknown Stage"  # Default case to handle unexpected stages
    
def customer_emotion(chat_history):
   
    prompt = f"""
    Bạn là một chuyên gia tâm lý tinh tế. Hãy phát hiện cảm xúc hiện tại của khách hàng dựa trên lịch sử hội thoại:
    {chat_history}
    Chỉ trả về trạng thái cảm xúc mà không giải thích thêm
    """
    response= llm.invoke(prompt).content
    return response.strip()

def get_customer_info():
    """
    Dùng LLM để phân tích lịch sử hội thoại và trích xuất thông tin khách hàng.
    """
    # Load the entire chat history from memory
    chat_history = memory.load_memory_variables({})["history"]
    
    # Extract text from each message in the chat history
    chat_history_str = "\n".join([message.content for message in chat_history if hasattr(message, 'content')])
    print("Formatted chat_history_str:", repr(chat_history_str))

    prompt = f"""
    
    Dưới đây là lịch sử hội thoại giữa AI và khách hàng:
   {chat_history_str}

   Hãy trích xuất các thông tin sau từ cuộc trò chuyện:
   - Tên khách hàng (name)
   - Tuổi (age)
   - Giới tính (gender)
   - Nghề nghiệp (occupation)
   - Sở thích (interests)
   - Lịch sử mua hàng (purchase_history)
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

        # Ensure the JSON is properly formatted
        clean_json = clean_json.replace("'", '"')  # Replace single quotes with double quotes

        # Parse JSON
        extracted_info = json.loads(clean_json)
        print("Extracted customer info:", extracted_info)

        # Convert extracted_info to a JSON string
        if not extracted_info or all(value == "" for value in extracted_info.values()):
            extracted_info_str = "No customer information extracted."
        else:
            extracted_info_str = json.dumps(extracted_info)

        # Save the context with the string representation
        memory.save_context({"input": "customer_info"}, {"output": extracted_info_str})

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

def get_conversation_goal(customer_info, user_message, customer_stage,next_stop):
    """
    Xác định mục tiêu hội thoại dựa trên thông tin khách hàng, nội dung tin nhắn và giai đoạn khách hàng.
    """
    print("user_message in the determine_conversation_goal:", user_message)
    print("customer_info in the determine_conversation_goal:", customer_info)
    print("customer_stage in the determine_conversation_goal:", customer_stage)

    # Nếu thông tin khách còn thiếu, cần tiếp tục hỏi để hoàn chỉnh
    # Xác định mục tiêu tiếp theo bằng cách suy luận từ company_goal
    prompt = f"""
    Dựa trên giai đoạn khách hàng trong hành trình mua hàng: "{customer_stage}", 
    và tin nhắn: "{user_message}", hãy xác định bước hợp lý tiếp theo để dẫn khách hàng đến mục tiêu tiến tới được {next_stop} .
    
    Trả về chỉ một mục tiêu hội thoại cụ thể (không giải thích), ví dụ: "Giới thiệu sản phẩm", "Thuyết phục khách hàng", "Hướng dẫn thanh toán".
    """

    response = llm.invoke(prompt).content  # Gọi LLM để phân tích


    return response.strip()


def propose_best_approach(conversation_goal, customer_info, product_info):
    """
    Sử dụng LLM để suy luận Best Approach phù hợp dựa trên conversation_goal, customer_info và product_info.
    """

    print("customer_info in propose_best_approach:", customer_info)

    prompt = f"""
    🏆 Mục tiêu hội thoại: "{conversation_goal}"
    👤 Thông tin khách hàng: {json.dumps(customer_info, ensure_ascii=False)}
    📦 Thông tin sản phẩm: {json.dumps(product_info, ensure_ascii=False)}

    ✅ Hãy tạo một hướng dẫn (Best Approach) giúp nhân viên bán hàng nói chuyện hợp lý với khách.
    ✅ Best Approach không phải là câu trả lời trực tiếp, mà là cách tiếp cận tổng quan giúp cuộc trò chuyện hiệu quả hơn.

    🔹 Trả lời CHỈ DƯỚI ĐỊNH DẠNG JSON như sau:
    ```json
    {{ "best_approach": "<Hướng dẫn ngắn gọn, súc tích, tối đa 2 câu>" }}
    ```
    🚫 Không thêm bất kỳ văn bản nào bên ngoài JSON.
    """

    response = llm.invoke(prompt).content  # Gọi LLM

    try:
        # 💡 Fix: Clean and parse JSON response
        json_str = response.strip().strip("```json").strip("```").strip()
        best_approach_data = json.loads(json_str)  # Parse cleaned JSON

        if "best_approach" not in best_approach_data:
            raise ValueError("Missing 'best_approach' in response")

        return best_approach_data["best_approach"]

    except Exception as e:
        print(f"⚠️ Error parsing best_approach: {e}, raw response: {response}")
        return "Hãy tạo sự tin tưởng và khuyến khích khách hàng."  # Fallback approach
import json
import re
import json
import re

def analyse_approach(customer_stage,conversation_goal, customer_info, product_info):
    """
    Sử dụng LLM để suy luận chiến thuật tiếp cận khách hàng và tạo hướng dẫn cho response prompt.
    
    📌 Output gồm:
    - best_approach: Cách tiếp cận ngắn gọn để đạt conversation_goal.
    - instruction: Hướng dẫn cụ thể để truyền vào response prompt.
    """

    print("customer_info in analyse_approach:", customer_info)

    prompt = f"""
    🏆 Mục tiêu hội thoại: "{conversation_goal}"
    📌 Giai đoạn khách hàng: "{customer_stage}"
    👤 Thông tin khách hàng: {json.dumps(customer_info, ensure_ascii=False)}
    📦 Thông tin sản phẩm: {json.dumps(product_info, ensure_ascii=False)}

    ✅ Hãy phân tích hiện trạng khách hàng và đề xuất cách tiếp cận hiệu quả để đạt mục tiêu hội thoại.
    ✅ Sau đó, tạo hướng dẫn (instruction) giúp AI sinh ra phản hồi hợp lý trong cuộc trò chuyện.

    🔹 Trả lời CHỈ DƯỚI ĐỊNH DẠNG JSON như sau:
    ```json
    {{
        "best_approach": "<Hướng dẫn tiếp cận ngắn gọn, tối đa 2 câu>",
        "instruction": "<Hướng dẫn chi tiết để truyền vào response prompt>"
    }}
    ```
    🚫 Không thêm bất kỳ văn bản nào bên ngoài JSON.
    """

    response = llm.invoke(prompt)

    if not response or not response.content:
        print("⚠️ LLM response is empty or None")
        return {
            "best_approach": "Hãy tạo sự tin tưởng và khuyến khích khách hàng.",
            "instruction": "Hãy phản hồi lịch sự, tạo sự tin tưởng và cung cấp thêm thông tin hữu ích."
        }

    raw_response = response.content.strip()
    
    # 💡 Sử dụng regex để lấy JSON chính xác (phòng khi LLM trả về text lẫn JSON)
    match = re.search(r'\{.*\}', raw_response, re.DOTALL)

    if not match:
        print(f"⚠️ No valid JSON found in response: {raw_response}")
        return {
            "best_approach": "Hãy tạo sự tin tưởng và khuyến khích khách hàng.",
            "instruction": "Hãy phản hồi lịch sự, tạo sự tin tưởng và cung cấp thêm thông tin hữu ích."
        }

    json_str = match.group(0)

    try:
        result = json.loads(json_str)  # Parse JSON

        if "best_approach" not in result or "instruction" not in result:
            raise ValueError("Missing keys in JSON response")

        return result

    except Exception as e:
        print(f"⚠️ JSON parsing error: {e}, raw response: {json_str}")
        return {
            "best_approach": "Hãy tạo sự tin tưởng và khuyến khích khách hàng.",
            "instruction": "Hãy phản hồi lịch sự, tạo sự tin tưởng và cung cấp thêm thông tin hữu ích."
        }

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
    Hàm chính xử lý hội thoại bán hàng của AMI.
    """
    global memory  # Ensure we are using the global memory instance

    # Save the user message to memory
    memory.save_context({"input": user_message}, {"output": ""})

    # Load the entire chat history
    chat_history_list = memory.load_memory_variables({})["history"]

    
    # Extract customer information from the chat history
    extracted_info = get_customer_info()
    
    # Update memory with extracted customer information (if needed)
    if extracted_info:
        extracted_info_str = json.dumps(extracted_info)
        memory.save_context({"input": "customer_info"}, {"output": extracted_info_str})
        user_context["customer_info"] = extracted_info
    else:
        memory.save_context({"input": "customer_info"}, {"output": "No customer information extracted."})

    print("Updated user_context:", user_context)

    company_goal = "Khách chuyển khoản"
    product_info = retrieve_product(user_message)

    # Gọi handle_user_message để lấy phản hồi chính theo Best_map
    response = ami_drive(user_message, user_context, company_goal, product_info)

    return response


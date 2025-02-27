
from amibrainv4 import extract_customer_info, update_customer_info, handle_user_message, chat_pipeline
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", streaming=True)

chat_history = """
User: Tôi tên là Nam, 30 tuổi, làm kỹ sư IT. Tôi muốn cao thêm 5cm.
AI: Bạn muốn tăng chiều cao bao nhiêu cm? Bạn đã thử phương pháp nào chưa?
User: Tôi muốn cao thêm 5cm. Tôi chưa thử phương pháp nào cả.
"""

customer_info = {
    "status": "missing_info",
    "missing_fields": ["gender", "purchase_history"]
}

user_message = "Tôi muốn mua sản phẩm tăng chiều cao."
response, customer_info = chat_pipeline(user_message, chat_history, customer_info, llm)

print("AMI:", response)
print("Customer Info:", customer_info)
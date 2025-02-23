from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from knowledge import retrieve_relevant_info, tobrain
# Initialize LLM for summarization
summarization_llm = ChatOpenAI(model="gpt-4o")

def summarize_text(text):
    prompt = f"""
    Summarize the following text in a concise and structured format:
    {text}
    """
    response = summarization_llm.invoke(prompt)
    return response.content


# Sample product data
raw_text = """
1	HÀN QUỐC	"1. Giá: 270,000 KRW
2. Lộ trình: Sử dụng trong 1,5 tháng, hỗ trợ tăng chiều cao từ 2-3 cm (khoảng 1,2 inch).
3. Khuyến mãi: Mua 4 tặng 1.
4. Đối tượng phù hợp: Khách hàng muốn tăng chiều cao nhẹ trong thời gian ngắn."	"1. Giá: 350,000 KRW
2. Lộ trình: Sử dụng trong 2 tháng, hỗ trợ tăng chiều cao từ 3-4 cm (khoảng 1,57 inch).
3. Khuyến mãi: Mua 5 tặng 2.
4. Đối tượng phù hợp: Khách hàng muốn tăng chiều cao đáng kể trong thời gian ngắn."	"1. Giá: 580,000 KRW
2. Lộ trình: Sử dụng trong 4 tháng, hỗ trợ tăng chiều cao từ 4-5 cm (khoảng 2 inch).
3. Khuyến mãi: Mua 10 tặng 3.
4. Đối tượng phù hợp: Khách hàng muốn cam kết lâu dài để cải thiện chiều cao rõ rệt."	"1. Giá: 880,000 KRW
2. Lộ trình: Sử dụng trong 8 tháng, hỗ trợ tăng chiều cao từ 8-10 cm (khoảng 3,2 inch).
3. Khuyến mãi: Mua 16 tặng 4.
4. Đối tượng phù hợp: Khách hàng muốn đạt tối đa tiềm năng chiều cao với liệu trình dài hạn."	Khách hàng có thể thanh toán 100% giá trị đơn hàng trước khi giao hàng.	Khách hàng có thể đặt cọc trước 30% và thanh toán phần còn lại khi nhận hàng.	"1. Ngân hàng: KEB HANA BANK 하나
2. Số tài khoản: 13589-16008- 6307"	*Hướng dẫn: Sau khi thanh toán, vui lòng chụp ảnh biên lai và gửi lại để chúng tôi xác nhận.
		"1. Price: 270,000 KRW
2. Usage Plan: Use within 1.5 months, supports height increase from 2-3 cm (approximately 1.2 inches).
3. Promotion: Buy 4, get 1 free.
4. Recommended for: Customers who want a moderate increase in height in a short period."	"1. Price: 350,000 KRW
2. Usage Plan: Use within 2 months, supports height increase from 3-4 cm (approximately 1.57 inches).
3. Promotion: Buy 5, get 2 free.
4. Recommended for: Customers seeking a significant height increase in a short time."	"1. Price: 580,000 KRW
2. Usage Plan: Use within 4 months, supports height increase from 4-5 cm (approximately 2 inches).
3. Promotion: Buy 10, get 3 free.
4. Recommended for: Customers willing to commit to a longer course for noticeable height improvement."	"1. Price: 880,000 KRW
2. Usage Plan: Use within 8 months, supports height increase from 8-10 cm (approximately 3.2 inches).
3. Promotion: Buy 16, get 4 free.
4. Recommended for: Customers aiming for maximum height potential with a long-term commitment.
"	Customers can pay 100% of the order value before shipment.	The customer can make a 30% deposit in advance and pay the remaining amount upon delivery.	"1. Bank: KEB HANA BANK 하나
2. Account number: 13589-16008-6307
"	*After transferring the payment, please take a photo of the receipt and send it to us for confirmation.

"""

# Example usage
summary = summarize_text(raw_text)
print("Summary: ", summary)
tobrain(summary)

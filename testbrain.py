from knowledge import tobrain, retrieve_relevant_info
from summarizer import summarize_text


# Example usage
raw_text = """
CHÂU ÂU	"1. Giá: 330 EUR
2. Lộ trình: Sử dụng trong 2 tháng, hỗ trợ tăng chiều cao từ 2-3 cm (~1.2 inch).
3. Khuyến mãi: Mua 5 tặng 1.
4. Phù hợp cho: Khách hàng muốn tăng chiều cao nhẹ trong thời gian ngắn.
"	"1. Giá: 480 EUR
2. Lộ trình: Sử dụng trong 3 tháng, hỗ trợ tăng chiều cao từ 4-5 cm (~1.57 inch).
3. Khuyến mãi: Mua 7 tặng 2.
4. Phù hợp cho: Khách hàng muốn cải thiện chiều cao đáng kể trong thời gian ngắn hơn."	"1. Giá: 627 EUR
2. Lộ trình: Sử dụng trong 4 tháng, hỗ trợ tăng chiều cao từ 5-7 cm (~2 inch).
3. Khuyến mãi: Mua 9 tặng 3.
4. Phù hợp cho: Khách hàng muốn cam kết lâu dài để đạt kết quả cao nhất."	"1. Giá: 1090 EUR
2. Lộ trình: Sử dụng trong 8 tháng, hỗ trợ tăng chiều cao từ 8-10 cm (~3.2 inch).
3. Khuyến mãi: Mua 16 tặng 6.
4. Phù hợp cho: Khách hàng muốn tối ưu hóa tiềm năng chiều cao và cam kết sử dụng lâu dài."	Khách hàng có thể thanh toán 100% trước khi giao hàng.	Khách hàng có thể đặt cọc trước 30% và thanh toán phần còn lại khi nhận hàng.	"Tên tài khoản: Thi Kim Quynh Lam
IBAN: ES0700492601032817087324
SWIFT: BSCHESMM
Ngân hàng: Santander S.A
Địa chỉ ngân hàng: Rambla Nova 33, 43003, Tarragona"	*Hướng dẫn: Sau khi thanh toán, vui lòng chụp ảnh biên lai và gửi lại để chúng tôi xác nhận.
		"1. Price: 330 EUR
2. Usage Plan: Use within 2 months, supports height increase from 2-3 cm (~1.2 inch).
3. Promotion: Buy 5, get 1 free.
4. Suitable for: Customers looking for moderate height improvement in a short period.
"	"1. Price: 480 EUR
2. Usage Plan: Use within 3 months, supports height increase from 4-5 cm (~1.57 inch).
3. Promotion: Buy 7, get 2 free.
4. Suitable for: Customers seeking noticeable height improvement in less time."	"1. Price: 627 EUR
2. Usage Plan: Use within 4 months, supports height increase from 5-7 cm (~2 inch).
3. Promotion: Buy 9, get 3 free.
4. Suitable for: Customers committed to long-term use for optimal results."	"1. Price: 1090 EUR
2. Usage Plan: Use within 8 months, supports height increase from 8-10 cm (~3.2 inch).
3. Promotion: Buy 16, get 6 free.
4. Suitable for: Customers aiming to maximize their height potential with a long-term commitment.
"	Customers can pay 100% upfront before shipment.	The customer can make a 30% deposit in advance and pay the remaining amount upon delivery.	"Account Name: Thi Kim Quynh Lam
IBAN: ES0700492601032817087324
SWIFT: BSCHESMM
Bank Name: Santander S.A
Bank Address: Rambla Nova 33, 43003, Tarragona"	After completing the payment, please take a photo of the receipt and send it to us for confirmation.
"""

#summary = summarize_text(raw_text)
#print("Summary: ", summary)
#tobrain(summary,raw_text)


#tobrain("Hỗ trợ tăng chiều cao HITO", raw_text)

#query = "Mua ở đâu Paris?"
#results = retrieve_relevant_info(query)
#print(results)

query = "How to pay in Paris?"
retrieved_data = retrieve_relevant_info(query)

# Print results
print("🔍 Retrieved Data:", retrieved_data)

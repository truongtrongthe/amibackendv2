
input_text = """
Một người tò mò, ham học hỏi là người luôn có khao khát tìm hiểu, khám phá những điều mới và không ngừng nâng cao kiến thức, kỹ năng của mình. 
Họ không chấp nhận những câu trả lời hời hợt mà luôn muốn đào sâu vấn đề để hiểu rõ bản chất.

Biểu hiện của người tò mò, ham học hỏi:
🔎 1. Luôn đặt câu hỏi "Tại sao?" và "Như thế nào?"
Họ không dễ dàng chấp nhận mọi thứ theo cách nó vốn có, mà luôn muốn hiểu sâu hơn.

Ví dụ: Khi thấy một công nghệ mới, họ không chỉ hỏi “Nó hoạt động như thế nào?” mà còn hỏi “Tại sao nó lại hiệu quả hơn cái cũ?”

📖 2. Chủ động tìm kiếm kiến thức mới
Họ không chờ ai đó dạy mà tự mình khám phá, đọc sách, học hỏi từ nhiều nguồn khác nhau.

Họ thích thử nghiệm những điều mới, không ngại bước ra khỏi vùng an toàn.

🎯 3. Học từ thất bại, không ngại thử thách
Khi gặp khó khăn hoặc thất bại, họ không nản lòng mà xem đó là cơ hội để học hỏi.

Họ phân tích lỗi sai, tìm cách cải thiện và làm tốt hơn vào lần sau.

👂 4. Luôn lắng nghe và cởi mở với ý kiến mới
Họ không bảo thủ mà sẵn sàng tiếp nhận quan điểm khác, dù nó trái ngược với suy nghĩ ban đầu của họ.

🚀 5. Đặt mình vào môi trường có nhiều thách thức
Họ chủ động tham gia vào những công việc hoặc dự án khó để học hỏi nhanh hơn.

💡 Ví dụ thực tế:
Một nhân viên bán hàng ham học hỏi không chỉ làm theo kịch bản có sẵn mà sẽ:

Quan sát cách khách hàng phản ứng để điều chỉnh cách giao tiếp.

Đọc thêm về tâm lý khách hàng để hiểu nhu cầu sâu hơn.

Học hỏi từ những người bán hàng giỏi hơn để nâng cao kỹ năng của mình.

📌 Tóm lại: Người tò mò, ham học hỏi luôn muốn hiểu sâu vấn đề, không ngại khó khăn, sẵn sàng học từ bất kỳ ai và luôn tìm cách phát triển bản thân.
"""


inputs =[
     input_text,
    "Luôn luôn thể hiện mình là người tò mò trong lúc giao tiếp với người khác"
    ]

import asyncio
from pinecone_datastores import infer_categories,save_pretrain,save_to_convo_history,blend_and_rank_brain
async def main():
    for i, input_text in enumerate(inputs, start=1):
            categories = await infer_categories(input_text)
            #await save_pretrain(input_text)
            #await save_to_convo_history(input_text,"brian")
            print(f"Input: {input_text[:100]}...")  # Print a shortened preview for readability
            print(f"Categories: {categories}")

    #brain_output = await blend_and_rank_brain("Anh Minh muốn mua nhà, trả lời thế nào?", "brian",top_n_categories=5)
    #top_categories = brain_output["categories"]
    #wisdoms = brain_output["wisdoms"]
    #confidence = brain_output["confidence"]

    for cat in categories:
        if cat["english"] == "character":
            response = f"Để trả lời tốt, tôi nhớ rằng: {categories["english"]} \n"
            print(response)
asyncio.run(main())
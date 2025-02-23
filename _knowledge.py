import openai
import json
import os
# Khóa API của OpenAI (thay thế bằng khóa API thực của bạn)
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gọi OpenAI API để chuyển đổi dữ liệu
#response = client.chat.completions.create(
#    model="gpt-4-turbo",  # Sử dụng model GPT-4 Turbo
#    messages=[
#        {"role": "system", "content": "Bạn là AI giúp chuyển đổi dữ liệu thô thành JSON theo cấu trúc cụ thể."},
#        {"role": "user", "content": f"Chuyển đổi nội dung sau thành JSON. Dữ liệu đầu vào:\n{raw_text}"}
#    ],
#    response_format={"type": "json_object"}  # Cấu hình để trả về JSON
#)

# Chuyển đổi sang JSON object
#structured_data = response.choices[0].message.content
#print("data:",structured_data)

def convert_text_to_json(raw_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that extracts structured data from unstructured text."},
            {"role": "user", "content": "json: " + raw_text}
        ],
        response_format={"type": "json_object"}
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def summarize(rawtext):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that summarizes text."},
            {"role": "user", "content": "json: " + rawtext}
        ]
    )   
    return response.choices[0].message.content


# Lưu vào file JSON
#with open("products.json", "w", encoding="utf-8") as f:
#    json.dump(structured_data, f, indent=2, ensure_ascii=False)
# In kết quả
#print(json.dumps(structured_data, indent=2, ensure_ascii=False))



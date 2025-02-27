import openai


def generate_finetune_data():
    # Lấy dữ liệu từ Supabase
    training_samples = supabase_client.table("training_data").select(
        "user_input, original_response, improved_response"
    ).eq("accepted", True).execute().data

    fine_tune_data = []
    
    for sample in training_samples:
        fine_tune_data.append({
            "messages": [
                {"role": "user", "content": sample["user_input"]},
                {"role": "assistant", "content": sample["improved_response"]}
            ]
        })
    
    # Lưu thành file JSON
    with open("finetune_dataset.json", "w") as f:
        json.dump(fine_tune_data, f, ensure_ascii=False, indent=4)
    
    return "finetune_dataset.json"

def fine_tune_model():
    file_path = generate_finetune_data()

    # Upload file lên OpenAI
    upload_response = openai.File.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )

    file_id = upload_response["id"]

    # Bắt đầu quá trình fine-tune
    fine_tune_response = openai.FineTune.create(
        training_file=file_id,
        model="gpt-4o"
    )

    return fine_tune_response

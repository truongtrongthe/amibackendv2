from langchain_openai import ChatOpenAI

# 1️⃣ Khởi tạo LLM với OpenAI API (bật streaming)
llm = ChatOpenAI(model="gpt-4o", streaming=True)

def generate_response_v2(prompt):
   response = llm.stream(prompt)  # LangChain handles streaming automatically
   for chunk in response:
      print("Chunk received:", chunk)
      if chunk.content:
         yield chunk.content

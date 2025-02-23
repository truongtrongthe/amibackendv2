from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from knowledge import retrieve_relevant_info, tobrain
# Initialize LLM for summarization
summarization_llm = ChatOpenAI(model="gpt-4o")

def summarize_text(text):
    prompt = f"""
    Summarize the following text in a concise and structured format. Remember to extract the most important information and make it easy to understand.
    {text}
    """
    response = summarization_llm.invoke(prompt)
    return response.content


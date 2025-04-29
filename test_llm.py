import asyncio
from profile_helper import LLM, llm_based_profiling
from langchain.schema import HumanMessage, SystemMessage

async def test_langchain_model():
    print("Testing LangChain model directly...")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Return some JSON data about the weather for the following cities: New York, Tokyo, London")
    ]
    
    try:
        response = await LLM.ainvoke(messages)
        print(f"LangChain response: {response.content}")
        print("LangChain model working correctly!")
    except Exception as e:
        print(f"Error with LangChain model: {str(e)}")

async def test_profiling_function():
    print("Testing llm_based_profiling function...")
    try:
        result = await llm_based_profiling("Anh bị xuất tinh sớm, có cách nào giúp được không?")
        print(f"Profiling result: {result}")
        print("Profiling function working correctly!")
    except Exception as e:
        print(f"Error with profiling function: {str(e)}")

async def main():
    await test_langchain_model()
    print("\n" + "-"*50 + "\n")
    await test_profiling_function()

if __name__ == "__main__":
    asyncio.run(main()) 
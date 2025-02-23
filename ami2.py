from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import  LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o", streaming=True)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

prompt = PromptTemplate(
    input_variables=["history", "user_input"],
    template="History:\n{history}\nUser: {user_input}\nAmi:"
)
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)


def response_with_memory(query):
    # Print the current memory before generating a response
    print("Current Memory:", memory.load_memory_variables({}))
    
    # Prepare the input dictionary
    input_data = {
        "user_input": query
    }
    
    # Use the conversation.run method to handle streaming
    for chunk in conversation.run(**input_data, return_only_outputs=True):  # Unpack the input dictionary
        yield chunk  # Yield each chunk of the response
    
    # Print the updated memory after generating a response
    print("Updated Memory:", memory.load_memory_variables({}))

def ami_response(query):  # Replace with your actual new function name
    # Print the current memory before generating a response
    print("Current Memory:", memory.load_memory_variables({}))
    
    # Prepare the input dictionary
    input_data = {
        "user_input": query
    }
    
    # Use the conversation.predict method to get a single response
    response = conversation.predict(**input_data)  # Unpack the input dictionary
    
    # Print the updated memory after generating a response
    print("Updated Memory:", memory.load_memory_variables({}))
    
    yield response  # Yield the single response


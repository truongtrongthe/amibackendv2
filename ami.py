from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace with your actual API key

def generate_response(prompt):
    # Stream the response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        print("Chunk received:", chunk)  # Debugging: Log the entire chunk
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:  # Check if delta and content exist
                content = delta.content
                print("Yielding content:", content)  # Debugging: Log the content
                yield content

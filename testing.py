import spacy
class State:
    text = ""      # Alex’s input + Ami’s response
    topics = {}    # e.g., {"Pitching": "Chào hàng bằng câu hỏi"}
    history = []   # Your FAISS-stored chat history

nlp = spacy.load("en_core_web_sm")

def tag_topics(state):
    doc = nlp(input)  # Extract Alex’s input
    sales_terms = {
        "pitch": "Pitching", 
        "objection": "Objections", 
        "close": "Closing",
        "qualifying":"Qualifying"
        }
    for token in doc:
        if token.text.lower() in sales_terms:
            state.topics[sales_terms[token.text.lower()]] = state.text
    return state


# utilities.py (Final Blue Print 4.0 - Enterprise Brain, deployed March 20, 2025)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import logging

# Setup logging

#logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)  # Flask server noise
logging.getLogger("http.client").setLevel(logging.WARNING)  # HTTP requests
logging.getLogger("urllib3").setLevel(logging.WARNING)  # HTTP-related
logger = logging.getLogger(__name__)

# Config
LLM = ChatOpenAI(model="gpt-4o", streaming=True)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")


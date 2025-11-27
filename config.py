import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """
    Get secret from Streamlit secrets (cloud) or environment variables (local)
    """
    # # Try Streamlit secrets first (for Streamlit Cloud)
    # if hasattr(st, 'secrets') and key in st.secrets:
    #     return st.secrets[key]
    # # Fallback to environment variables
    return os.getenv(key, default)

# Azure Vision
AZURE_VISION_ENDPOINT = get_secret("AZURE_VISION_ENDPOINT")
AZURE_VISION_KEY = get_secret("AZURE_VISION_KEY")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")

# Azure OpenAI API Versions
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_EMBEDDINGS_API_VERSION = get_secret("AZURE_OPENAI_EMBEDDINGS_API_VERSION", "2023-05-15")

# Azure OpenAI Deployments
CHAT_DEPLOYMENT = get_secret("CHAT_DEPLOYMENT", "gpt-5-chat")  
EMBEDDING_DEPLOYMENT = get_secret("EMBEDDING_DEPLOYMENT", "text-embedding-3-small") 

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """
    Get secret from Streamlit secrets (cloud) or environment variables (local)
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    # Fallback to environment variables
    except:
        return os.getenv(key, default)

GOOGLE_APPLICATION_CREDENTIALS = "google-credentials.json"
GOOGLE_PROJECT_ID = get_secret("GOOGLE_PROJECT_ID")

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4.1"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ============================================================================
# AZURE BLOB STORAGE (for saving uploads - temporary)
# ============================================================================
AZURE_STORAGE_CONNECTION_STRING = get_secret("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = get_secret("AZURE_STORAGE_CONTAINER", "user-uploads-google")


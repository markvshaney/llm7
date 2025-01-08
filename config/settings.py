import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Settings:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

if __name__ == "__main__":
    print("Settings loaded successfully!")
    print(f"OLLAMA_BASE_URL: {Settings.OLLAMA_BASE_URL}")
    print(f"OLLAMA_MODEL: {Settings.OLLAMA_MODEL}")

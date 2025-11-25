"""
Interactive Chatbot Launcher
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv
load_dotenv()

from src.rag.langgraph_chatbot import LangGraphChatbot

if __name__ == "__main__":
    print("\nStarting IISc Research Chatbot...\n")
    chatbot = LangGraphChatbot()
    chatbot.interactive_chat()

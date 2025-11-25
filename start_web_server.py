"""
Start Web Server for IISc Research Assistant
"""

import os
import sys

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load .env
from dotenv import load_dotenv
load_dotenv()

print("\n" + "="*60)
print("IISc Research Assistant - Web Server")
print("="*60)
print("\nStarting server...")
print("\n📱 Web UI: http://localhost:8080")
print("📚 API Docs: http://localhost:8080/docs")
print("\nPress Ctrl+C to stop\n")
print("="*60 + "\n")

# Start server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)

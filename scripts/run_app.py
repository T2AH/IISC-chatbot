# c:\Users\harsh\Documents\chat application\scripts\run_app.py
import subprocess
import sys
import os

def main():
    # Activate virtual environment if not already active
    if "VIRTUAL_ENV" not in os.environ:
        venv_path = os.path.join(os.getcwd(), "venv", "Scripts", "activate")
        if os.path.exists(venv_path):
            print("Activating virtual environment...")
            subprocess.call(f'"{venv_path}"', shell=True)

    # Start FastAPI app
    api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"])

    # Start Streamlit app
    streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"])

    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        api_process.terminate()
        streamlit_process.terminate()
        print("Application stopped.")

if __name__ == "__main__":
    main()

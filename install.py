import nltk
import subprocess

def download_nltk_data():
    try:
        nltk.download('punkt_tab')
        print("Downloaded NLTK punkt_tab successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    download_nltk_data()
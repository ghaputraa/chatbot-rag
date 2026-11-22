import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY tidak ditemukan di .env")
else:
    client = Groq(api_key=api_key)
    models = client.models.list()
    print("📋 Daftar Model Groq yang Aktif untuk Akun Anda:")
    for model in models.data:
        print(f" - {model.id}")
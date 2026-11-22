FROM python:3.12-slim

WORKDIR /app

# Install compiler dasar yang dibutuhkan ChromaDB
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install PyTorch versi CPU terlebih dahulu agar ringan dan cepat (~200MB vs 2.5GB CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Salin dan install seluruh dependensi proyek
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi dan koleksi PDF
COPY . .

# Siapkan folder database dengan hak akses penuh untuk Hugging Face Spaces
RUN mkdir -p chroma_db data/raw && chmod -R 777 chroma_db data

# Jalankan server FastAPI di port standar Hugging Face (7860)
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

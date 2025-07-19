FROM python:3.10-slim

# Install system packages required for image processing and TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files to /app inside container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Start the FastAPI server (change 'main:app' to your filename:app if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

FROM python:3.10-slim

WORKDIR /app

# Install dependency OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Run script
CMD ["python", "main.py"]

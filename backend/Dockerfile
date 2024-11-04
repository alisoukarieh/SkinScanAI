# Step 1: Base image for Python
FROM python:3.10-slim

# Step 2: Install system dependencies required by FastAI and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Set a working directory
WORKDIR /app

# Step 3: Copy the requirements file to the container
COPY requirements.txt .
COPY SkinScanAI_DensNet121_88.22.pkl .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the entire Flask application to the container
COPY . .

# Step 6: Set environment variables
ENV FLASK_ENV=production

# Step 7: Expose the port Flask runs on
EXPOSE 5000

# Step 8: Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]

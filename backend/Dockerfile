# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (optional, depending on your Flask app)
# For example, if you need build tools or specific libraries
# RUN apt-get update && apt-get install -y build-essential
# dependecnies for pytorch and fastai
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py
# Set Flask to production mode
ENV FLASK_ENV=production

# Command to run the Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

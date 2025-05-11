# Use the official Python image as the base image
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    xauth \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    tk \
    libx11-6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PDF/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    build-essential \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt if you have one, else install directly
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command to run the Streamlit app
CMD ["streamlit", "run", "eval_all_pages.py", "--server.port=8501", "--server.address=0.0.0.0"]
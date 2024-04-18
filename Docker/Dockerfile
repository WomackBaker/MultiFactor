# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for deep learning libraries and audio processing
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 30080

# Run the application
CMD ["python", "voiceface.py"]

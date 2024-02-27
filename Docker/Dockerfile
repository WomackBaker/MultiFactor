# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
&& rm -rf /var/lib/apt/lists/*

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "face.py"]